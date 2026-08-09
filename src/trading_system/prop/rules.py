"""One prop firm's account rules, as data.

**Leverage is not here, and its absence is the design.** Margin rates per asset
class and the ceiling on total notional were closed in P10 stage 3 and live in
``configs/prop_profiles.yaml``, enforced inside
:meth:`~trading_system.risk.engine.RiskEngine.evaluate`. A rule set names its
counterpart with :attr:`PropRules.prop_profile` and restates no leverage figure
of its own: a second copy of a number is a second number, free to disagree with
the first. This file answers "what may I lose", that one answers "what may I
hold".

**The reset timezone is the firm's, and it is an IANA zone.** FTMO measures its
day at Prague midnight, The5ers at Israeli midnight, and neither coincides with
the FX market convention of 17:00 New York that a run's own
:attr:`~trading_system.backtest.config.BacktestConfig.day_origin` carries. Both
boundaries are legitimate and they are allowed to differ — what is not allowed
is a second *definition* of how an instant becomes a day label, so both go
through :func:`~trading_system.data.resample.trading_day` with different
:class:`~trading_system.data.resample.DayOrigin` values. A stored UTC offset
would be wrong for half the year in either zone.

**Every number loaded here is secondary and undated.** See ``README.md`` beside
this module: prop rulebooks are revised without notice, and a simulation built
on this file describes the rules as written, not the account anybody would
actually open.
"""

from datetime import time
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.exceptions import ValidationError
from trading_system.data.resample import DayOrigin
from trading_system.risk.margin import PropProfileLibrary

#: Where the shipped rule sets live.
DEFAULT_RULES_PATH = Path("configs/prop_rules.yaml")


class DailyLossBasis(StrEnum):
    """What the day's loss allowance is measured from.

    The two differ by exactly the floating P&L of whatever was open at the
    rollover, which is the whole of the disagreement between firms on this
    point — and it is not a rounding difference: a position carried through the
    reset one per cent in the red starts the new day having already spent a
    fifth of a five-per-cent allowance under one reading and nothing at all
    under the other.
    """

    #: Closed equity at the reset. Floating P&L carried through does not count
    #: against the new day's allowance.
    BALANCE_AT_DAY_START = "balance_at_day_start"

    #: Equity at the reset, floating included.
    EQUITY = "equity"


class TotalLossBasis(StrEnum):
    """Where the account's hard floor sits."""

    #: A fixed fraction below the starting balance. The floor never moves, so
    #: profit is a genuine buffer against later losses.
    STATIC = "static"

    #: The same fraction below the highest equity ever reached. The floor
    #: ratchets up with every new peak and never falls, so profit buys no
    #: buffer at all — it moves the floor up behind it.
    TRAILING_HIGH_WATER = "trailing_high_water"


class PropRules(BaseModel):
    """One firm plan's loss limits, target and consistency requirement.

    Attributes:
        name: Key the rule set is selected by.
        prop_profile: Name of the entry in ``configs/prop_profiles.yaml``
            carrying this plan's leverage. A reference rather than a copy —
            see the module docstring.
        source: Where these numbers were read, verbatim enough to re-check.
        account_size: Starting balance the percentages are taken of.
        profit_target_pct: Gain over ``account_size`` that passes the account.
        max_daily_loss_pct: Fraction of :attr:`daily_loss_basis` that may be
            lost inside one firm-day before trading stops until the next reset.
        daily_loss_basis: What that fraction is measured from.
        daily_reset_time: Local wall-clock time the firm's day rolls over.
        daily_reset_tz: IANA zone that time is expressed in — **the firm's**,
            never the run's and never the machine's.
        max_total_loss_pct: Fraction below :attr:`total_loss_basis` at which
            the account is gone.
        total_loss_basis: Whether that floor is fixed or trails the high-water
            mark.
        max_single_day_profit_share: Largest share of total profit one firm-day
            may contribute. ``None`` for a plan with no consistency rule —
            required with no default, so "this plan has none" cannot be
            confused with "nobody looked".
        min_trading_days: Days with at least one closed trade required before
            the target counts as met. Not a veto and not a post-hoc check: it
            modifies when an episode is allowed to stop, see
            :mod:`trading_system.prop.simulator`.
        notes: Anything a reader needs to know about the plan. Read by nobody.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    prop_profile: str = Field(min_length=1)
    source: str = Field(min_length=1)
    account_size: Decimal = Field(gt=0)
    profit_target_pct: float = Field(gt=0, le=10)
    max_daily_loss_pct: float = Field(gt=0, lt=1)
    daily_loss_basis: DailyLossBasis
    daily_reset_time: time
    daily_reset_tz: str = Field(min_length=1)
    max_total_loss_pct: float = Field(gt=0, lt=1)
    total_loss_basis: TotalLossBasis
    max_single_day_profit_share: float | None = Field(ge=0, le=1)
    min_trading_days: int = Field(ge=0)
    notes: str = ""

    @pydantic.model_validator(mode="after")
    def _limits_are_ordered(self) -> "PropRules":
        """Reject a daily limit that cannot bind before the total one does.

        Returns:
            The validated rules.

        Raises:
            ValueError: If the daily allowance is at or above the total one. A
                daily limit no smaller than the account's whole floor can never
                stop trading before the account is already gone, which makes it
                a field that reads as a protection and is not one.
        """
        if self.max_daily_loss_pct >= self.max_total_loss_pct:
            raise ValueError(
                f"{self.name}: max_daily_loss_pct {self.max_daily_loss_pct} is not below "
                f"max_total_loss_pct {self.max_total_loss_pct}; a daily limit that cannot "
                "bind before the account floor does is not a limit"
            )
        return self

    @property
    def day_origin(self) -> DayOrigin:
        """The firm's day boundary, in the one shape the rest of the system uses.

        Returns:
            The origin. Built here rather than stored as one so that the YAML
            keeps two readable scalar fields instead of a nested object, while
            every consumer still receives the same type
            :func:`~trading_system.data.resample.trading_day` takes.
        """
        return DayOrigin(tz=self.daily_reset_tz, at=self.daily_reset_time)

    @property
    def profit_target_amount(self) -> Decimal:
        """Equity at which the target is met, in account currency."""
        return self.account_size * (1 + Decimal(str(self.profit_target_pct)))

    def resolve_profile(self, profiles: PropProfileLibrary) -> str:
        """Check that this plan's leverage profile exists, and name it.

        Args:
            profiles: The loaded profile library.

        Returns:
            The profile name, once proven resolvable.

        Raises:
            ValidationError: If no such profile — raised rather than deferred,
                because a rule set pointing at a profile nobody has is a rule
                set whose leverage half is silently absent.
        """
        profiles.get(self.prop_profile)
        return self.prop_profile


class PropRuleLibrary(BaseModel):
    """Every rule set read into the repository.

    Attributes:
        source: Provenance of the file as a whole.
        rules: The rule sets, keyed by :attr:`PropRules.name`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    rules: tuple[PropRules, ...] = Field(min_length=1)

    def get(self, name: str) -> PropRules:
        """Look one rule set up by name.

        Args:
            name: The rule set's ``name``.

        Returns:
            The rules.

        Raises:
            ValidationError: If no rule set carries that name. Named rather
                than returning ``None``: a mistyped plan silently falling back
                to no rules at all is a run that reports having passed limits
                it never applied.
        """
        for item in self.rules:
            if item.name == name:
                return item
        available = ", ".join(sorted(item.name for item in self.rules))
        raise ValidationError(f"no prop rules named {name!r}; known rule sets: {available}")

    @property
    def names(self) -> tuple[str, ...]:
        """Every rule-set name, in file order."""
        return tuple(item.name for item in self.rules)


def load_prop_rules(path: Path = DEFAULT_RULES_PATH) -> PropRuleLibrary:
    """Load and validate the prop-firm rule sets from a YAML file.

    Args:
        path: Location of the YAML file.

    Returns:
        The library.

    Raises:
        ValidationError: If the file cannot be read, is not a mapping, fails
            validation, or declares the same rule-set name twice.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"prop rules {path}: cannot be read: {error}") from error

    loaded: Any = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValidationError(
            f"prop rules {path}: expected a mapping with 'source' and 'rules' keys, "
            f"got {type(loaded).__name__}"
        )

    try:
        library = PropRuleLibrary.model_validate(loaded)
    except pydantic.ValidationError as error:
        raise ValidationError(f"prop rules {path}: {error}") from error

    seen: set[str] = set()
    for item in library.rules:
        if item.name in seen:
            raise ValidationError(
                f"prop rules {path}: {item.name} is declared twice; a duplicate name makes "
                "which set of limits applies a matter of file order"
            )
        seen.add(item.name)
    return library


def day_origin_divergence(rules: PropRules, run_origin: DayOrigin) -> str | None:
    """A sentence naming the gap between the firm's day and the run's, or ``None``.

    Both boundaries are legitimate and they are allowed to differ — the firm
    measures its loss allowance in its own city, the market data is bucketed by
    the convention of the instrument. What must not happen is the difference
    going unnoticed, because it changes which trades count against which day's
    limit while every number on the page still looks ordinary. A log line alone
    is not enough: logs are read when something has already gone wrong, and
    this needs to be read when nothing has.

    Args:
        rules: The firm's rules.
        run_origin: The run's own data-bucketing origin.

    Returns:
        The sentence, or ``None`` when the two agree exactly.
    """
    firm = rules.day_origin
    if firm == run_origin:
        return None
    return (
        f"The firm's day resets at {firm.at} {firm.tz}; this run buckets market data at "
        f"{run_origin.at} {run_origin.tz}. Both are legitimate and they are not the same "
        "boundary: a trade closing between them counts against one day for the daily loss "
        "limit and the other for every metric grouped by the equity curve's own day label."
    )
