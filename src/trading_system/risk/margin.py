"""What the broker locks up, and the ceiling the firm puts on top of it.

``InstrumentSpec.margin_rate`` was declared in P10 stage 1 and read by nothing
until this module. Until then the system could approve a position the broker
would physically decline to open, which is not a conservative error in either
direction: the backtest reports a trade that could not have happened, and every
figure derived from it is a figure about a different account.

**Two rules, deliberately two refusals.** They answer different questions and a
run that conflates them cannot say why a trade did not happen:

* **Margin** is a broker mechanic. ``contract_size * price * size *
  margin_rate``, converted to account currency, must fit inside
  ``equity - used_margin``. Failing it means *the money is not there*, and it is
  checked on every run because a margin requirement always exists — the field is
  mandatory on :class:`~trading_system.core.instruments.InstrumentSpec`.
* **The leverage cap** is a firm rule: an upper bound on *total notional*,
  expressed as a multiple of equity. Failing it means *a rule was broken* while
  the money was there. It is opt-in — most firms publish no such figure — and
  lives on a :class:`PropProfile`.

**A third kind of rule exists and is NOT implemented here.** Some firms cap
*margin utilisation* — "max 40% margin rule", "your margin must be +150%" —
rather than notional. At one single leverage the two are the same rule written
twice, which is exactly why it is worth stating that they are not the same rule
in general. Take two accounts each holding **$1,000,000 of total notional**:

* all of it gold, at the profile's metals leverage of 1:33 → margin used
  ``1,000,000 / 33 = $30,303``;
* all of it an index, at that same profile's index leverage of 1:25 → margin
  used ``1,000,000 / 25 = $40,000``.

A notional cap of 10x equity on a $100k account passes both. A "margin used
must stay under 35% of equity" rule passes the first and refuses the second.
The rules order portfolios differently the moment the portfolio holds more than
one asset class, so a reader who finds "not implemented" here must not conclude
"already covered by the cap". Implementing it means a second ceiling compared
against ``AccountState.used_margin`` rather than ``used_notional``; the reason
it is absent is that no profile in ``configs/prop_profiles.yaml`` currently
carries such a figure, not that it would be redundant.

**Where the check happens, and the gap that leaves.** In
:meth:`~trading_system.risk.engine.RiskEngine.evaluate`, after the size has been
quantised to whole lot steps and bounded, because the requirement is computed
from the size that will actually be traded — quantisation only ever rounds down,
so checking the unquantised quotient would refuse trades that do in fact fit, on
a number no run ever trades. That places the check at the moment of *decision*.
The fill happens at the next bar's open, so an order approved against one
snapshot is consumed against another, and a real broker rejects at the fill.

There is no second check at fill time, and the size of what that concedes is
measured rather than asserted. Across the four recorded walk-forwards (EURUSD,
1% risk, FX leverage 1:30) peak margin utilisation is 16%, 33%, 16% and **63%**,
the last being ``ema-pullback-h1``; the superseded parameter selection for that
same strategy peaked at **81%**, which is the high-water mark anything in this
repository has produced. So the tightest run still opens its worst position with
roughly **a fifth of equity in free margin to spare**, and no ordering of
decision against fill inside that band turns an approval into a rejection. The
criterion for revisiting is stated so it does not have to be rediscovered:
**when peak utilisation approaches 95%**, the gap between the snapshot and the
fill starts to decide trades, and the check has to be repeated where the fill
happens.

**What the check does bite is the search, not the trades.** Enabling it left
three of the four walk-forwards bit-for-bit identical and every out-of-sample
run at zero margin refusals — and still moved ``ema-pullback-h1`` from +0.0553
to −0.0650 stitched expectancy. 52 of its 4400 in-sample trials changed, all of
them at the smallest ATR stop multiple in the search space, where a narrow stop
buys a large size and therefore a large notional. Two folds then selected
different parameters. A constraint invisible on the trades a run reports can
still decide which parameters produced them.
"""

from collections.abc import Mapping
from decimal import ROUND_UP, Decimal, localcontext
from pathlib import Path
from typing import Any

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import InstrumentClass, InstrumentSpec

#: Step money figures in this module are quantised to, always **upwards**.
#:
#: Upwards, unlike every other quantisation in this layer, and the asymmetry is
#: the point: a margin requirement rounded down is a requirement the account
#: might not actually meet, which is the one error the check exists to prevent.
#: The rest of :mod:`trading_system.risk` rounds down because it is quantising
#: what the trader *gets*; this quantises what the trader *owes*.
MARGIN_STEP = Decimal("0.01")


def quantise_up(amount: Decimal) -> Decimal:
    """Round a money figure up to :data:`MARGIN_STEP`.

    Args:
        amount: The unrounded figure, non-negative.

    Returns:
        The figure at or above ``amount``, on the cent.
    """
    with localcontext() as context:
        context.rounding = ROUND_UP
        return amount.quantize(MARGIN_STEP)


class PropProfile(BaseModel):
    """One firm's account rules, as far as they bear on position size.

    A profile **overrides** leverage per asset class; it does not replace the
    instrument registry. A class the profile does not name keeps the broker
    figure already declared on the instrument, because the registry is the
    declared authority on what an instrument is and a profile is a firm's
    amendment to it. That is why a missing class is not an error: FXIFY
    publishes no metals figure, and inventing one would be worse than deferring
    to the file that does carry one.

    Attributes:
        name: Key the profile is selected by.
        source: Where these numbers were read, verbatim enough to re-check.
        leverage: Maximum leverage per asset class, as the firm publishes it —
            ``30.0`` means 1:30 and therefore a margin rate of 1/30. Leverage
            rather than the rate because that is the form every firm publishes,
            and translating at the point of reading is one place to make the
            reciprocal mistake instead of one per entry.
        leverage_cap: Ceiling on total notional as a multiple of equity, or
            ``None`` when the firm publishes no such rule. **Required, with no
            default**: a profile that does not say whether the firm caps
            exposure is a profile nobody has finished reading, and a default of
            ``None`` would make "checked, there is none" indistinguishable from
            "did not look".
        notes: Anything about the plan a reader needs in order to know whether
            it is the right one — free text, read by nobody.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    leverage: Mapping[InstrumentClass, float]
    leverage_cap: float | None
    notes: str = ""

    @model_validator(mode="after")
    def _check_bounds(self) -> "PropProfile":
        """Reject leverage figures that cannot mean what they say.

        Returns:
            The validated profile.

        Raises:
            ValueError: If any leverage is not positive, or the cap is not.
        """
        for asset_class, leverage in self.leverage.items():
            if leverage <= 0:
                raise ValueError(
                    f"{self.name}: {asset_class.value} leverage must be positive, got {leverage}"
                )
        if self.leverage_cap is not None and self.leverage_cap <= 0:
            raise ValueError(
                f"{self.name}: leverage_cap must be positive when set, got {self.leverage_cap}"
            )
        return self

    def margin_rate(self, asset_class: InstrumentClass) -> Decimal | None:
        """The fraction of notional this profile requires as margin.

        Args:
            asset_class: Class to look up.

        Returns:
            ``1 / leverage`` for a class this profile names, ``None`` for one it
            does not — the caller then keeps the instrument's own rate.
        """
        leverage = self.leverage.get(asset_class)
        if leverage is None:
            return None
        return Decimal(1) / Decimal(str(leverage))


class PropProfileLibrary(BaseModel):
    """Every profile that has been read into the repository.

    Attributes:
        source: Provenance of the file as a whole, repeated into every error so
            a number that looks wrong names where it came from.
        profiles: The profiles, keyed by :attr:`PropProfile.name`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    profiles: tuple[PropProfile, ...] = Field(min_length=1)

    def get(self, name: str) -> PropProfile:
        """Look one profile up by name.

        Args:
            name: The profile's ``name``.

        Returns:
            The profile.

        Raises:
            ValidationError: If no profile carries that name. Named rather than
                returning ``None``: a mistyped profile name silently falling
                back to the broker's own rates is a run whose firm rules were
                never applied, reported as a run that passed them.
        """
        for profile in self.profiles:
            if profile.name == name:
                return profile
        available = ", ".join(sorted(profile.name for profile in self.profiles))
        raise ValidationError(f"no prop profile named {name!r}; known profiles: {available}")

    @property
    def names(self) -> tuple[str, ...]:
        """Every profile name, in file order."""
        return tuple(profile.name for profile in self.profiles)


def load_prop_profiles(path: Path) -> PropProfileLibrary:
    """Load and validate the prop-firm profiles from a YAML file.

    Args:
        path: Location of the YAML file.

    Returns:
        The library.

    Raises:
        ValidationError: If the file cannot be read, is not a mapping, fails
            validation, or declares the same profile name twice.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"prop profiles {path}: cannot be read: {error}") from error

    loaded: Any = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValidationError(
            f"prop profiles {path}: expected a mapping with 'source' and 'profiles' keys, "
            f"got {type(loaded).__name__}"
        )

    try:
        library = PropProfileLibrary.model_validate(loaded)
    except pydantic.ValidationError as error:
        raise ValidationError(f"prop profiles {path}: {error}") from error

    seen: set[str] = set()
    for profile in library.profiles:
        if profile.name in seen:
            raise ValidationError(
                f"prop profiles {path}: {profile.name} is declared twice; a duplicate name "
                "makes which set of rules applies a matter of file order"
            )
        seen.add(profile.name)
    return library


def margin_rate_for(instrument: InstrumentSpec, profile: PropProfile | None) -> Decimal:
    """The margin rate one instrument is sized against on this run.

    Args:
        instrument: Contract specification, carrying the broker's own rate.
        profile: Firm profile overriding it per asset class, or ``None`` for a
            run stating it trades under the broker's published rates alone.

    Returns:
        The fraction of notional that must be posted as margin.
    """
    if profile is not None:
        override = profile.margin_rate(instrument.asset_class)
        if override is not None:
            return override
    return Decimal(str(instrument.margin_rate))


def notional_per_lot(instrument: InstrumentSpec, price: float, fx_rate: Decimal) -> Decimal:
    """Exposure of one lot, in account currency.

    ``contract_size * price`` is the notional in the instrument's **quote**
    currency for every class in the registry, and this is not a coincidence to
    be re-derived per class: an FX lot is 100 000 units of base priced in quote,
    a gold lot is 100 ounces priced in quote, an index CFD is a per-point
    multiplier times a price in points. One expression, converted once.

    Args:
        instrument: Contract specification.
        price: Price the position is opened at, in the instrument's own units.
        fx_rate: Quote-to-account rate, frozen at the moment of the decision.

    Returns:
        Money one lot puts on the market, account currency.
    """
    return instrument.contract_size * Decimal(str(price)) * fx_rate
