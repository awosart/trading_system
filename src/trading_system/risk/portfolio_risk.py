"""Limits on how much of the account may be at stake, and in what.

Five limits, each with its own reason so a run can say which one bound:

* **portfolio** — total money at risk across everything open,
* **instrument** — per symbol,
* **cluster** — per group of instruments that amount to one bet,
* **strategy** — per strategy id,
* **direction** — per side, long against short.

All five are expressed as fractions of equity, and all five are checked the same
way: what is already at stake in that bucket, plus what this signal would add,
against the bucket's ceiling. A signal that would exceed one is **trimmed** to
the headroom rather than refused, and refused only if the trimmed figure cannot
be traded. That asymmetry with ``max_lot`` — which refuses instead of trimming —
is deliberate: reducing size to fit a risk budget is this layer performing its
function, whereas hitting the venue's maximum means the sizing has diverged from
what the instrument supports, which is worth surfacing rather than papering over.

**Cluster heat sums absolute risk, with no netting.** The correct treatment is
``sqrt(wᵀΣw)``, which needs a covariance matrix — and the whole reason clustering
falls back to manual groups is that the matrix may not exist yet. The cost is
stated plainly: a genuine hedge, long EURUSD against long USDCHF, is counted as
risk rather than as its reduction. The error is in the conservative direction,
which is the standing disposition of this module — a measurement that is missing
or uncertain must never buy more permission than one that is present.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.types import Side
from trading_system.risk.correlation import CorrelationMatrix
from trading_system.risk.models import AccountState, RiskReason

#: Cluster label for an instrument in no manual group and correlated with
#: nothing measurable. It is its own cluster: constrained by the per-instrument
#: limit, but not held against anything else. The honest limit of the fallback.
UNGROUPED = "ungrouped"


class PortfolioLimitsConfig(BaseModel):
    """Ceilings on concurrent risk, each a fraction of equity.

    Attributes:
        max_portfolio_risk_pct: Total across every open position.
        max_instrument_risk_pct: Per symbol.
        max_cluster_risk_pct: Per cluster of instruments that move together.
        max_strategy_risk_pct: Per strategy id.
        max_direction_risk_pct: Per side. A ceiling of 1.0 disables it, which is
            the default: "too long overall" is a real exposure, but on a mixed
            universe it binds constantly and is better set deliberately.
        groups: Manual clusters, name to member symbols. The floor that measured
            correlation may extend but never undo — three dollar pairs are one
            bet whether or not the window has enough history to prove it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_portfolio_risk_pct: float = Field(default=0.06, gt=0, le=1)
    max_instrument_risk_pct: float = Field(default=0.02, gt=0, le=1)
    max_cluster_risk_pct: float = Field(default=0.04, gt=0, le=1)
    max_strategy_risk_pct: float = Field(default=0.04, gt=0, le=1)
    max_direction_risk_pct: float = Field(default=1.0, gt=0, le=1)
    groups: dict[str, list[str]] = Field(default_factory=dict)


def cluster_map(
    symbols: Sequence[str],
    *,
    groups: Mapping[str, Sequence[str]],
    matrix: CorrelationMatrix | None,
    threshold: float,
) -> dict[str, str]:
    """Assign each symbol to a cluster, merging manual groups by correlation.

    Manual groups seed the clusters; a measured ``|correlation| >= threshold``
    then unions two symbols' clusters together. Union only — nothing here can
    split a manual group, so the configured prior is a floor on how constrained
    the portfolio is, and losing the matrix loses only the extra tightening.

    Merging is single-linkage: one correlated pair joins two whole clusters.
    Single-linkage is usually criticised for chaining, and here that criticism
    inverts — a chain produces a *larger* cluster and therefore a *tighter*
    limit, so the failure mode errs toward restraint.

    Args:
        symbols: Symbols to place, typically every open position plus the
            candidate.
        groups: Manual clusters, name to members.
        matrix: Measured correlations, or ``None`` to use manual groups alone.
        threshold: Absolute correlation at or above which two symbols merge.

    Returns:
        Symbol to cluster label.
    """
    label: dict[str, str] = {}
    for name, members in groups.items():
        for member in members:
            label[member] = name
    # A symbol in no manual group is its own cluster. Sharing one "ungrouped"
    # label would fuse unrelated leftovers into a single artificial bet, which
    # is a limit nobody configured.
    for symbol in symbols:
        label.setdefault(symbol, f"{UNGROUPED}:{symbol}")

    if matrix is None:
        return {symbol: label[symbol] for symbol in symbols}

    parent = {cluster: cluster for cluster in set(label.values())}

    def find(cluster: str) -> str:
        while parent[cluster] != cluster:
            parent[cluster] = parent[parent[cluster]]
            cluster = parent[cluster]
        return cluster

    for index, symbol_a in enumerate(symbols):
        for symbol_b in symbols[index + 1 :]:
            correlation = matrix.get(symbol_a, symbol_b)
            if correlation is None or abs(correlation) < threshold:
                continue
            root_a, root_b = find(label[symbol_a]), find(label[symbol_b])
            if root_a != root_b:
                # Deterministic: the alphabetically first label wins, so the
                # same universe always produces the same cluster names.
                first, second = sorted((root_a, root_b))
                parent[second] = first

    return {symbol: find(label[symbol]) for symbol in symbols}


@dataclass(frozen=True)
class LimitCheck:
    """How one limit stood when a signal was evaluated.

    Attributes:
        reason: The refusal reason this limit raises when there is no headroom.
        bucket: What the limit was applied to — a symbol, a cluster name, a
            strategy id, a side, or ``"portfolio"``.
        used: Money already at risk in that bucket.
        ceiling: The bucket's limit, in money.
    """

    reason: RiskReason
    bucket: str
    used: Decimal
    ceiling: Decimal

    @property
    def headroom(self) -> Decimal:
        """Money that may still be put at risk in this bucket, never negative.

        Clamped at zero because a bucket can be over its ceiling without anything
        being wrong: equity falls, and a limit expressed as a fraction of equity
        falls with it while the open positions do not change.
        """
        return max(Decimal(0), self.ceiling - self.used)


class PortfolioRisk:
    """Evaluates the five concurrent-risk limits for one candidate signal."""

    __slots__ = ("_config",)

    def __init__(self, config: PortfolioLimitsConfig | None = None) -> None:
        """Configure the limits.

        Args:
            config: Ceilings and manual groups. Defaults are conservative.
        """
        self._config = config if config is not None else PortfolioLimitsConfig()

    def __repr__(self) -> str:
        """Compact description naming the portfolio ceiling."""
        return f"PortfolioRisk(max_portfolio={self._config.max_portfolio_risk_pct})"

    @property
    def config(self) -> PortfolioLimitsConfig:
        """The configured ceilings and groups."""
        return self._config

    def checks(
        self,
        *,
        account: AccountState,
        symbol: str,
        strategy_id: str,
        side: Side,
        matrix: CorrelationMatrix | None,
        threshold: float,
    ) -> tuple[LimitCheck, ...]:
        """Every limit that applies to this candidate, with its current headroom.

        Args:
            account: Account snapshot, carrying the open positions.
            symbol: Instrument the candidate would trade.
            strategy_id: Strategy the candidate came from.
            side: Direction the candidate would take.
            matrix: Measured correlations, or ``None`` to cluster by manual
                groups alone.
            threshold: Absolute correlation at or above which symbols merge.

        Returns:
            One check per limit, in the order they are reported.
        """
        equity = account.equity
        open_risks = account.open_risks

        symbols = sorted({position.symbol for position in open_risks} | {symbol})
        clusters = cluster_map(
            symbols, groups=self._config.groups, matrix=matrix, threshold=threshold
        )
        candidate_cluster = clusters[symbol]

        portfolio_used = account.open_risk_amount
        instrument_used = sum((p.risk_amount for p in open_risks if p.symbol == symbol), Decimal(0))
        cluster_used = sum(
            (p.risk_amount for p in open_risks if clusters[p.symbol] == candidate_cluster),
            Decimal(0),
        )
        strategy_used = sum(
            (p.risk_amount for p in open_risks if p.strategy_id == strategy_id), Decimal(0)
        )
        direction_used = sum((p.risk_amount for p in open_risks if p.side is side), Decimal(0))

        return (
            LimitCheck(
                reason=RiskReason.PORTFOLIO_HEAT_EXCEEDED,
                bucket="portfolio",
                used=portfolio_used,
                ceiling=equity * Decimal(str(self._config.max_portfolio_risk_pct)),
            ),
            LimitCheck(
                reason=RiskReason.INSTRUMENT_LIMIT_EXCEEDED,
                bucket=symbol,
                used=instrument_used,
                ceiling=equity * Decimal(str(self._config.max_instrument_risk_pct)),
            ),
            LimitCheck(
                reason=RiskReason.CLUSTER_LIMIT_EXCEEDED,
                bucket=candidate_cluster,
                used=cluster_used,
                ceiling=equity * Decimal(str(self._config.max_cluster_risk_pct)),
            ),
            LimitCheck(
                reason=RiskReason.STRATEGY_LIMIT_EXCEEDED,
                bucket=strategy_id,
                used=strategy_used,
                ceiling=equity * Decimal(str(self._config.max_strategy_risk_pct)),
            ),
            LimitCheck(
                reason=RiskReason.DIRECTION_LIMIT_EXCEEDED,
                bucket=side.value,
                used=direction_used,
                ceiling=equity * Decimal(str(self._config.max_direction_risk_pct)),
            ),
        )


def binding_limit(checks: Sequence[LimitCheck]) -> LimitCheck:
    """The check with the least headroom — the one that actually constrains.

    Args:
        checks: Every applicable limit.

    Returns:
        The tightest. Ties resolve to the earliest in the sequence, which puts
        the portfolio-wide limit ahead of the narrower ones and makes a
        fully-exhausted account report the reason a reader most expects.
    """
    return min(checks, key=lambda check: check.headroom)
