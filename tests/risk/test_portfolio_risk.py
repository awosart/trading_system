"""Concurrent-risk limits and the clustering that feeds the cluster one."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_system.core.types import Side
from trading_system.risk.correlation import CorrelationMatrix
from trading_system.risk.models import AccountState, OpenRisk, RiskReason
from trading_system.risk.portfolio_risk import (
    UNGROUPED,
    PortfolioLimitsConfig,
    PortfolioRisk,
    binding_limit,
    cluster_map,
)

NOW = datetime(2024, 3, 5, 12, 0, tzinfo=UTC)
DOLLAR_PAIRS = {"usd_shorts": ["EURUSD", "GBPUSD", "AUDUSD"]}


def matrix(**pairs: float) -> CorrelationMatrix:
    """Build a matrix from ``AB_CD=0.9`` style keyword pairs."""
    values: dict[tuple[str, str], float] = {}
    observations: dict[tuple[str, str], int] = {}
    for key, value in pairs.items():
        left, right = key.split("__")
        values[(left, right)] = value
        values[(right, left)] = value
        observations[(left, right)] = 60
        observations[(right, left)] = 60
    return CorrelationMatrix(as_of=NOW, values=values, observations=observations)


def account(*positions: OpenRisk, equity: str = "100000") -> AccountState:
    """A USD account holding ``positions``."""
    return AccountState(
        currency="USD",
        balance=Decimal(equity),
        equity=Decimal(equity),
        as_of=NOW,
        open_risks=positions,
    )


def position(symbol: str, risk: str, *, strategy: str = "s1", side: Side = Side.BUY) -> OpenRisk:
    """One open position risking ``risk``."""
    return OpenRisk(symbol=symbol, strategy_id=strategy, side=side, risk_amount=Decimal(risk))


class TestClustering:
    def test_manual_groups_apply_with_no_matrix_at_all(self) -> None:
        # The prior does not need history to be true: three dollar pairs are
        # one bet whether or not anything has been measured.
        clusters = cluster_map(
            ["EURUSD", "GBPUSD", "AUDUSD"], groups=DOLLAR_PAIRS, matrix=None, threshold=0.7
        )
        assert len(set(clusters.values())) == 1

    def test_an_ungrouped_symbol_is_its_own_cluster(self) -> None:
        # Not a shared "ungrouped" bucket: that would fuse unrelated leftovers
        # into one artificial bet nobody configured.
        clusters = cluster_map(["XAUUSD", "BTCUSD"], groups={}, matrix=None, threshold=0.7)
        assert clusters["XAUUSD"] != clusters["BTCUSD"]
        assert all(label.startswith(UNGROUPED) for label in clusters.values())

    def test_correlation_merges_two_manual_groups(self) -> None:
        clusters = cluster_map(
            ["EURUSD", "NAS100"],
            groups={"fx": ["EURUSD"], "indices": ["NAS100"]},
            matrix=matrix(EURUSD__NAS100=0.85),
            threshold=0.7,
        )
        assert clusters["EURUSD"] == clusters["NAS100"]

    def test_a_negative_correlation_merges_too(self) -> None:
        # -0.9 is one bet with a leg inverted, not two independent ones.
        clusters = cluster_map(
            ["EURUSD", "USDCHF"],
            groups={},
            matrix=matrix(EURUSD__USDCHF=-0.9),
            threshold=0.7,
        )
        assert clusters["EURUSD"] == clusters["USDCHF"]

    def test_correlation_below_the_threshold_leaves_them_apart(self) -> None:
        clusters = cluster_map(
            ["EURUSD", "NAS100"], groups={}, matrix=matrix(EURUSD__NAS100=0.3), threshold=0.7
        )
        assert clusters["EURUSD"] != clusters["NAS100"]

    def test_correlation_can_never_split_a_manual_group(self) -> None:
        # The prior is a floor. Measuring EURUSD and GBPUSD at 0.1 does not buy
        # permission to treat them as two bets -- losing the matrix must only
        # ever lose the extra tightening, never the baseline.
        clusters = cluster_map(
            ["EURUSD", "GBPUSD"],
            groups=DOLLAR_PAIRS,
            matrix=matrix(EURUSD__GBPUSD=0.1),
            threshold=0.7,
        )
        assert clusters["EURUSD"] == clusters["GBPUSD"]

    def test_merging_chains_which_errs_toward_more_restriction(self) -> None:
        # Single-linkage: A~B and B~C puts all three together even though A and
        # C were never measured against each other. The usual criticism of
        # chaining inverts here -- a larger cluster is a tighter limit.
        clusters = cluster_map(
            ["A", "B", "C"],
            groups={},
            matrix=matrix(A__B=0.9, B__C=0.9),
            threshold=0.7,
        )
        assert len(set(clusters.values())) == 1

    def test_cluster_labels_are_deterministic(self) -> None:
        first = cluster_map(["A", "B"], groups={}, matrix=matrix(A__B=0.9), threshold=0.7)
        second = cluster_map(["B", "A"], groups={}, matrix=matrix(A__B=0.9), threshold=0.7)
        assert first == second


class TestLimits:
    @pytest.fixture
    def portfolio(self) -> PortfolioRisk:
        return PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=0.06,
                max_instrument_risk_pct=0.02,
                max_cluster_risk_pct=0.03,
                max_strategy_risk_pct=0.04,
                max_direction_risk_pct=0.05,
                groups=DOLLAR_PAIRS,
            )
        )

    def checks_for(
        self, portfolio: PortfolioRisk, state: AccountState, symbol: str = "EURUSD"
    ) -> dict[RiskReason, Decimal]:
        """Headroom by reason, for a candidate on ``symbol``."""
        checks = portfolio.checks(
            account=state,
            symbol=symbol,
            strategy_id="s1",
            side=Side.BUY,
            matrix=None,
            threshold=0.7,
        )
        return {check.reason: check.headroom for check in checks}

    def test_an_empty_account_has_the_full_headroom_everywhere(
        self, portfolio: PortfolioRisk
    ) -> None:
        headroom = self.checks_for(portfolio, account())
        assert headroom[RiskReason.PORTFOLIO_HEAT_EXCEEDED] == Decimal("6000.00")
        assert headroom[RiskReason.INSTRUMENT_LIMIT_EXCEEDED] == Decimal("2000.00")
        assert headroom[RiskReason.CLUSTER_LIMIT_EXCEEDED] == Decimal("3000.00")

    def test_an_open_position_consumes_its_own_instruments_headroom(
        self, portfolio: PortfolioRisk
    ) -> None:
        headroom = self.checks_for(portfolio, account(position("EURUSD", "500")))
        assert headroom[RiskReason.INSTRUMENT_LIMIT_EXCEEDED] == Decimal("1500.00")

    def test_a_group_member_consumes_the_clusters_headroom(self, portfolio: PortfolioRisk) -> None:
        # GBPUSD is not EURUSD, so the instrument limit is untouched -- but they
        # are the same bet, so the cluster limit is not.
        headroom = self.checks_for(portfolio, account(position("GBPUSD", "1000")))
        assert headroom[RiskReason.INSTRUMENT_LIMIT_EXCEEDED] == Decimal("2000.00")
        assert headroom[RiskReason.CLUSTER_LIMIT_EXCEEDED] == Decimal("2000.00")

    def test_an_unrelated_instrument_touches_only_the_portfolio_limit(
        self, portfolio: PortfolioRisk
    ) -> None:
        headroom = self.checks_for(portfolio, account(position("XAUUSD", "1000")))
        assert headroom[RiskReason.CLUSTER_LIMIT_EXCEEDED] == Decimal("3000.00")
        assert headroom[RiskReason.PORTFOLIO_HEAT_EXCEEDED] == Decimal("5000.00")

    def test_the_direction_limit_counts_only_the_matching_side(
        self, portfolio: PortfolioRisk
    ) -> None:
        state = account(
            position("XAUUSD", "1000", side=Side.BUY),
            position("NAS100", "1000", side=Side.SELL),
        )
        headroom = self.checks_for(portfolio, state)
        assert headroom[RiskReason.DIRECTION_LIMIT_EXCEEDED] == Decimal("4000.00")

    def test_headroom_never_goes_negative(self, portfolio: PortfolioRisk) -> None:
        # Equity can fall while positions stay open, so a bucket can sit over a
        # ceiling expressed as a fraction of it without anything being wrong.
        state = account(position("EURUSD", "5000"), equity="10000")
        headroom = self.checks_for(portfolio, state)
        assert headroom[RiskReason.INSTRUMENT_LIMIT_EXCEEDED] == Decimal(0)

    def test_the_binding_limit_is_the_tightest_one(self, portfolio: PortfolioRisk) -> None:
        checks = portfolio.checks(
            account=account(position("EURUSD", "1800")),
            symbol="EURUSD",
            strategy_id="s1",
            side=Side.BUY,
            matrix=None,
            threshold=0.7,
        )
        assert binding_limit(checks).reason is RiskReason.INSTRUMENT_LIMIT_EXCEEDED
        assert binding_limit(checks).headroom == Decimal("200.00")


class TestOpenRiskTotal:
    def test_the_total_is_derived_not_stored(self) -> None:
        # Two fields holding the same quantity drift the moment one is updated
        # and the other is not.
        state = account(position("EURUSD", "500"), position("XAUUSD", "250"))
        assert state.open_risk_amount == Decimal("750")

    def test_with_opened_appends_and_leaves_equity_alone(self) -> None:
        state = account(position("EURUSD", "500"))
        after = state.with_opened(
            "XAUUSD", "s2", Side.SELL, Decimal("300"), margin=Decimal(0), notional=Decimal(0)
        )
        assert after.open_risk_amount == Decimal("800")
        assert after.equity == state.equity
        # And the original is untouched: it is a value, not a mutable ledger.
        assert state.open_risk_amount == Decimal("500")

    def test_a_position_risking_nothing_is_allowed(self) -> None:
        # A trailing stop past breakeven genuinely risks nothing, and charging
        # the portfolio for it would reserve headroom against a risk that is
        # already gone.
        state = account(position("EURUSD", "0"))
        assert state.open_risk_amount == Decimal(0)

    def test_a_negative_risk_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            OpenRisk(symbol="EURUSD", strategy_id="s", side=Side.BUY, risk_amount=Decimal("-1"))
