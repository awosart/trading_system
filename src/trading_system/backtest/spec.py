"""A run described as data, so it can be hashed, varied and shipped to a worker.

:class:`~trading_system.backtest.orchestrator.Orchestrator` takes assembled
engines — a :class:`~trading_system.risk.engine.RiskEngine` that accumulates
counters, a :class:`~trading_system.execution.costs.CostModel` that accumulates
statistics — which is right for running one backtest and wrong for everything
around it. Three things need a run *before* it exists: the run id has to digest
what it was given, a walk-forward or a sweep has to vary one field of it, and a
process pool has to pickle it. All three want the description, not the machine.

So this holds the description and knows how to build the machine from it, once,
in :meth:`RunInputs.orchestrator`. Every field is either a pydantic model or a
small parameter object, which is what makes the whole thing picklable and what
lets :meth:`RunInputs.manifest` describe it without the caller having to
enumerate its own wiring — the omission that would otherwise give two different
runs one id.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import (
    BacktestResult,
    Orchestrator,
    StrategyBinding,
    derive_run_calendar,
)
from trading_system.backtest.reproducibility import CodeVersion, RunManifest
from trading_system.core.instruments import InstrumentRegistry
from trading_system.data.models import OHLCVFrame
from trading_system.execution.config import CostConfig
from trading_system.execution.costs import CostModel
from trading_system.prop.guard import PropGuard
from trading_system.prop.rules import PropRules
from trading_system.risk.circuit_breakers import CircuitBreakerConfig, CircuitBreakers
from trading_system.risk.conversion import FxConverter, SameCurrencyConverter
from trading_system.risk.engine import RiskEngine, RiskEngineConfig
from trading_system.risk.portfolio_risk import PortfolioLimitsConfig, PortfolioRisk
from trading_system.risk.sizing.base import SizingMethod


@dataclass(frozen=True)
class RunInputs:
    """Everything one backtest is wired from.

    Attributes:
        config: The run's own parameters.
        streams: Bars per instrument and timeframe.
        bindings: Strategies, their exit presets and the streams they trade.
        instruments: Contract specifications.
        costs: Spread, slippage, commission, financing, and the run seed.
        sizing: How much money goes at stake per trade.
        converter: Where quote-to-account FX rates come from. Same-currency by
            default, which is a *statement* that every instrument in the run is
            quoted in the account currency — the run refuses to price a fill it
            cannot convert rather than assuming parity.
        risk: Engine-wide risk settings.
        limits: Concurrent-risk ceilings.
        breakers: When trading stops altogether.
        prop_rules: The firm's account rules, or ``None`` for a run on nobody's
            prop account. A rule *set* rather than an assembled
            :class:`~trading_system.prop.guard.PropGuard`, for the same reason
            every other field here is a config rather than an engine: the guard
            accumulates counters over one run, and a description has to be
            hashable, picklable and varyable before the machine exists.
    """

    config: BacktestConfig
    streams: Mapping[StreamKey, OHLCVFrame]
    bindings: tuple[StrategyBinding, ...]
    instruments: InstrumentRegistry
    costs: CostConfig
    sizing: SizingMethod
    converter: FxConverter = field(default_factory=SameCurrencyConverter)
    risk: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    limits: PortfolioLimitsConfig = field(default_factory=PortfolioLimitsConfig)
    breakers: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    prop_rules: PropRules | None = None

    def orchestrator(self) -> Orchestrator:
        """Assemble the run this describes.

        Returns:
            A fresh orchestrator. Fresh every time: the engines it holds
            accumulate counters over one run, so a second run through the same
            instance would report the first one's as well.
        """
        breakers = self.breakers
        if breakers.calendar is None:
            # Same derivation Orchestrator applies to Portfolio, applied here
            # too: a run built through RunInputs always gets a real calendar
            # when its instruments make one unambiguous. `None` stays
            # reachable only by constructing CircuitBreakerConfig directly,
            # not through this assembly point — an explicit calendar the
            # caller already set is left alone rather than overridden.
            breakers = breakers.model_copy(
                update={"calendar": derive_run_calendar(self.streams, self.instruments)}
            )
        return Orchestrator(
            config=self.config,
            streams=self.streams,
            bindings=self.bindings,
            instruments=self.instruments,
            risk_engine=RiskEngine(
                instruments=self.instruments,
                sizing=self.sizing,
                converter=self.converter,
                config=self.risk,
                portfolio=PortfolioRisk(self.limits),
                breakers=CircuitBreakers(breakers),
            ),
            cost_model=CostModel(
                {key.symbol: self.instruments[key.symbol] for key in self.streams}, self.costs
            ),
            converter=self.converter,
            run_seed=self.costs.run_seed,
            prop_guard=PropGuard(self.prop_rules) if self.prop_rules is not None else None,
        )

    def run(self) -> BacktestResult:
        """Walk the whole run and report what it produced.

        Returns:
            The equity curve, the trades and every counter the run kept.
        """
        return self.orchestrator().run()

    def components(self) -> dict[str, object]:
        """The named components a manifest digests beyond bars and strategies.

        Returns:
            Every field that is not already a first-class part of
            :class:`~trading_system.backtest.reproducibility.RunManifest`. Built
            here rather than at the call site so that adding a field to this
            class and forgetting to put it in the run id is one mistake instead
            of two independent ones.
        """
        return {
            "costs": self.costs,
            "sizing": self.sizing,
            "converter": self.converter,
            "risk": self.risk,
            "limits": self.limits,
            "breakers": self.breakers,
            "prop_rules": self.prop_rules,
        }

    def manifest(self, *, code: CodeVersion | None = None) -> RunManifest:
        """Digest these inputs into the run's identity.

        Args:
            code: Code version to record. Measured from the package on disk when
                not given.

        Returns:
            The manifest, whose ``run_id`` names this run.
        """
        return RunManifest.build(
            config=self.config,
            streams=self.streams,
            bindings=self.bindings,
            instruments=self.instruments,
            seed=self.costs.run_seed,
            components=self.components(),
            code=code,
        )

    def with_streams(self, streams: Mapping[StreamKey, OHLCVFrame]) -> "RunInputs":
        """The same run over different bars.

        Args:
            streams: Bars to run instead.

        Returns:
            A copy. What truncation, poisoning and walk-forward all vary.
        """
        return replace(self, streams=dict(streams))

    def with_bindings(self, bindings: Sequence[StrategyBinding]) -> "RunInputs":
        """The same wiring with different strategies.

        Args:
            bindings: Strategies and the streams they trade.

        Returns:
            A copy.
        """
        return replace(self, bindings=tuple(bindings))
