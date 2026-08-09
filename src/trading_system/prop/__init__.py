"""Prop Guard: the firm's own account rules, as the last veto before execution.

Margin and the leverage ceiling are **not** here — they were closed in P10
stage 3 and are enforced inside
:meth:`~trading_system.risk.engine.RiskEngine.evaluate`. This package adds the
drawdown limits, the profit target and the consistency requirement that a prop
firm layers on top, plus a simulator that asks what a finished strategy's
chances of passing actually are.
"""
