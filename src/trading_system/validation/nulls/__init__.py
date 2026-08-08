"""The two zero processes: permutation and random entry, and what each answers.

Permutation (:mod:`~trading_system.validation.nulls.permutation`) answers "is
the fold sequence itself sound" — a system with no edge, run over data with no
predictability, must score around zero. Random entry
(:mod:`~trading_system.validation.nulls.random_entry`) answers a narrower,
harder question: "are *these* entries better than entries placed at random,
with the same sizing profile, on the same real data, through the same costs
and exit rules." Neither is an optimiser and neither renders a verdict — both
belong to :mod:`trading_system.validation.calibration`, which runs many
iterations of one and reports where the real result sits in the distribution.
"""
