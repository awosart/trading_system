"""Categorical bar labels: which patterns, sessions and regimes a bar carries.

The counterpart to features. A feature answers "what number does this series
carry on bar t"; a label answers "what kind of bar is this" — a hammer, a London
bar, a ranging market. P04 expresses the question as ``pattern_is`` /
``session_is`` / ``regime_is`` over a
:class:`~trading_system.strategies.schema.LabelSet`, and this module builds the
columns those operators read.

Two conventions, both inherited rather than invented:

* **A bar belongs to the session it opened in.** Bar timestamps are open times
  throughout the system, and a 07:45–08:00 bar traded entirely before the London
  open however close to it the close time lands. Labelling by close time would
  put that bar in London, which is the opposite of what a session-open strategy
  means.
* **Pattern warmup is uniform.** ``detect_patterns`` reports a one-bar pattern on
  bar 0 and a three-bar pattern as null there, but a label *set* has nowhere to
  record "DOJI is false and MORNING_STAR is unknown". Rather than quietly read
  the second as false, the first ``max(PATTERN_WINDOW) - 1`` bars carry ``None``
  for the whole category — the same rule P03 applies to a multi-channel
  indicator, where a row is either wholly usable or not published.

Regimes have no builder: the Regime module does not exist yet. A caller may
supply the column, and :mod:`trading_system.entry.compiler` refuses to compile
``regime_is`` until there is something real to fill it.
"""

from collections.abc import Sequence
from enum import StrEnum

from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.data.sessions import session_of
from trading_system.features.patterns import PATTERN_WINDOW, Pattern, detect_patterns

#: Bars at the start of a frame that cannot carry a pattern label, being shorter
#: than the longest pattern's window.
PATTERN_WARMUP = max(PATTERN_WINDOW.values()) - 1


class LabelCategory(StrEnum):
    """A vocabulary of bar labels, and the column that carries it."""

    PATTERN = "pattern"
    REGIME = "regime"
    SESSION = "session"


#: One label column: the set of labels on each bar, or ``None`` where the bar
#: cannot be classified at all. ``None`` is *unknown*; an empty set is *known,
#: and none apply* — a distinction the operator layer relies on to keep a
#: negated label test from firing during warmup.
type LabelColumn = list[frozenset[str] | None]


def pattern_labels(frame: OHLCVFrame) -> LabelColumn:
    """The candlestick patterns present on each bar.

    Args:
        frame: Bars to classify.

    Returns:
        One label set per bar, ``None`` for the leading :data:`PATTERN_WARMUP`
        bars.
    """
    detected = detect_patterns(frame)
    names = [pattern.value for pattern in Pattern]
    columns = [detected[name].to_list() for name in names]
    return [
        None
        if index < PATTERN_WARMUP
        else frozenset(name for name, column in zip(names, columns, strict=True) if column[index])
        for index in range(len(frame))
    ]


def session_labels(frame: OHLCVFrame) -> LabelColumn:
    """The trading sessions active when each bar opened.

    Sessions overlap, so a bar routinely carries several — including the derived
    ``LONDON_NY_OVERLAP``. A bar outside every session carries an empty set,
    which is known rather than unknown: the market being closed is an answer.

    Args:
        frame: Bars to classify.

    Returns:
        One label set per bar.
    """
    return [
        frozenset(session.value for session in session_of(timestamp))
        for timestamp in frame.df[TIMESTAMP_COLUMN].to_list()
    ]


def label_columns(
    frame: OHLCVFrame, categories: Sequence[LabelCategory] = ()
) -> dict[str, LabelColumn]:
    """Build the label columns a compiled entry asked for.

    Only the requested categories are built: classifying patterns walks every bar
    in Python, which is not worth paying for on a strategy that never mentions
    them.

    Args:
        frame: Bars to classify.
        categories: Categories to build. ``REGIME`` is not buildable and is
            rejected rather than silently returned empty.

    Returns:
        Category name to column, ready for
        :meth:`~trading_system.entry.context.BarSeries.from_frame`.

    Raises:
        ValueError: If ``REGIME`` is requested.
    """
    columns: dict[str, LabelColumn] = {}
    for category in categories:
        if category is LabelCategory.PATTERN:
            columns[category.value] = pattern_labels(frame)
        elif category is LabelCategory.SESSION:
            columns[category.value] = session_labels(frame)
        else:
            raise ValueError(
                "regime labels cannot be built: no Regime module exists yet. Supply the column "
                "explicitly once one does."
            )
    return columns
