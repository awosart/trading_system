"""P04's leaf operators, as pure functions over closed bars.

Two decisions shape this module.

**Three-valued logic.** A condition is ``True``, ``False``, or ``None`` meaning
"not decidable from the bars available" — an indicator still in warmup, a
``cross_above`` on the first bar of the series, a feature whose value depends on
data that has not arrived. Collapsing unknown into ``False`` looks harmless until
a :class:`~trading_system.strategies.schema.Not` node sits above it: a warmup bar
where "RSI < 30" is unknown would become "not (RSI < 30)" = ``True``, and the
strategy would trade the warmup. So unknown propagates: :func:`negate` maps it to
itself, :func:`and_all` reports it unless some child is already ``False``, and
:func:`or_any` reports it unless some child is already ``True``. Only ``True``
at the root of a tree is a fired condition.

**Every function is a plain function of values.** No context, no state, no
series. A crossing takes the four numbers it needs — both operands on bar ``t``
and both on ``t-1`` — which makes each operator checkable against a hand-written
answer, and makes it structurally impossible for one to reach a bar its caller
did not hand it.

The crossing definition is the load-bearing one:
``cross_above(a, b) == a[t] > b[t] and a[t-1] <= b[t-1]``. Both bars are closed,
so no part of it can be known before bar ``t`` closes. The previous bar's
comparison is non-strict on purpose: a series that sits exactly on the level and
then breaks it has crossed, and requiring ``a[t-1] < b[t-1]`` would drop exactly
the touch-and-go case a breakout strategy is looking for.
"""

from collections.abc import Iterable

#: A condition's outcome. ``None`` means "not decidable from the bars available",
#: which is neither true nor false and must not be confused with either.
type Truth = bool | None

#: An operand's value at one bar, or ``None`` where the series has none.
type Value = float | None


def and_all(values: Iterable[Truth]) -> Truth:
    """Kleene conjunction: ``False`` beats unknown, unknown beats ``True``.

    Args:
        values: Child outcomes. Consumed lazily, so a ``False`` short-circuits.

    Returns:
        ``False`` if any child is ``False``; else ``None`` if any is unknown;
        else ``True``. An empty input is vacuously ``True``.
    """
    unknown = False
    for value in values:
        if value is False:
            return False
        if value is None:
            unknown = True
    return None if unknown else True


def or_any(values: Iterable[Truth]) -> Truth:
    """Kleene disjunction: ``True`` beats unknown, unknown beats ``False``.

    Args:
        values: Child outcomes. Consumed lazily, so a ``True`` short-circuits.

    Returns:
        ``True`` if any child is ``True``; else ``None`` if any is unknown; else
        ``False``. An empty input is vacuously ``False``.
    """
    unknown = False
    for value in values:
        if value is True:
            return True
        if value is None:
            unknown = True
    return None if unknown else False


def negate(value: Truth) -> Truth:
    """Kleene negation, which leaves unknown unknown.

    Args:
        value: Outcome to negate.

    Returns:
        The negation, or ``None`` if ``value`` is ``None``. Mapping unknown to
        ``True`` here is the bug this whole module exists to prevent: it would
        let a negated condition fire on bars where nothing was actually known.
    """
    return None if value is None else not value


def gt(left: Value, right: Value) -> Truth:
    """Whether ``left`` is strictly greater than ``right``.

    Args:
        left: Left operand on bar ``t``.
        right: Right operand on bar ``t``.

    Returns:
        The comparison, or ``None`` if either operand has no value.
    """
    if left is None or right is None:
        return None
    return left > right


def gte(left: Value, right: Value) -> Truth:
    """Whether ``left`` is greater than or equal to ``right``.

    Args:
        left: Left operand on bar ``t``.
        right: Right operand on bar ``t``.

    Returns:
        The comparison, or ``None`` if either operand has no value.
    """
    if left is None or right is None:
        return None
    return left >= right


def lt(left: Value, right: Value) -> Truth:
    """Whether ``left`` is strictly less than ``right``.

    Args:
        left: Left operand on bar ``t``.
        right: Right operand on bar ``t``.

    Returns:
        The comparison, or ``None`` if either operand has no value.
    """
    if left is None or right is None:
        return None
    return left < right


def lte(left: Value, right: Value) -> Truth:
    """Whether ``left`` is less than or equal to ``right``.

    Args:
        left: Left operand on bar ``t``.
        right: Right operand on bar ``t``.

    Returns:
        The comparison, or ``None`` if either operand has no value.
    """
    if left is None or right is None:
        return None
    return left <= right


def cross_above(left_now: Value, left_prev: Value, right_now: Value, right_prev: Value) -> Truth:
    """Whether ``left`` crossed above ``right`` between the two closed bars.

    Defined as ``left[t] > right[t] and left[t-1] <= right[t-1]``. Both bars are
    closed, so the result is knowable exactly at the close of bar ``t`` and never
    earlier. The previous-bar comparison is non-strict so that a series resting
    on the level and then breaking it counts as a crossing.

    Args:
        left_now: Left operand on bar ``t``.
        left_prev: Left operand on bar ``t-1``.
        right_now: Right operand on bar ``t``.
        right_prev: Right operand on bar ``t-1``.

    Returns:
        Whether the crossing happened, or ``None`` if any of the four values is
        missing — including on the first bar of a series, where ``t-1`` does not
        exist and no crossing can be asserted or denied.
    """
    if left_now is None or left_prev is None or right_now is None or right_prev is None:
        return None
    return left_now > right_now and left_prev <= right_prev


def cross_below(left_now: Value, left_prev: Value, right_now: Value, right_prev: Value) -> Truth:
    """Whether ``left`` crossed below ``right`` between the two closed bars.

    The mirror of :func:`cross_above`:
    ``left[t] < right[t] and left[t-1] >= right[t-1]``.

    Args:
        left_now: Left operand on bar ``t``.
        left_prev: Left operand on bar ``t-1``.
        right_now: Right operand on bar ``t``.
        right_prev: Right operand on bar ``t-1``.

    Returns:
        Whether the crossing happened, or ``None`` if any value is missing.
    """
    if left_now is None or left_prev is None or right_now is None or right_prev is None:
        return None
    return left_now < right_now and left_prev >= right_prev


def between(value: Value, low: float, high: float) -> Truth:
    """Whether ``value`` lies within ``[low, high]``, bounds included.

    Args:
        value: Operand on bar ``t``.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        The test, or ``None`` if ``value`` is missing.
    """
    if value is None:
        return None
    return low <= value <= high


def inside_range(value: Value, low: float, high: float) -> Truth:
    """Whether ``value`` lies strictly within ``(low, high)``, bounds excluded.

    P04 carries both ``between`` and ``inside_range`` with identical operand
    shapes and no stated difference. Making them synonyms would leave one of the
    two meaningless, so the exclusive reading is assigned here: ``between`` is
    "at or within the bounds", ``inside_range`` is "strictly inside them". The
    distinction matters at a boundary — RSI exactly 30 is ``between(30, 70)`` but
    not ``inside_range(30, 70)``.

    Args:
        value: Operand on bar ``t``.
        low: Exclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        The test, or ``None`` if ``value`` is missing.
    """
    if value is None:
        return None
    return low < value < high


def rising(now: Value, past: Value) -> Truth:
    """Whether a series is higher than it was ``n`` bars ago.

    Compares the two endpoints only. "Rose over five bars" here means
    ``value[t] > value[t-5]``, not that every bar in between was an increase — a
    monotonic test is a different question and is not what P04's ``rising`` says.

    Args:
        now: Operand on bar ``t``.
        past: Operand on bar ``t-n``.

    Returns:
        The comparison, or ``None`` if either endpoint is missing.
    """
    if now is None or past is None:
        return None
    return now > past


def falling(now: Value, past: Value) -> Truth:
    """Whether a series is lower than it was ``n`` bars ago.

    The mirror of :func:`rising`, comparing endpoints only.

    Args:
        now: Operand on bar ``t``.
        past: Operand on bar ``t-n``.

    Returns:
        The comparison, or ``None`` if either endpoint is missing.
    """
    if now is None or past is None:
        return None
    return now < past
