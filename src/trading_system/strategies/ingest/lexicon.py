"""The vocabulary the converter is willing to read, and the words that stop it.

Everything this pipeline understands is enumerated here, and nothing is
inferred from spelling. A phrase that is not in one of these tables is not
"probably an EMA" — it is a word the converter does not know, which makes the
clause holding it unreadable and the card a refusal. That is the whole
discipline: the tables grow when a human adds to them, never when a card
happens to look familiar.

Three kinds of table live here:

* **Vocabulary** — indicator, channel, price, pattern and session phrases that
  map onto something :class:`~trading_system.strategies.schema.StrategySpec`
  can actually say.
* **Blockers** — phrases whose presence proves a rule needs something the
  schema does not have (another timeframe, arithmetic over bar fields, a
  market regime). These are checked before parsing, so such a card is reported
  under the limitation that really stopped it rather than as an unreadable
  sentence.
* **Off-registry names** — indicators that are real and well known but have no
  implementation in this system. Listing them buys nothing at conversion time
  (an unknown word refuses either way); it buys a report that says "Parabolic
  SAR" instead of "unparsed clause", which is the difference between a
  measurement and a shrug.
"""

from trading_system.data.sessions import Session
from trading_system.features.patterns import Pattern

#: Indicator phrase to registry key. Phrases are matched on whole tokens,
#: longest first, so ``"exponential moving average"`` wins over ``"average"``.
#: A bare ``"moving average"`` is absent on purpose: EMA and SMA give different
#: signals, and picking one for a card that did not say which is a guess.
INDICATOR_PHRASES: dict[str, str] = {
    "ema": "ema",
    "e ma": "ema",
    "exponential moving average": "ema",
    "exponential ma": "ema",
    "sma": "sma",
    "simple moving average": "sma",
    "simple ma": "sma",
    "ma simple": "sma",
    "wma": "wma",
    "weighted moving average": "wma",
    "hma": "hma",
    "hull moving average": "hma",
    "vwma": "vwma",
    "volume weighted moving average": "vwma",
    "rsi": "rsi",
    "relative strength index": "rsi",
    "macd": "macd",
    "stochastic": "stoch",
    "stochastic oscillator": "stoch",
    "stoch": "stoch",
    "cci": "cci",
    "commodity channel index": "cci",
    "adx": "adx",
    "average directional index": "adx",
    "atr": "atr",
    "average true range": "atr",
    "bollinger bands": "bbands",
    "bollinger band": "bbands",
    "bollinger": "bbands",
    "bb": "bbands",
    "keltner": "keltner",
    "keltner channel": "keltner",
    "keltner channels": "keltner",
    "donchian": "donchian",
    "donchian channel": "donchian",
    "supertrend": "supertrend",
    "super trend": "supertrend",
    "williams r": "willr",
    "williams %r": "willr",
    "willr": "willr",
    "mfi": "mfi",
    "money flow index": "mfi",
    "obv": "obv",
    "on balance volume": "obv",
    "roc": "roc",
    "rate of change": "roc",
    "ichimoku": "ichimoku",
    "choppiness index": "chop",
    "standard deviation": "stddev",
    "relative volume": "rvol",
    "volume ma": "volume_ma",
    "volume moving average": "volume_ma",
    "vwap": "vwap_session",
    "pivot point": "pivots",
    "pivot points": "pivots",
    "swing high": "swing",
    "swing low": "swing",
    "swing highs": "swing",
    "swing lows": "swing",
    "fractal": "swing",
    "fractals": "swing",
}

#: Constructor parameters each supported indicator takes, in the order a card
#: writes them ("Stochastic (14, 5, 5)"). A mention carrying a different count
#: of numbers is ambiguous and refused rather than padded with defaults.
INDICATOR_PARAMS: dict[str, tuple[str, ...]] = {
    "ema": ("period",),
    "sma": ("period",),
    "wma": ("period",),
    "hma": ("period",),
    "vwma": ("period",),
    "rsi": ("period",),
    "cci": ("period",),
    "adx": ("period",),
    "atr": ("period",),
    "willr": ("period",),
    "mfi": ("period",),
    "roc": ("period",),
    "rvol": ("period",),
    "volume_ma": ("period",),
    "stddev": ("period",),
    "chop": ("period",),
    "donchian": ("period",),
    "swing": ("lookback",),
    "structure": ("lookback",),
    "obv": (),
    "vwap_session": (),
    "pivots": (),
    "macd": ("fast_period", "slow_period", "signal_period"),
    "stoch": ("k_period", "k_smooth", "d_period"),
    "bbands": ("period", "num_std"),
    "keltner": ("ema_period", "atr_period", "multiplier"),
    "supertrend": ("period", "multiplier"),
    "ichimoku": ("tenkan_period", "kijun_period", "senkou_b_period", "displacement"),
}

#: Parameters that must stay whole numbers when a card writes them as ``20.0``.
_INTEGER_PARAMS: frozenset[str] = frozenset(
    {
        "period",
        "lookback",
        "fast_period",
        "slow_period",
        "signal_period",
        "k_period",
        "k_smooth",
        "d_period",
        "ema_period",
        "atr_period",
        "tenkan_period",
        "kijun_period",
        "senkou_b_period",
        "displacement",
    }
)


def coerce_param(name: str, value: float) -> float | int:
    """Give ``value`` the type the indicator's constructor expects.

    Args:
        name: Parameter name.
        value: Number as the card wrote it.

    Returns:
        An ``int`` for lookback-style parameters, the float unchanged otherwise.
    """
    return int(value) if name in _INTEGER_PARAMS else value


#: Phrase to channel name, per indicator. Only spellings that name one line
#: unambiguously: "the bollinger bands" names three lines and is not here.
CHANNEL_PHRASES: dict[str, dict[str, str]] = {
    "bbands": {
        "upper band": "upper",
        "upper bands": "upper",
        "top band": "upper",
        "lower band": "lower",
        "lower bands": "lower",
        "bottom band": "lower",
        "middle band": "middle",
        "middle line": "middle",
        "center line": "middle",
        "centre line": "middle",
        "basis": "middle",
    },
    "keltner": {
        "upper band": "upper",
        "upper line": "upper",
        "lower band": "lower",
        "lower line": "lower",
        "middle band": "middle",
        "middle line": "middle",
        "center line": "middle",
        "centre line": "middle",
    },
    "donchian": {
        "upper band": "upper",
        "upper line": "upper",
        "lower band": "lower",
        "lower line": "lower",
        "middle band": "middle",
        "middle line": "middle",
    },
    "macd": {
        "signal line": "signal",
        "signal": "signal",
        "histogram": "histogram",
        "main line": "macd",
        "macd line": "macd",
    },
    "stoch": {
        "%k": "k",
        "k line": "k",
        "%d": "d",
        "d line": "d",
        "signal line": "d",
    },
    "adx": {
        "+di": "plus_di",
        "-di": "minus_di",
        "di+": "plus_di",
        "di-": "minus_di",
        "d+": "plus_di",
        "d-": "minus_di",
        "plus di": "plus_di",
        "minus di": "minus_di",
        "main line": "adx",
    },
    "supertrend": {
        "line": "line",
        "upper band": "upper",
        "lower band": "lower",
    },
    # Only the two-word spellings: a bare "high" is a price field far more often
    # than it is a swing channel, and the swing phrases already carry their side.
    "swing": {
        "swing high": "swing_high",
        "swing low": "swing_low",
    },
    "ichimoku": {
        "tenkan": "tenkan",
        "tenkan sen": "tenkan",
        "kijun": "kijun",
        "kijun sen": "kijun",
        "senkou a": "senkou_a",
        "senkou b": "senkou_b",
    },
    "pivots": {
        "pivot": "pivot",
        "r1": "r1",
        "r2": "r2",
        "r3": "r3",
        "s1": "s1",
        "s2": "s2",
        "s3": "s3",
    },
}

#: The channel a bare mention of a multi-output indicator means. Small on
#: purpose: every entry is a reading the card did not spell out, so each one
#: is recorded as an assumption on the converted spec rather than applied
#: silently. Indicators absent here must be named with a channel or refused —
#: "the bollinger bands are rising" says nothing about which of three lines.
DEFAULT_CHANNELS: dict[str, str] = {
    "macd": "macd",
    "stoch": "k",
    "adx": "adx",
    "supertrend": "line",
}

#: Price phrases to the bar field they name.
PRICE_PHRASES: dict[str, str] = {
    "price": "close",
    "the price": "close",
    "price action": "close",
    "close": "close",
    "closing price": "close",
    "close price": "close",
    "candle close": "close",
    "closes": "close",
    "candle": "close",
    "candles": "close",
    "bar": "close",
    "candlestick": "close",
    "high": "high",
    "candle high": "high",
    "bar high": "high",
    "low": "low",
    "candle low": "low",
    "bar low": "low",
    "open": "open",
    "opening price": "open",
}

#: Phrases meaning the constant zero, which oscillator rules lean on heavily.
ZERO_PHRASES: tuple[str, ...] = ("zero", "zero line", "the zero line", "centre line", "0 line")

#: Candlestick pattern phrases the label set understands.
PATTERN_PHRASES: dict[str, Pattern] = {
    "doji": Pattern.DOJI,
    "inside bar": Pattern.INSIDE_BAR,
    "outside bar": Pattern.OUTSIDE_BAR,
    "bullish engulfing": Pattern.BULLISH_ENGULFING,
    "bearish engulfing": Pattern.BEARISH_ENGULFING,
    "hammer": Pattern.HAMMER,
    "shooting star": Pattern.SHOOTING_STAR,
    "morning star": Pattern.MORNING_STAR,
    "evening star": Pattern.EVENING_STAR,
}

#: Session phrases. "Asian session" is absent: it spans Sydney and Tokyo, and
#: choosing one of them would narrow a rule the card wrote wider.
SESSION_PHRASES: dict[str, Session] = {
    "london session": Session.LONDON,
    "european session": Session.LONDON,
    "new york session": Session.NEWYORK,
    "us session": Session.NEWYORK,
    "tokyo session": Session.TOKYO,
    "sydney session": Session.SYDNEY,
}

#: Words a rule can carry without changing what it says.
FILLER_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "current",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "indicator",
        "indicators",
        "is",
        "it",
        "its",
        "level",
        "levels",
        "must",
        "of",
        "on",
        "or",
        "period",
        "periods",
        "setting",
        "settings",
        "should",
        "that",
        "the",
        "then",
        "this",
        "to",
        "value",
        "values",
        "was",
        "were",
        "when",
        "while",
        "with",
    }
)

#: Clauses that carry no rule at all: captions, filenames, calls to action and
#: the scraper's leftovers. Matched as substrings against the lower-cased
#: clause; a clause that matches is dropped rather than refused, because a
#: caption is not a sentence the converter failed to understand.
NOISE_MARKERS: tuple[str, ...] = (
    "in the picture",
    "in the pictures",
    "picture below",
    "pictures below",
    "example of trade",
    "examples of trade",
    "write a comment",
    "share this",
    "telegram",
    "download",
    ".rar",
    ".zip",
    ".ex4",
    ".mq4",
    "template mt4",
    "mt4 template",
    "trading system",
    "trading strategy",
    "forex strategies resources",
    "submit by",
    "conditions",
    "recommended",
)

#: Phrases proving a rule reasons across timeframes. The schema declares
#: ``htf_filter_tf`` but the entry compiler reads one stream, so a rule of this
#: shape cannot be honoured — see the cross-timeframe limitation in CLAUDE.md.
CROSS_TIMEFRAME_MARKERS: tuple[str, ...] = (
    "higher time frame",
    "higher timeframe",
    "higher tf",
    "lower time frame",
    "lower timeframe",
    "multi time frame",
    "multi-time frame",
    "multitimeframe",
    "mtf",
    "on the daily",
    "daily chart",
    "weekly chart",
    "monthly chart",
    "h4 chart",
    "h1 chart",
    "m15 chart",
    "m5 chart",
    "4h chart",
    "1h chart",
    "on h4",
    "on h1",
    "on m15",
    "on m5",
    "on the h4",
    "on the h1",
    "bigger time frame",
    "larger time frame",
    "two time frames",
    "second chart",
)

#: A chart named by its bar size — "60min chart", "the 15 minute charts", "the
#: 4 hour chart". A phrase table cannot catch these: the number runs into the
#: unit ("60min"), so the word boundary a phrase match needs is not there.
CHART_TIMEFRAME_PATTERN = r"\d+\s*(?:min|minute|minutes|hour|hours|h|m)\s*charts?"

#: Phrases proving a rule does arithmetic over bar fields. Operands are a
#: feature, a label, a price field or a constant; there are no expressions over
#: them, so "the body is larger than the previous one" has no encoding.
BAR_ARITHMETIC_MARKERS: tuple[str, ...] = (
    "body",
    "bodies",
    "wick",
    "wicks",
    "shadow",
    "shadows",
    "tail of the candle",
    "half of the",
    "halfway",
    "midpoint",
    "middle of the candle",
    "size of the candle",
    "candle size",
    "bar size",
    "range of the bar",
    "range of the candle",
    "average price of the previous bar",
    "upper third",
    "lower third",
    "larger than the previous",
    "bigger than the previous",
    "twice the",
    "% of the",
    "percent of the",
    "pips above",
    "pips below",
    "pips higher",
    "pips lower",
)

#: Phrases naming a market regime. ``regime_is`` parses but the compiler
#: rejects it — P08 was dropped — so a rule gated on one cannot be encoded.
REGIME_MARKERS: tuple[str, ...] = (
    "trending market",
    "market is trending",
    "ranging market",
    "market is flat",
    "flat market",
    "sideways market",
    "in a range",
    "range bound",
    "ranging phase",
    "choppy market",
    "low volatility period",
    "high volatility period",
    "only in the direction of the trend",
    "trade only in trend",
    "uptrend only",
    "downtrend only",
)

#: Phrases naming something a human sees on a chart rather than something a
#: series carries: indicator arrows, colour changes, hand-drawn levels.
VISUAL_MARKERS: tuple[str, ...] = (
    "arrow",
    "arrows",
    "dot",
    "dots",
    "colour",
    "color",
    "turns green",
    "turns red",
    "turns blue",
    "blue line",
    "red line",
    "green line",
    "yellow line",
    "aqua",
    "histogram colour",
    "support zone",
    "resistance zone",
    "support and resistance",
    "support level",
    "resistance level",
    "supply zone",
    "demand zone",
    "trend line",
    "trendline",
    "trend lines",
    "channel line",
    "fibonacci",
    "fib level",
    "semafor",
    "sun",
    "signal appears",
    "alert",
    "sub window",
    "subwindow",
    "sub-window",
    "chart pattern",
    "flag",
    "wedge",
    "triangle",
    "head and shoulders",
    "double top",
    "double bottom",
)

#: Indicators that exist in the wider world but not in this system's registry.
#: Purely for reporting: an unknown word refuses a card either way, but a
#: refusal that names "Parabolic SAR" is a finding and one that says "unparsed
#: clause" is not.
OFF_REGISTRY_INDICATORS: tuple[str, ...] = (
    "parabolic sar",
    "psar",
    "awesome oscillator",
    "accelerator oscillator",
    "alligator",
    "gator",
    "zigzag",
    "zig zag",
    "heiken ashi",
    "heikin ashi",
    "momentum indicator",
    "detrended price oscillator",
    "detrend price oscillator",
    "laguerre",
    "tma band",
    "tma bands",
    "half trend",
    "halftrend",
    "trend envelopes",
    "renko",
    "ssl channel",
    "qqe",
    "waddah attar",
    "murrey math",
    "gann",
    "elliott",
    "harmonic pattern",
    "camarilla",
    "starc band",
    "starc bands",
    "hull trend",
    "step ma",
    "t3 ma",
    "kijun flat",
    "vortex",
    "aroon",
    "trix",
    "dmi",
    "elder ray",
    "chaikin",
    "force index",
    "klinger",
    "schaff",
    "coral",
    "fisher",
    "cyber cycle",
    "rsioma",
    "stochrsi",
    "stochastic rsi",
    "supply demand",
    "order block",
    "ichimoku cloud kumo twist",
    "point and figure",
    "squeeze momentum",
    "ttm squeeze",
    "volume profile",
    "market profile",
    "cog",
    "regression channel",
    "polynomial regression",
    "super passband",
    "passband filter",
    "damiani",
    "volameter",
    "magic trend",
    "follow line",
    "nrtr",
    "hlc trend",
    "pivot ray",
)
