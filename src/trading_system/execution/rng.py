"""Deterministic randomness, seeded per fill rather than per run.

A single ``random.Random`` advanced once per fill is the obvious implementation
and it is wrong, for a reason that only shows up in portfolio runs: the value a
fill draws then depends on **how many fills happened before it**. Add a second
instrument to a run and every fill of the first instrument gets a different
number, because the two now interleave in the shared stream. The equity curve of
EURUSD changes because NAS100 was added to the portfolio, which is not a
modelling assumption anybody made — it is an artefact of stream position.

So the stream is keyed by the fill's own identity instead. Seed material is
``(run_seed, symbol, timestamp, order_id)`` hashed into a 64-bit integer; the
same fill draws the same value no matter what else the run contains, in what
order, or on how many threads. Reproducibility becomes a property of the fill,
not of the run's execution schedule.

``blake2b`` rather than :func:`hash`, because :func:`hash` is randomised per
process for :class:`str` unless ``PYTHONHASHSEED`` is pinned — a reproducibility
guarantee that depends on an environment variable is not one.
"""

import hashlib
from datetime import datetime
from random import Random

#: Separator between seed components. A character that cannot appear in an ISO
#: timestamp, so ``("AB", "C")`` and ``("A", "BC")`` cannot hash alike.
_SEPARATOR = "\x00"


def fill_seed(run_seed: int, *, symbol: str, ts: datetime, order_id: str) -> int:
    """Derive this fill's 64-bit seed from its identity.

    Args:
        run_seed: The run's seed. Changing it reshuffles every fill in the run
            together, which is what a seed is for.
        symbol: Instrument being filled.
        ts: When the fill happens, tz-aware.
        order_id: Identifier unique within the run.

    Returns:
        A seed depending only on the four arguments — never on call order.
    """
    material = _SEPARATOR.join((str(run_seed), symbol, ts.isoformat(), order_id))
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def fill_rng(run_seed: int, *, symbol: str, ts: datetime, order_id: str) -> Random:
    """A random stream belonging to one fill.

    Args:
        run_seed: The run's seed.
        symbol: Instrument being filled.
        ts: When the fill happens, tz-aware.
        order_id: Identifier unique within the run.

    Returns:
        A freshly seeded generator. Fresh rather than shared: a generator handed
        out twice would reintroduce the order dependence this module exists to
        remove.
    """
    return Random(fill_seed(run_seed, symbol=symbol, ts=ts, order_id=order_id))
