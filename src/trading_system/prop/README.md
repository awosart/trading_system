# Prop Guard

The last veto before an order reaches execution: the firm's own account rules,
applied to a size the Risk Engine has already computed.

## The numbers in this module are secondary and will go stale

`configs/prop_rules.yaml` carries drawdown limits, profit targets and
consistency rules for four plans. **Every figure in it is a secondary reading
of unknown date.** Prop firms revise their rulebooks often and without notice:

- profit targets move between plans and between phases;
- the daily-loss basis switches between balance-at-day-start and equity, which
  changes the limit by exactly the floating P&L carried through the reset;
- total-loss floors switch between static and trailing high-water, which is the
  difference between a floor that never moves and one that ratchets up with
  every new equity peak;
- consistency rules appear, change threshold, and disappear;
- minimum trading days change, and some plans have dropped the requirement.

**Reconcile against the firm's own rulebook before trusting any result computed
from this file.** A simulation that says "92% chance of passing" is a statement
about the rules as written here, not about the account you would actually open.

The same warning already stands on `configs/prop_profiles.yaml`, which carries
the *leverage* side of the same firms. The two files are deliberately separate:
that one says what the firm lets you hold, this one says what it lets you lose.
Each entry here names its counterpart by `prop_profile` and restates no
leverage figure of its own — margin and the exposure ceiling were closed in
P10 stage 3 and are enforced inside
`trading_system.risk.engine.RiskEngine.evaluate`, not here.

## What is in the loop and what is not

| rule | where | why |
|---|---|---|
| daily loss limit | in the event loop | blocks entries for the rest of the firm's day, changing which trades exist |
| total loss limit | in the event loop | same, for the rest of the episode |
| `max_allowed_risk_now` | in the event loop | reduces or refuses a size that does not fit the remaining allowance |
| margin, leverage cap | in the event loop, **P10 stage 3** | `RiskEngine.evaluate` step 6.5 |
| consistency (`max_single_day_profit_share`) | after a finished sequence | needs total profit, which does not exist until the sequence ends |
| `min_trading_days` | neither | a *pass predicate*, and a modifier of the episode's stopping rule — see `simulator.py` |

## Two definitions of "day", both legitimate, both named

`BacktestConfig.day_origin` is a **market-data** boundary: it resets a session
VWAP, labels `EquityPoint.day`, and cuts D1 bars. `PropRules.daily_reset_tz`
plus `daily_reset_time` is a **firm rule** boundary: it refills the daily loss
allowance. These genuinely differ — the FX convention rolls at 17:00 New York
while FTMO measures its day at Prague midnight — and forcing them to agree
would mis-state one of the two.

What does *not* differ is the function: both go through
`trading_system.data.resample.trading_day`, called with two different
`DayOrigin` values. There is one definition of how an instant becomes a day
label, used twice. When the two origins disagree, the run logs it **and the
report prints it**, because the divergence changes what the daily limit means
and a log is only read when something has already gone wrong.
