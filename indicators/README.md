# Indicators

## Buyers & Sellers Profile + Dynamic S/R — with Signal Dashboard

`buyers_sellers_profile_dynamic_sr_dashboard.pine`

The original Zeiierman indicator, unchanged, plus a statistics dashboard.
No existing input, calculation, level, drawing or alert was modified — the
dashboard is additive and can be switched off with a single toggle.

### What counts as a signal

A signal is fired when the profile engine stores a **new** profile-guided
level — exactly the same events that drive the existing alerts:

| Event | Direction |
|---|---|
| New profile-confirmed **support** | Long |
| New profile-confirmed **resistance** | Short |

Signals are only tallied on a **closed** bar, so counts never flicker on the
forming candle.

### How win / loss is decided

Each signal is tracked forward from the close of its confirmation bar:

- **Target** = `Target ATR` × ATR(14) in the signal's direction → **Win**
- **Stop** = `Stop ATR` × ATR(14) against it → **Loss**
- Neither reached within `Max Hold Bars` → **Expired** (counts as an entry,
  excluded from win rate)
- Both touched inside the same bar → scored conservatively as a **Loss**

Every outcome is tallied against the **weekday of the entry bar**, so a signal
taken on Tuesday that resolves on Thursday still counts toward Tuesday.

### Dashboard layout

```
DASHBOARD                    15m
Week      Sig    W    L   Win%
Mon        4     3    1  75.0%
Tue        6     2    3  40.0%
Wed        2     1    ·  100.0%
...
This Week 18    11    5  68.8%
Last Week 23    12    9  57.1%
All Time 412   201  176  53.3%
Signals  Long Short Open  Exp
Week       10     8    2    1
Last     Long 1.2345 ago 7
Levels    Sup     6  Res   5
```

- **Day rows** — per-weekday entries, wins, losses and win rate. The current
  day is highlighted; win-rate cells are shaded green above 50% and red below.
- **Day Rows** input switches the day rows between **This Week** (Mon, Tue,
  Wed… of the current week only) and **All History** (every signal on the
  chart, aggregated by weekday).
- **This Week / Last Week / All Time** — weekly and overall win rate.
- **Signals** — long vs short counts for the selected scope, signals still
  open, and total expiries.
- **Last** — direction, price and age of the most recent signal.
- **Levels** — active support and resistance levels right now.

### Dashboard inputs

| Input | Default | Purpose |
|---|---|---|
| Show Dashboard | on | Master toggle |
| Position / Text Size | Top Right / Small | Table placement |
| Day Rows | This Week | Scope of the weekday rows |
| Week Start | Monday | Monday for equities/futures, Sunday for FX/crypto |
| Target ATR | 2.0 | Win distance |
| Stop ATR | 1.0 | Loss distance |
| Max Hold Bars | 200 | Expiry window |

### Notes

- Statistics are built from the chart's loaded history, so a longer history
  and a lower timeframe give a larger sample. Changing timeframe, symbol or
  any engine input recalculates everything from scratch.
- Outcomes are a forward ATR target/stop simulation of the indicator's own
  signals — useful for comparing days and settings, not a substitute for a
  strategy backtest with costs and slippage.

Original indicator © Zeiierman, CC BY-NC-SA 4.0.
