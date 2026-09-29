# Indicators

## Buyers & Sellers Profile + Dynamic S/R — with Signal Dashboard

`buyers_sellers_profile_dynamic_sr_dashboard.pine`

The original Zeiierman indicator, unchanged, plus a statistics dashboard.
No existing input, calculation, level, drawing or alert was modified — the
dashboard is additive and can be switched off with a single toggle.

### What counts as a signal

A signal is a **reaction to an existing level**, not the creation of one. On a
closed bar the engine scans the active profile-guided levels and takes the
strongest one that price just rejected:

| Setup | Condition | Trade |
|---|---|---|
| Support hold | wick into the level, close back **above** it, bullish bar | **Long** |
| Resistance hold | wick into the level, close back **below** it, bearish bar | **Short** |

Filters that keep it clean: the level must score at least `Min Level Score`,
only one trade may be open at a time (optional), and `Cooldown Bars` must have
passed since the last signal.

### Entry, stop and target

- **Entry** — close of the signal bar, drawn as a dashed line.
- **Stop** — beyond both the level and the signal bar extreme, plus
  `Stop Buffer ATR`. Drawn as a red zone from entry down to the stop.
- **Target** — `Reward : Risk` × the risk distance. Drawn as a green zone from
  entry up to the target.
- Signals whose stop would sit further than `Max Risk ATR` away are skipped.
- Zones extend to the right while the trade is open and stop at the bar that
  closed it. The label reads `L ✓`, `S ✕` or `L –`; hover it for entry, stop,
  target, risk and status.

Only the last `Draw Last` trades stay drawn, so the chart never fills up.

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
| Reward : Risk | 2.0 | Target as a multiple of risk |
| Min Level Score | 60 | Level quality gate |
| One Trade At A Time | on | Blocks overlapping trades |

### Notes

- Statistics are built from the chart's loaded history, so a longer history
  and a lower timeframe give a larger sample. Changing timeframe, symbol or
  any engine input recalculates everything from scratch.
- Outcomes are a forward ATR target/stop simulation of the indicator's own
  signals — useful for comparing days and settings, not a substitute for a
  strategy backtest with costs and slippage.

Original indicator © Zeiierman, CC BY-NC-SA 4.0.
