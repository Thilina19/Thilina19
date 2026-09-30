# XAUUSD Disciplined Signal Agent

A self-reviewing signal, risk-management and journaling system for gold.

It produces trade plans — entry, stop, two targets, position size — and it
refuses to produce actionable ones until its own statistics say it has earned
the right. **It does not place orders.**

```
  4h structure  ──>  1h volume-backed order block  ──>  15m Heikin Ashi trigger
       │                        │                              │
       └──────── confluence score /100, threshold 70 ──────────┘
                               │
                    risk manager (hard limits)
                               │
                  journal  ──>  statistical gate  ──>  paper or live
```

---

## Status

- 93 unit tests, all passing. Pure standard library — no dependencies.
- Validated end-to-end against real `OANDA:XAUUSD` data.
- **Currently in PAPER mode**, which is correct: zero forward-tested trades so
  far, so it has no basis for publishing live entries.

---

## Quick start

```bash
python3 -m unittest discover -s tests -t .        # 93 tests

# where you stand
python3 -m xau_agent.cli status

# analyse the latest bar
python3 -m xau_agent.cli --entry-tf 1h --zone-tf 4h --bias-tf 1D \
  analyse --entry data/XAUUSD_1h_oanda.csv --bias data/XAUUSD_1D_oanda.csv
```

Global flags (`--equity`, `--entry-tf`, …) come **before** the subcommand.

---

## Configuration for this account

| | |
|---|---|
| Account | $50,000 |
| Risk per trade | 0.5% = $250 |
| Daily target | +$500 (1%) — stop when reached |
| Daily loss cap | −$500 — stop when reached |
| Max trades/day | 3 |
| Two losses | session over |
| Cool-off after a loss | 30 minutes |
| Execution | signals only; you place every order |

Full rules and the daily routine: **[docs/PLAYBOOK.md](docs/PLAYBOOK.md)**.

---

## Getting data for a real backtest

The included samples (399 1h bars, 300 daily bars) were enough to validate the
engine but are **far too short to evaluate the strategy**. Two constraints:

- A 4h EMA200 needs ~250 4h bars ≈ **42 days** of continuous history just to
  warm up.
- A win rate needs **60–100 closed trades** to mean anything.

Export from your own broker — that is also the feed you will be filled on, so
it removes a real source of backtest optimism:

**MT5** → Tools → History Center → XAUUSD → M15 → Export, or right-click a
chart → Save As. Then:

```bash
python3 -m xau_agent.cli backtest --entry data/XAUUSD_M15.csv
```

`load_csv` handles the usual MT5 and TradingView layouts. Pass
`tz_offset_hours` if your server stamps bars in broker time rather than UTC —
getting that wrong silently misplaces every session filter.

---

## Layout

```
xau_agent/
  config.py      RiskLimits (frozen, immutable by the learning loop)
                 StrategyParams (tunable, with a whitelist)
  indicators.py  Heikin Ashi, ATR, EMA, RSI, volume ratio, fractal swings
  structure.py   bias, BOS/CHoCH, order blocks, FVG  (all lookahead-guarded)
  strategy.py    confluence scoring -> Signal or Rejection
  risk.py        position sizing and every guardrail
  journal.py     SQLite journal, statistics, Wilson CI, promotion gate
  backtest.py    bar-by-bar simulation with spread, slippage, pessimistic fills
  review.py      walk-forward parameter review + discipline audit
  data.py        CSV/JSON loading, resampling, calibrated test fixture
  cli.py         analyse / backtest / status / review / log-trade
tests/           93 tests, risk rules and lookahead most heavily covered
docs/PLAYBOOK.md the daily loop
```

---

## Design decisions worth knowing

**Risk limits are frozen and unreachable by the learning loop.**
`StrategyParams.with_overrides` raises on any field outside an explicit
whitelist, so no amount of "self-improvement" can widen risk or raise the trade
cap.

**The volume filter sits on the displacement leg, not the order-block candle.**
The usual retail formulation puts it on the block candle. Measured on 400 real
gold 1h bars, that is backwards:

| | volume vs 20-bar average |
|---|---|
| order-block candle | mean 1.01, median 0.84 — 3 of 14 above 1.4× |
| displacement candle | mean 2.15, median 1.83 — 10 of 14 above 1.4× |

With the filter on the block candle the detector found 3 zones in 399 bars and
the strategy produced **no signals at all**. The institutional footprint is the
displacement. Moved there, same 399 bars yield 10 zones.

**A zone is invalidated by a close through it, not by a touch.** An early version
treated any touch as mitigation — which made the setup logically impossible,
since "price is at the zone" and "the zone is untouched" cannot both hold. A tap
*is* the trigger. `tests/test_engine.py` guards both of these regressions.

**Insufficient data raises instead of returning nothing.** Too little history
means every bar is skipped during warm-up and the system reports no setups
forever — indistinguishable from patience, actually a dead pipeline. It now
fails loudly and says how many days it needs.

**The backtest is deliberately pessimistic.** Spread and slippage on both sides;
entry filled at the *next* bar's open; and when a bar contains both the stop and
the target, the stop is taken first. Without tick data the order is unknowable,
and assuming the good outcome is how backtests become fiction.

**Self-improvement is slow and evidence-gated.** A parameter change is proposed
only if it improved expectancy on a training window *and* on a later window it
was never fitted to, by more than a noise threshold, on a sample of 30+ trades.
Proposals are printed for you to approve — never auto-applied. Nightly retuning
on a handful of trades is curve fitting, and it reliably destroys a working
strategy.

---

## Honest limitations

- **No order execution.** The TradingView MCP has no order tool.
- **Delayed data.** TradingView bars lag 15+ minutes; execution prices must come
  from your broker.
- **No performance guarantee.** The system measures whether 60% and $500/day are
  being achieved. It cannot cause them.
- **Early evidence suggests the signal rate may be too low for $500/day.** On a
  23-day real sample it produced roughly one signal every six days. If a longer
  backtest confirms that, the right response is to lower the target — not to
  loosen filters until trades appear. The sample is currently far too small to
  conclude either way.
