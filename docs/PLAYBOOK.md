# Daily Playbook

The loop Claude and you run together. The Python engine does the deterministic
work (structure, sizing, statistics); Claude fetches data through the
TradingView connector and handles the judgement calls that surround it.

**Claude cannot place trades.** The TradingView MCP has no order tool. Every
entry below is something you place yourself, after reading it.

---

## Hard rules (these never bend)

| Rule | Value | Why |
|---|---|---|
| Risk per trade | 0.5% = **$250** | One 2R winner meets the daily target |
| Daily target | **+$500** | Hit it and stop. Giving profit back is a loss |
| Daily loss cap | **-$500** | Hit it and stop. No exceptions, ever |
| Max trades/day | **3** | Quality over quantity |
| Consecutive losses | **2 then done** | The edge is not present today |
| Cool-off after a loss | **30 min** | This is the anti-revenge rule |
| Open positions | **1** | No grid, no averaging down, no hedging |
| Minimum R:R | **1.8** | Below this the win rate has to carry everything |
| Weekly / monthly stop | **-3% / -6%** | Survive the bad stretch |

The engine enforces all of these in code. `RiskLimits` is frozen and the
self-review loop is structurally forbidden from touching it — it can only
propose changes to strategy parameters, never to risk.

---

## Morning routine (~10 minutes)

**1. Check where you stand.**
```bash
python3 -m xau_agent.cli status
```
Reports mode (PAPER or LIVE), current stats, and how many trades remain before
the promotion gate can open.

**2. Ask Claude for the session brief.** Claude fetches:
- 15m, 1h and 4h bars for `OANDA:XAUUSD`
- the economic calendar for the day (CPI, NFP, FOMC, PPI, jobless claims)
- overnight gold headlines

**3. Mark the news blackout.** No new entry within **30 minutes either side** of
a high-impact release. Gold's reaction to CPI and FOMC is not tradeable with
structure — it is a coin flip with a wide spread.

**4. Note the day's number.** $500 target, $500 loss cap, 3 trades maximum.
Write it down. The point of writing it is that you have committed before the
market has a chance to argue.

---

## During the session

Ask Claude to analyse whenever price approaches a marked zone, or run it
yourself once you have exported fresh bars:

```bash
python3 -m xau_agent.cli analyse --entry data/XAUUSD_15m.csv \
    --zone data/XAUUSD_1h.csv --bias data/XAUUSD_4h.csv
```

You get either a full trade plan (entry, SL, TP1, TP2, lots, dollar risk) or a
scorecard showing exactly which condition failed.

### The setup, in words

1. **4h decides direction.** Higher highs and higher lows, or lower highs and
   lower lows. If price is on the wrong side of both the 50 and 200 EMA, the
   bias is vetoed and there is no trade in either direction.
2. **1h supplies the zone.** The last opposing candle before a displacement leg
   of at least 1.3 ATR *whose displacement carried at least 1.4x average
   volume*. A zone price has already closed through is dead.
3. **15m times the entry.** Two clean Heikin Ashi candles in the direction of
   bias, neither mostly wick. Heikin Ashi is the trigger only — every price
   level comes from real OHLC, never from the synthetic HA values.
4. **Score must reach 70/100.** Bias and the HA trigger are mandatory; scoring
   zero on either is an automatic no regardless of everything else.
5. **Asian session pays a 12-point penalty.** Gold produces its most false
   breaks between 00:00 and 07:00 UTC. Not banned, just held to a higher bar.

### Stops and targets

- **SL** goes beyond *both* the recent swing and the far edge of the order
  block, plus 0.55 ATR. Taking the further of the two stops a stop from sitting
  inside the zone that is supposed to defend it.
- **TP1** at 1R — close half, move the stop to breakeven plus costs.
- **TP2** at 2.2R — the remainder.

### Before you click

Read this list. Every item is a real way traders lose money that the engine
cannot see:

- [ ] Is the score ≥ 70 and R:R ≥ 1.8?
- [ ] Is this trade 1, 2 or 3 today? (There is no 4.)
- [ ] Is the 30-minute cool-off clear?
- [ ] Am I inside a news blackout?
- [ ] Is my loss budget for today still intact?
- [ ] **Am I taking this because it qualified, or because I want to be in?**

That last one has no code behind it. It is the one that actually decides
whether this works.

---

## After every trade

Log it. An unlogged trade teaches the system nothing, and the journal is the
only thing standing between "self-improving" and "guessing".

```bash
python3 -m xau_agent.cli log-trade --side long \
  --entry 4201.50 --stop 4189.00 --tp1 4214.00 --tp2 4229.00 \
  --exit 4229.00 --pnl 512.50 --lots 0.20 --score 78 \
  --session newyork --mode paper --notes "clean london OB tap"
```

Log the paper trades too. Those are what open the gate.

---

## End of day

```bash
python3 -m xau_agent.cli review --entry data/XAUUSD_15m.csv
```

The review reports four things:

1. **Performance** — win rate with a confidence interval, expectancy in R,
   drawdown.
2. **Promotion gate** — whether the agent may publish live entries yet.
3. **Discipline audit** — this one audits *you*. It flags days that exceeded the
   trade cap, entries taken inside the cool-off, and days that ran past the loss
   limit. Those cannot come from the system, so they came from you.
4. **Parameter proposals** — only changes that improved results on a training
   window *and* on a later window they were not fitted to. Usually none, which
   is the correct answer. Proposals are printed, never auto-applied.

---

## How the agent earns the right to publish live entries

You asked for live entries once it reaches a 60% win rate. Holding that promise
honestly means refusing to believe a win rate until the sample can support it.

The gate tests the **lower bound of the 95% Wilson confidence interval**, not
the raw percentage:

| Trades | 60% observed | CI lower bound |
|---|---|---|
| 10 | 6 wins | 26% — meaningless |
| 40 | 24 wins | 44.6% |
| 60 | 36 wins | 47.4% |
| 100 | 60 wins | 50.2% |

**Gate to publish live entries** — all four must hold:
- ≥ **60** forward-tested (paper) trades
- CI lower bound ≥ **45%**
- expectancy ≥ **+0.20R**
- peak drawdown ≤ **8R**

Why 45% and not 60%? With TP1 at 1R and TP2 at 2.2R the average winner is about
1.6R, so breakeven is `1/(1+1.6)` ≈ **38%**. A 45% floor clears breakeven with
real margin. Requiring the CI lower bound to clear 60% would need roughly 250
trades — a year of forward testing before a single live entry.

So the gate opens at a level that is genuinely profitable and reachable, and the
report keeps telling you that 60% itself is not yet *proven* until ~100 trades.
Backtest trades never count toward the gate. Only forward-tested ones do.

The gate is not a one-way door. If performance decays, the agent demotes itself
back to paper on the next review.

---

## What this system will not do

Stated plainly, because a tool that overstates itself is dangerous:

- **It cannot place orders.** No broker connection exists.
- **It cannot guarantee 60%, or $500 a day, or any return.** It measures whether
  those are being achieved and stays quiet until the evidence exists.
- **TradingView data is delayed 15+ minutes.** Structure analysis is fine on
  that; your execution price must come from your broker's live feed.
- **It cannot stop you from overriding it.** It can only record that you did,
  and show you the cost at the next review.

---

## The $500/day question, honestly

On $50,000, $500/day is **1% daily** — roughly 20% a month compounded. Very few
traders sustain that. The per-trade arithmetic is sane (one 2R winner at 0.5%
risk), but the *consistency* required is not typical.

Treat $500 as a good day, not a quota. A quota is what makes someone take trade
number four, and trade number four is where the account goes.

Two things the early data already suggests you should watch:

1. **Signal frequency may be too low for the target.** On a 23-day real-gold
   sample the strategy produced roughly one signal per six days at a threshold
   of 60. If that holds up over a longer backtest, then $500/day is arithmetically
   unreachable at this selectivity, and the honest response is to lower the
   target — not to loosen the filters until trades appear.
2. **The sample so far proves nothing either way.** Six tradeable days cannot
   establish a frequency any more than ten trades can establish a win rate.

Run the long backtest before drawing any conclusion. See the README for how to
export the history that makes that possible.
