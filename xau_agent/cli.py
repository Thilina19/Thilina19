"""Command line interface.

    python3 -m xau_agent.cli analyse   --entry data/XAUUSD_15m.csv [--zone ...] [--bias ...]
    python3 -m xau_agent.cli backtest  --entry data/XAUUSD_15m.csv
    python3 -m xau_agent.cli status
    python3 -m xau_agent.cli review    --entry data/XAUUSD_15m.csv
    python3 -m xau_agent.cli log-trade --side long --entry 4200 --stop 4188 ...

`analyse` prints the current trade plan, or the reason there isn't one. It
refuses to publish an actionable entry while the promotion gate has the agent
in paper mode -- that refusal is the feature, not a limitation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

from .backtest import Backtester, InsufficientData, align_index, check_data_sufficiency
from .config import AgentConfig, RiskLimits, StrategyParams
from .data import describe, load_csv, load_json, resample
from .indicators import Bar
from .journal import (
    Journal, JournalEntry, Mode, Outcome, PerformanceGate, compute_stats,
)
from .review import Reviewer
from .risk import RiskManager
from .strategy import MarketView, Rejection, Side, Strategy


def _load(path: str) -> list[Bar]:
    if path.endswith(".json"):
        return load_json(path)
    return load_csv(path)


def _series(args: argparse.Namespace) -> tuple[list[Bar], list[Bar], list[Bar]]:
    """Load the three timeframes, resampling from the entry series if needed."""
    entry = _load(args.entry)
    if not entry:
        raise SystemExit(f"no usable bars in {args.entry}")
    zone = _load(args.zone) if args.zone else resample(entry, args.zone_tf)
    bias = _load(args.bias) if args.bias else resample(entry, args.bias_tf)
    return entry, zone, bias


def _config(args: argparse.Namespace) -> AgentConfig:
    cfg = AgentConfig(
        risk=RiskLimits(account_equity=args.equity,
                        risk_per_trade_pct=args.risk_pct,
                        daily_profit_target=args.target,
                        daily_loss_limit=args.max_loss),
        strategy=StrategyParams(zone_tf=args.zone_tf, bias_tf=args.bias_tf,
                                entry_tf=args.entry_tf,
                                min_confluence_score=args.min_score),
    )
    cfg.validate()
    return cfg


# ----------------------------------------------------------------- commands


def cmd_analyse(args: argparse.Namespace) -> int:
    cfg = _config(args)
    entry, zone, bias = _series(args)

    print(describe(entry, f"entry {cfg.strategy.entry_tf}"))
    print(describe(zone, f"zone  {cfg.strategy.zone_tf}"))
    print(describe(bias, f"bias  {cfg.strategy.bias_tf}"))
    print()

    try:
        check_data_sufficiency(entry, zone, bias, cfg)
    except InsufficientData as e:
        print(f"CANNOT ANALYSE\n{e}")
        return 2

    strat = Strategy(cfg.strategy)
    ev = MarketView.build(entry, cfg.strategy)
    zv = MarketView.build(zone, cfg.strategy)
    bv = MarketView.build(bias, cfg.strategy)

    i = len(entry) - 1
    zi = align_index([b.t for b in zone], entry[i].t)
    bi = align_index([b.t for b in bias], entry[i].t)

    out = strat.evaluate(entry_view=ev, entry_idx=i, bias_view=bv, bias_idx=bi,
                         zone_view=zv, zone_idx=zi)

    # What mode are we in? This decides whether a signal is actionable.
    with Journal(args.db) as j:
        gate = PerformanceGate(cfg).evaluate(j.trades())
        state = j.load_state(date.today())

    print(gate.report())
    print()

    if isinstance(out, Rejection):
        print(f"NO SETUP at {entry[i].dt:%Y-%m-%d %H:%M} UTC "
              f"(price {entry[i].c:.2f})")
        print(f"  reason: {out.reason}")
        if out.items:
            print("  scorecard:")
            for it in out.items:
                print(f"    {'+' if it.passed else '-'} {it.name}: "
                      f"{it.points}/{it.max_points}  {it.detail}")
        print("\nNo trade is the correct output most of the time.")
        return 0

    rm = RiskManager(cfg.risk)
    size = rm.size_position(out.entry, out.stop)
    decision = rm.check_can_trade(state, now=datetime.now(tz=timezone.utc),
                                  entry=out.entry, stop=out.stop, target=out.tp2)

    print("SETUP FOUND")
    print(out.explain())
    print()
    print(f"  narrative: {out.narrative}")
    print()
    if size:
        print(f"  position size   {size.lots:.2f} lots ({size.ounces:.0f} oz)")
        print(f"  risk if stopped ${size.risk_dollars:.2f}")
        print(f"  reward at TP2   ${abs(out.tp2 - out.entry) * size.ounces:.2f}")
    print(f"  {rm.progress_note(state)}")
    print()

    if not decision.allowed:
        print(f"RISK RULES BLOCK THIS TRADE -- {decision.summary}")
        for n in decision.notes:
            print(f"  - {n}")
        return 0

    if not gate.passed:
        print("PAPER MODE -- log this as a paper trade, do NOT place it live.")
        print("  The agent has not yet earned the right to publish live entries.")
    else:
        print("LIVE -- this entry is actionable. Place it with the SL and TP above.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _config(args)
    entry, zone, bias = _series(args)
    print(describe(entry, "entry"), "\n" + describe(zone, "zone"),
          "\n" + describe(bias, "bias"), "\n")
    try:
        res = Backtester(cfg).run(entry, zone, bias)
    except InsufficientData as e:
        print(f"CANNOT BACKTEST\n{e}")
        return 2
    print(res.report())

    if args.save:
        with Journal(args.db) as j:
            n = j.record_many(res.trades)
        print(f"\nsaved {n} backtest trades to {args.db}")
    if res.stats.n < 30:
        print("\nWARNING: fewer than 30 trades. These numbers carry no weight.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _config(args)
    with Journal(args.db) as j:
        trades = j.trades()
        gate = PerformanceGate(cfg).evaluate(trades)
        state = j.load_state(date.today())
    print(gate.report())
    print()
    print(RiskManager(cfg.risk).progress_note(state))
    paper = [t for t in trades if t.mode == Mode.PAPER.value]
    live = [t for t in trades if t.mode == Mode.LIVE.value]
    print(f"journal: {len(trades)} closed trades "
          f"({len(paper)} paper, {len(live)} live)")
    need = cfg.min_trades_for_live - len([t for t in trades
                                          if t.mode in (Mode.PAPER.value,
                                                        Mode.LIVE.value)])
    if need > 0:
        print(f"{need} more forward-tested trades before the gate can open.")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    cfg = _config(args)
    entry = zone = bias = None
    if args.entry:
        try:
            entry, zone, bias = _series(args)
        except SystemExit:
            pass
    with Journal(args.db) as j:
        report = Reviewer(cfg).review(j, entry_bars=entry, zone_bars=zone,
                                      bias_bars=bias)
    print(report.render())
    return 0


def cmd_log_trade(args: argparse.Namespace) -> int:
    """Record a trade you actually took, so the agent learns from reality."""
    outcome = Outcome.OPEN.value
    if args.exit is not None:
        if args.pnl is None:
            raise SystemExit("--pnl is required when --exit is given")
        outcome = (Outcome.WIN.value if args.pnl > 0.5 else
                   Outcome.LOSS.value if args.pnl < -0.5 else
                   Outcome.BREAKEVEN.value)
    risk = abs(args.entry - args.stop)
    r = 0.0
    if args.exit is not None and risk > 0:
        signed = (args.exit - args.entry) if args.side == "long" else (args.entry - args.exit)
        r = signed / risk

    e = JournalEntry(
        signal_time=datetime.fromisoformat(args.time) if args.time
        else datetime.now(tz=timezone.utc),
        side=args.side, entry=args.entry, stop=args.stop,
        tp1=args.tp1 if args.tp1 is not None else args.entry,
        tp2=args.tp2 if args.tp2 is not None else args.entry,
        score=args.score, session=args.session, mode=args.mode,
        lots=args.lots, exit_price=args.exit,
        exit_time=datetime.now(tz=timezone.utc) if args.exit is not None else None,
        pnl=args.pnl or 0.0, r_multiple=r, outcome=outcome, notes=args.notes,
    )
    with Journal(args.db) as j:
        tid = j.record(e)
        stats = compute_stats(j.trades())
    print(f"logged trade #{tid}: {outcome} {r:+.2f}R")
    print(stats.summary())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Measure a feed's displacement and volume distributions.

    Run this on any new symbol or broker feed BEFORE trusting the thresholds.
    Volume is reported differently by every provider, and a feed whose volume is
    smoothed has little dispersion -- on such a feed the volume filter never
    triggers, no order blocks are found, and the system silently reports no
    setups forever. That failure looks exactly like discipline.

    Reference, measured on real gold 1h data:
        OANDA       volume ratio p90 1.90, max 6.53, 20.6% clear 1.4x
        CAPITALCOM  volume ratio p90 1.54, max 2.14, 16.7% clear 1.4x
    """
    from .indicators import atr as _atr, volume_ratio as _vr
    from .structure import find_order_blocks

    bars = _load(args.entry)
    if len(bars) < 60:
        raise SystemExit("need at least 60 bars to say anything useful")
    p = _config(args).strategy

    a = _atr(bars, p.atr_period)
    vr = _vr(bars, p.ob_volume_lookback)
    disp = [bars[i].body / a[i] for i in range(len(bars)) if a[i]]
    vols = [v for v in vr if v]
    atrp = [a[i] / bars[i].c * 100 for i in range(len(bars)) if a[i]]

    def q(xs: list[float], f: float) -> float:
        return sorted(xs)[min(len(xs) - 1, int(len(xs) * f))]

    print(describe(bars, "feed"))
    print()
    print(f"displacement (body / ATR)")
    print(f"  median {q(disp, .5):.2f}   p90 {q(disp, .9):.2f}   "
          f"p99 {q(disp, .99):.2f}   max {max(disp):.2f}")
    for th in (0.8, 1.0, 1.3, 1.5, 2.0):
        share = 100 * sum(1 for x in disp if x >= th) / len(disp)
        mark = "  <- configured" if abs(th - p.ob_displacement_atr) < 0.01 else ""
        print(f"   >= {th:.1f} ATR : {share:5.1f}% of bars{mark}")

    print()
    print(f"volume ratio (bar volume / {p.ob_volume_lookback}-bar average)")
    print(f"  mean {sum(vols) / len(vols):.2f}   median {q(vols, .5):.2f}   "
          f"p90 {q(vols, .9):.2f}   max {max(vols):.2f}")
    for th in (1.2, 1.4, 1.8, 2.5):
        share = 100 * sum(1 for x in vols if x >= th) / len(vols)
        mark = "  <- configured" if abs(th - p.ob_min_volume_ratio) < 0.01 else ""
        print(f"   >= {th:.1f}x    : {share:5.1f}% of bars{mark}")

    print()
    print(f"ATR as % of price: median {q(atrp, .5):.3f}%  "
          f"min {min(atrp):.3f}%  max {max(atrp):.3f}%")
    print(f"   configured band {p.min_atr_pct * 100:.3f}% - {p.max_atr_pct * 100:.3f}%")
    outside = sum(1 for x in atrp
                  if x < p.min_atr_pct * 100 or x > p.max_atr_pct * 100)
    print(f"   {100 * outside / len(atrp):.1f}% of bars fall outside it "
          f"(those are skipped)")

    blocks = find_order_blocks(
        bars, len(bars) - 1, atr_values=a, vol_ratios=vr,
        displacement_atr=p.ob_displacement_atr,
        min_volume_ratio=p.ob_min_volume_ratio,
        volume_lookback=p.ob_volume_lookback,
        max_age_bars=max(p.ob_max_age_bars, len(bars) - 10),
    )
    live = [b for b in blocks if not b.mitigated]
    print()
    print(f"order blocks at current settings: {len(blocks)} found, "
          f"{len(live)} still valid")
    if blocks:
        per = len(bars) / len(blocks)
        print(f"   about one zone every {per:.0f} bars")

    print()
    if not blocks:
        print("VERDICT: no zones found. The thresholds are too strict for this "
              "feed, or\n  its volume is too smoothed to carry the filter. Lower "
              "--min-vol-ratio\n  toward the p90 above before trusting any result.")
        return 1
    if max(vols) < p.ob_min_volume_ratio * 1.2:
        print(f"WARNING: the highest volume ratio in this sample is only "
              f"{max(vols):.2f}, barely\n  above the {p.ob_min_volume_ratio:.2f} "
              f"threshold. This feed's volume is smoothed;\n  raising the "
              f"threshold further will silence the system entirely.")
    else:
        print("VERDICT: thresholds are usable on this feed.")
    return 0


def cmd_export_pine(args: argparse.Namespace) -> int:
    """Generate a Pine overlay drawing the journalled signals."""
    from .pine_export import MAX_TRADES, write_pine

    with Journal(args.db) as j:
        trades = j.trades(mode=args.mode, closed_only=False)
    if not trades:
        print("The journal is empty, so there is nothing to draw.")
        print("Log trades first:  python3 -m xau_agent.cli log-trade --help")
        print(f"\nFor the logic overlay instead, use pine/xau_agent_strategy.pine "
              f"-- it recomputes across all history and needs no journal.")
        return 1

    path, drawn = write_pine(trades, args.out, symbol_note=args.symbol)
    print(f"wrote {path}  ({drawn} of {len(trades)} trades drawn)")
    if len(trades) > MAX_TRADES:
        print(f"  TradingView caps drawings at 500 per type, so only the newest "
              f"{MAX_TRADES} are included.")
    print("\nNext: open the symbol in TradingView, Pine Editor -> paste the file "
          "-> Save -> Add to chart.")
    return 0


def cmd_alert_levels(args: argparse.Namespace) -> int:
    """Print the alert levels for the current setup, ready to create.

    TradingView's MCP interface only accepts simple price conditions -- no
    indicator alerts and no webhooks -- so a signal becomes three price alerts:
    one at the entry, one at the stop, one at the target.
    """
    cfg = _config(args)
    entry, zone, bias = _series(args)
    try:
        check_data_sufficiency(entry, zone, bias, cfg)
    except InsufficientData as e:
        print(f"CANNOT ANALYSE\n{e}")
        return 2

    strat = Strategy(cfg.strategy)
    ev = MarketView.build(entry, cfg.strategy)
    zv = MarketView.build(zone, cfg.strategy)
    bv = MarketView.build(bias, cfg.strategy)
    i = len(entry) - 1
    out = strat.evaluate(
        entry_view=ev, entry_idx=i,
        bias_view=bv, bias_idx=align_index([b.t for b in bias], entry[i].t),
        zone_view=zv, zone_idx=align_index([b.t for b in zone], entry[i].t),
    )
    if isinstance(out, Rejection):
        print(f"No setup, so no alerts to set: {out.reason}")
        return 0

    rm = RiskManager(cfg.risk)
    size = rm.size_position(out.entry, out.stop)
    print(f"{out.side.value.upper()} setup, score {out.score}/100\n")
    print("Create these three price alerts:")
    print(f"  1. ENTRY  {out.entry:.2f}   condition: "
          f"{'cross_down' if out.side is Side.LONG else 'cross_up'}")
    print(f"  2. STOP   {out.stop:.2f}   condition: "
          f"{'cross_down' if out.side is Side.LONG else 'cross_up'}")
    print(f"  3. TP2    {out.tp2:.2f}   condition: "
          f"{'cross_up' if out.side is Side.LONG else 'cross_down'}")
    if size:
        print(f"\nsize {size.lots:.2f} lots ({size.ounces:.0f} oz), "
              f"risking ${size.risk_dollars:.2f}")
    print("\nAsk Claude to create them, or add them on tradingview.com.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xau_agent",
        description="Disciplined XAUUSD signal agent. Generates trade plans; "
                    "never places orders.",
    )
    p.add_argument("--db", default="data/journal.db")
    p.add_argument("--equity", type=float, default=50_000.0)
    p.add_argument("--risk-pct", dest="risk_pct", type=float, default=0.005)
    p.add_argument("--target", type=float, default=500.0)
    p.add_argument("--max-loss", dest="max_loss", type=float, default=500.0)
    p.add_argument("--entry-tf", dest="entry_tf", default="15m")
    p.add_argument("--zone-tf", dest="zone_tf", default="1h")
    p.add_argument("--bias-tf", dest="bias_tf", default="4h")
    p.add_argument("--min-score", dest="min_score", type=int, default=70)

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data(sp: argparse.ArgumentParser, entry_required: bool = True) -> None:
        sp.add_argument("--entry", required=entry_required,
                        help="CSV/JSON of entry-timeframe bars")
        sp.add_argument("--zone", help="zone-timeframe bars (default: resampled)")
        sp.add_argument("--bias", help="bias-timeframe bars (default: resampled)")

    a = sub.add_parser("analyse", help="evaluate the current bar for a setup")
    add_data(a)
    a.set_defaults(func=cmd_analyse)

    b = sub.add_parser("backtest", help="run the strategy over history")
    add_data(b)
    b.add_argument("--save", action="store_true",
                   help="write trades to the journal as backtest mode")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("status", help="show performance and promotion status")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("review", help="run the daily self-review")
    add_data(r, entry_required=False)
    r.set_defaults(func=cmd_review)

    lt = sub.add_parser("log-trade", help="record a trade you took")
    lt.add_argument("--side", choices=["long", "short"], required=True)
    lt.add_argument("--entry", type=float, required=True)
    lt.add_argument("--stop", type=float, required=True)
    lt.add_argument("--tp1", type=float)
    lt.add_argument("--tp2", type=float)
    lt.add_argument("--exit", type=float, help="exit price if the trade is closed")
    lt.add_argument("--pnl", type=float, help="realised P&L in USD")
    lt.add_argument("--lots", type=float, default=0.0)
    lt.add_argument("--score", type=int, default=0)
    lt.add_argument("--session", default="unknown")
    lt.add_argument("--mode", default=Mode.PAPER.value,
                    choices=[m.value for m in Mode])
    lt.add_argument("--time", help="signal time in ISO format")
    lt.add_argument("--notes", default="")
    lt.set_defaults(func=cmd_log_trade)

    ep = sub.add_parser("export-pine",
                        help="draw the journalled signals on a TradingView chart")
    ep.add_argument("--out", default="pine/xau_agent_journal.pine")
    ep.add_argument("--mode", choices=[m.value for m in Mode],
                    help="only export this mode (default: all)")
    ep.add_argument("--symbol", default="CAPITALCOM:XAUUSD")
    ep.set_defaults(func=cmd_export_pine)

    cb = sub.add_parser("calibrate",
                        help="measure a feed's volume/displacement distributions")
    cb.add_argument("--entry", required=True, help="bars from the feed to check")
    cb.add_argument("--zone")
    cb.add_argument("--bias")
    cb.set_defaults(func=cmd_calibrate)

    al = sub.add_parser("alert-levels",
                        help="print the price alerts for the current setup")
    add_data(al)
    al.set_defaults(func=cmd_alert_levels)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
