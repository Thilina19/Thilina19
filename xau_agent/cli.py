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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
