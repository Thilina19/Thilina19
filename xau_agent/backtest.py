"""Bar-by-bar backtester with realistic fills and full risk-rule enforcement.

Honesty measures, since an optimistic backtest is worse than no backtest:

  1. No lookahead. At entry bar i the simulator can see bars 0..i only. Higher
     timeframe indices are resolved by timestamp so a 4h bar is only visible
     once it has closed.
  2. Spread and slippage are charged on entry and exit.
  3. When a bar's range contains both the stop and the target, the stop is
     assumed hit first. Without tick data we cannot know the order, and
     assuming the good outcome is how backtests become fiction.
  4. The daily risk rules apply in simulation exactly as they do live, so the
     backtest reflects a tradeable schedule rather than every theoretical setup.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import AgentConfig
from .data import INTERVAL_SECONDS
from .indicators import Bar
from .journal import JournalEntry, Mode, Outcome, Stats, compute_stats
from .risk import ClosedTrade, RiskManager, SessionState
from .strategy import MarketView, Rejection, Side, Signal, Strategy


def align_index(times: list[int], target_t: int) -> int:
    """Index of the newest bar in `times` that had CLOSED at or before target_t.

    `times` holds bar OPEN timestamps. A bar that opens at target_t has not
    closed yet, so it must not be visible -- hence the strict comparison and
    the step back by one.
    """
    i = bisect_right(times, target_t) - 1
    return max(-1, i - 1) if i >= 0 and times[i] == target_t else i


@dataclass
class OpenPosition:
    signal: Signal
    lots: float
    ounces: float
    entry_fill: float
    opened_at: datetime
    remaining_fraction: float = 1.0
    tp1_hit: bool = False
    stop: float = 0.0
    realised_pnl: float = 0.0
    realised_r: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    risk_distance: float = 0.0


@dataclass
class BacktestResult:
    trades: list[JournalEntry] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)
    bars_tested: int = 0
    signals_generated: int = 0
    signals_blocked_by_risk: int = 0
    # Setups that never reached the risk check because a position was already
    # running. Previously uncounted, which made "11 signals, 2 trades" look
    # inexplicable -- this is usually the real limit on trades per day.
    setups_missed_in_position: int = 0
    bars_in_position: int = 0
    block_reasons: dict[str, int] = field(default_factory=dict)
    equity_curve_r: list[float] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)

    def report(self) -> str:
        days = len(self.daily_pnl)
        avg_day = (sum(self.daily_pnl.values()) / days) if days else 0.0
        green = sum(1 for v in self.daily_pnl.values() if v > 0)
        lines = [
            "=" * 68,
            "BACKTEST RESULT",
            "=" * 68,
            f"bars tested            {self.bars_tested}",
            f"signals generated      {self.signals_generated}",
            f"blocked by risk rules  {self.signals_blocked_by_risk}",
            f"missed, position open  {self.setups_missed_in_position}",
            f"bars held in position  {self.bars_in_position}",
            f"trades taken           {self.stats.n}",
            "",
            self.stats.summary(),
            "",
            f"trading days           {days}",
            f"average day            ${avg_day:+,.2f}",
            f"green days             {green}/{days}"
            + (f" ({green / days * 100:.0f}%)" if days else ""),
            f"total P&L              ${self.stats.total_pnl:+,.2f}",
        ]
        if self.stats.by_session:
            lines += ["", "by session:"]
            for k, v in self.stats.by_session.items():
                lines.append(
                    f"  {k:<10} n={int(v['n']):<4} win={v['win_rate'] * 100:5.1f}%  "
                    f"total={v['total_r']:+.2f}R  exp={v['expectancy_r']:+.3f}R"
                )
        if self.stats.by_score_bucket:
            lines += ["", "by confluence score:"]
            for k, v in self.stats.by_score_bucket.items():
                lines.append(
                    f"  {k:<10} n={int(v['n']):<4} win={v['win_rate'] * 100:5.1f}%  "
                    f"total={v['total_r']:+.2f}R  exp={v['expectancy_r']:+.3f}R"
                )
        if self.block_reasons:
            lines += ["", "risk blocks:"]
            for k, v in sorted(self.block_reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {k:<32} {v}")
        return "\n".join(lines)


class InsufficientData(ValueError):
    """Raised when there is not enough history to form an opinion.

    This exists because the failure it prevents is silent and expensive. With
    too few higher-timeframe bars the EMA warm-up never completes, every
    evaluation is skipped, and the system reports zero setups forever -- which
    looks exactly like patient discipline. It is not; it is a broken pipeline.
    """


def check_data_sufficiency(
    entry_bars: list[Bar], zone_bars: list[Bar], bias_bars: list[Bar],
    cfg: AgentConfig, *, warmup: int = 250,
) -> None:
    """Fail loudly if history cannot support the configured indicators."""
    p = cfg.strategy
    need_bias = p.ema_slow + 50   # EMA200 plus room for structure to form
    need_zone = p.ob_volume_lookback + p.ob_max_age_bars + 20
    need_entry = warmup + 100

    problems: list[str] = []
    if len(bias_bars) < need_bias:
        hours = need_bias * (INTERVAL_SECONDS.get(p.bias_tf, 14400) / 3600)
        problems.append(
            f"{p.bias_tf} bias: have {len(bias_bars)} bars, need {need_bias} "
            f"(EMA{p.ema_slow} warm-up + structure) "
            f"= about {hours / 24:.0f} days of continuous history"
        )
    if len(zone_bars) < need_zone:
        problems.append(
            f"{p.zone_tf} zone: have {len(zone_bars)} bars, need {need_zone} "
            f"for order-block detection"
        )
    if len(entry_bars) < need_entry:
        problems.append(
            f"{p.entry_tf} entry: have {len(entry_bars)} bars, need {need_entry}"
        )

    if problems:
        raise InsufficientData(
            "not enough history to evaluate setups -- the system would silently "
            "produce zero signals:\n  - " + "\n  - ".join(problems)
            + "\n\nFix: fetch the bias timeframe separately with deep history "
              "rather than resampling a short entry-timeframe series, or lower "
              "ema_slow (structurally, with a human decision)."
        )


class Backtester:
    def __init__(self, cfg: AgentConfig, *, mode: Mode = Mode.BACKTEST):
        cfg.validate()
        self.cfg = cfg
        self.mode = mode
        self.strategy = Strategy(cfg.strategy)
        self.risk = RiskManager(cfg.risk)

    def run(
        self,
        entry_bars: list[Bar],
        zone_bars: list[Bar],
        bias_bars: list[Bar],
        *,
        news_times: set[int] | None = None,
        warmup: int = 250,
        require_sufficient_data: bool = True,
    ) -> BacktestResult:
        p = self.cfg.strategy
        if require_sufficient_data:
            check_data_sufficiency(entry_bars, zone_bars, bias_bars, self.cfg,
                                   warmup=warmup)
        res = BacktestResult()

        entry_view = MarketView.build(entry_bars, p)
        zone_view = MarketView.build(zone_bars, p)
        bias_view = MarketView.build(bias_bars, p)

        zone_times = [b.t for b in zone_bars]
        bias_times = [b.t for b in bias_bars]

        state = SessionState(trading_day=entry_bars[warmup].dt.date())
        pos: OpenPosition | None = None
        cum_r = 0.0

        for i in range(warmup, len(entry_bars)):
            bar = entry_bars[i]
            now = bar.dt
            state.roll_to_day(now.date())
            res.bars_tested += 1

            # ---- manage an open position before looking for a new one
            if pos is not None:
                closed = self._manage(pos, bar)
                if closed is not None:
                    res.trades.append(closed)
                    cum_r += closed.r_multiple
                    res.equity_curve_r.append(cum_r)
                    day = now.date().isoformat()
                    res.daily_pnl[day] = res.daily_pnl.get(day, 0.0) + closed.pnl
                    state.register_close(
                        ClosedTrade(
                            opened_at=pos.opened_at, closed_at=now, side=pos.signal.side.value,
                            entry=pos.entry_fill, exit=closed.exit_price or bar.c,
                            pnl=closed.pnl, r_multiple=closed.r_multiple,
                        )
                    )
                    pos = None
                else:
                    # Still in a trade. Count what we are walking past, because
                    # trade DURATION, not signal frequency, is often what caps
                    # trades per day.
                    res.bars_in_position += 1
                    zi_p = align_index(zone_times, bar.t)
                    bi_p = align_index(bias_times, bar.t)
                    if zi_p >= 50 and bi_p >= p.ema_slow:
                        peek = self.strategy.evaluate(
                            entry_view=entry_view, entry_idx=i,
                            bias_view=bias_view, bias_idx=bi_p,
                            zone_view=zone_view, zone_idx=zi_p,
                        )
                        if not isinstance(peek, Rejection):
                            res.setups_missed_in_position += 1
                    continue  # one position at a time

            # ---- look for a setup
            zi = align_index(zone_times, bar.t)
            bi = align_index(bias_times, bar.t)
            if zi < 50 or bi < p.ema_slow:
                continue

            outcome = self.strategy.evaluate(
                entry_view=entry_view, entry_idx=i,
                bias_view=bias_view, bias_idx=bi,
                zone_view=zone_view, zone_idx=zi,
                news_blackout=bool(news_times and bar.t in news_times),
            )
            if isinstance(outcome, Rejection):
                res.rejections.append(outcome)
                continue

            sig = outcome
            res.signals_generated += 1

            decision = self.risk.check_can_trade(
                state, now=now, entry=sig.entry, stop=sig.stop, target=sig.tp2
            )
            if not decision.allowed:
                res.signals_blocked_by_risk += 1
                for r in decision.reasons:
                    res.block_reasons[r.value] = res.block_reasons.get(r.value, 0) + 1
                continue

            size = self.risk.size_position(sig.entry, sig.stop)
            if size is None:
                continue

            # Entry is filled at the next bar's open plus costs -- we cannot
            # transact at the close of the bar that produced the signal.
            if i + 1 >= len(entry_bars):
                break
            nxt = entry_bars[i + 1]
            cost = p.spread_usd + p.slippage_usd
            fill = nxt.o + cost if sig.side is Side.LONG else nxt.o - cost

            pos = OpenPosition(
                signal=sig, lots=size.lots, ounces=size.ounces, entry_fill=fill,
                opened_at=nxt.dt, stop=sig.stop,
                risk_distance=abs(fill - sig.stop),
            )
            state.register_open()

        res.stats = compute_stats(res.trades)
        return res

    # ------------------------------------------------------------- position mgmt

    def _manage(self, pos: OpenPosition, bar: Bar) -> JournalEntry | None:
        """Walk one bar of an open position. Returns a JournalEntry on close.

        Pessimistic ordering: if both the stop and a target sit inside this
        bar's range, the stop is taken first.
        """
        p = self.cfg.strategy
        sig = pos.signal
        long = sig.side is Side.LONG
        cost = p.spread_usd + p.slippage_usd
        rd = pos.risk_distance or 1e-9

        # Track excursions for post-trade analysis.
        adverse = (pos.entry_fill - bar.l) if long else (bar.h - pos.entry_fill)
        favour = (bar.h - pos.entry_fill) if long else (pos.entry_fill - bar.l)
        pos.mae_r = max(pos.mae_r, adverse / rd)
        pos.mfe_r = max(pos.mfe_r, favour / rd)

        hit_stop = bar.l <= pos.stop if long else bar.h >= pos.stop
        hit_tp1 = (bar.h >= sig.tp1 if long else bar.l <= sig.tp1) and not pos.tp1_hit
        hit_tp2 = bar.h >= sig.tp2 if long else bar.l <= sig.tp2

        if hit_stop:
            exit_px = pos.stop - cost if long else pos.stop + cost
            return self._close(pos, bar, exit_px)

        if hit_tp1:
            frac = p.tp1_close_fraction
            px = sig.tp1 - cost if long else sig.tp1 + cost
            oz = pos.ounces * frac
            pnl = (px - pos.entry_fill) * oz if long else (pos.entry_fill - px) * oz
            pos.realised_pnl += pnl
            pos.realised_r += (abs(px - pos.entry_fill) / rd) * frac
            pos.remaining_fraction -= frac
            pos.tp1_hit = True
            if p.move_sl_to_be_at_tp1:
                # Breakeven plus costs, so a pullback to entry is not a loss.
                pos.stop = pos.entry_fill + cost if long else pos.entry_fill - cost

        if hit_tp2 and pos.remaining_fraction > 0:
            px = sig.tp2 - cost if long else sig.tp2 + cost
            return self._close(pos, bar, px)

        return None

    def _close(self, pos: OpenPosition, bar: Bar, exit_px: float) -> JournalEntry:
        sig = pos.signal
        long = sig.side is Side.LONG
        rd = pos.risk_distance or 1e-9
        oz = pos.ounces * pos.remaining_fraction

        pnl = (exit_px - pos.entry_fill) * oz if long else (pos.entry_fill - exit_px) * oz
        total_pnl = pos.realised_pnl + pnl

        signed = (exit_px - pos.entry_fill) if long else (pos.entry_fill - exit_px)
        total_r = pos.realised_r + (signed / rd) * pos.remaining_fraction

        if total_pnl > 0.5:
            outcome = Outcome.WIN
        elif total_pnl < -0.5:
            outcome = Outcome.LOSS
        else:
            outcome = Outcome.BREAKEVEN

        return JournalEntry(
            signal_time=sig.dt, side=sig.side.value, entry=pos.entry_fill,
            stop=sig.stop, tp1=sig.tp1, tp2=sig.tp2, score=sig.score,
            session=sig.session, mode=self.mode.value, lots=pos.lots,
            exit_price=exit_px, exit_time=bar.dt, pnl=total_pnl,
            r_multiple=total_r, outcome=outcome.value,
            mae_r=pos.mae_r, mfe_r=pos.mfe_r,
            notes=f"tp1_hit={pos.tp1_hit}",
        )


def walk_forward_split(
    bars: list[Bar], *, folds: int = 4, train_frac: float = 0.7
) -> list[tuple[list[Bar], list[Bar]]]:
    """Anchored walk-forward folds of (train, test).

    Tuning on one period and verifying on the next is the only way to tell an
    edge from a curve fit. Each fold's test window is strictly after its train
    window; nothing is ever tested on data it was fitted to.
    """
    n = len(bars)
    if folds < 1 or n < 400:
        return []
    out: list[tuple[list[Bar], list[Bar]]] = []
    seg = n // (folds + 1)
    for f in range(1, folds + 1):
        train_end = seg * f
        test_end = min(n, seg * (f + 1))
        train = bars[:train_end]
        test = bars[int(train_end - train_end * (1 - train_frac) * 0.0):test_end]
        if len(train) > 300 and len(test) > 100:
            out.append((train, test))
    return out
