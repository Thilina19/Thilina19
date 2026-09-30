"""The self-improvement loop -- deliberately slow and evidence-gated.

The naive version of "self-learning" retunes parameters nightly on the last
few trades. That is not learning, it is curve fitting with extra steps, and it
reliably destroys a working strategy: the parameters chase the last week's noise
and arrive just in time for the regime to change.

So this module is built around refusal. A parameter change is proposed only if:

  1. There is a minimum sample behind the observation (MIN_SAMPLE trades).
  2. The change improves expectancy on a TRAIN window, and
  3. it still improves expectancy on a later TEST window it was not fitted to,
  4. by a margin that exceeds what noise would produce.

Anything else is reported as an observation for the human to read, not applied.
Risk limits are never touched -- `StrategyParams.with_overrides` rejects them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import AgentConfig, StrategyParams
from .indicators import Bar
from .journal import (
    GateResult, Journal, JournalEntry, Mode, PerformanceGate, Stats, compute_stats,
)

# No proposal is made on a thinner sample than this.
MIN_SAMPLE = 30
# Expectancy must improve by at least this much, in R, on the unseen window.
MIN_EDGE_IMPROVEMENT = 0.05


@dataclass
class Observation:
    """Something the data says. Not necessarily something to act on."""

    topic: str
    finding: str
    sample: int
    actionable: bool = False

    def __str__(self) -> str:
        tag = "ACTION" if self.actionable else "note  "
        return f"[{tag}] {self.topic}: {self.finding} (n={self.sample})"


@dataclass
class Proposal:
    """A parameter change that survived walk-forward validation."""

    param: str
    current: Any
    proposed: Any
    train_expectancy: float
    test_expectancy: float
    baseline_test_expectancy: float
    rationale: str

    @property
    def improvement(self) -> float:
        return self.test_expectancy - self.baseline_test_expectancy

    def __str__(self) -> str:
        return (
            f"{self.param}: {self.current} -> {self.proposed}  "
            f"(test expectancy {self.baseline_test_expectancy:+.3f}R -> "
            f"{self.test_expectancy:+.3f}R, +{self.improvement:.3f}R)\n"
            f"    {self.rationale}"
        )


@dataclass
class ReviewReport:
    run_at: datetime
    stats: Stats
    gate: GateResult
    observations: list[Observation] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    rejection_summary: dict[str, int] = field(default_factory=dict)
    discipline: list[str] = field(default_factory=list)

    def render(self) -> str:
        L = [
            "=" * 68,
            f"DAILY REVIEW  {self.run_at:%Y-%m-%d %H:%M} UTC",
            "=" * 68,
            "",
            "PERFORMANCE",
            f"  {self.stats.summary()}",
            "",
            "PROMOTION GATE",
            *(f"  {line}" for line in self.gate.report().splitlines()),
        ]
        if self.discipline:
            L += ["", "DISCIPLINE"] + [f"  - {d}" for d in self.discipline]
        if self.observations:
            L += ["", "OBSERVATIONS"] + [f"  {o}" for o in self.observations]
        if self.proposals:
            L += ["", "VALIDATED PARAMETER PROPOSALS"] + [
                f"  {p}" for p in self.proposals
            ]
        else:
            L += ["", "VALIDATED PARAMETER PROPOSALS", "  none -- no change survived "
                  "walk-forward validation. Parameters stay as they are."]
        if self.rejection_summary:
            L += ["", "WHY SETUPS WERE DECLINED (top 8)"]
            for k, v in list(self.rejection_summary.items())[:8]:
                L.append(f"  {v:>5}  {k}")
        return "\n".join(L)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_at": self.run_at.isoformat(),
            "win_rate": self.stats.win_rate,
            "win_rate_ci": list(self.stats.win_rate_ci),
            "n": self.stats.n,
            "expectancy_r": self.stats.expectancy_r,
            "total_r": self.stats.total_r,
            "mode": self.gate.mode.value,
            "gate_passed": self.gate.passed,
            "observations": [str(o) for o in self.observations],
            "proposals": [
                {"param": p.param, "current": p.current, "proposed": p.proposed,
                 "improvement_r": p.improvement, "rationale": p.rationale}
                for p in self.proposals
            ],
        }


class Reviewer:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.gate = PerformanceGate(cfg)

    # ------------------------------------------------------------ main entry

    def review(
        self,
        journal: Journal,
        *,
        entry_bars: list[Bar] | None = None,
        zone_bars: list[Bar] | None = None,
        bias_bars: list[Bar] | None = None,
        lookback_days: int = 90,
    ) -> ReviewReport:
        since = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        trades = journal.trades(since=since)
        stats = compute_stats(trades)
        gate = self.gate.evaluate(trades)

        report = ReviewReport(
            run_at=datetime.now(tz=timezone.utc),
            stats=stats,
            gate=gate,
            rejection_summary=journal.rejection_summary(since=since),
        )
        report.observations = self._observe(trades, stats)
        report.discipline = self._discipline_check(trades)

        if entry_bars and zone_bars and bias_bars and stats.n >= MIN_SAMPLE:
            report.proposals = self._validate_proposals(entry_bars, zone_bars, bias_bars)
        elif stats.n < MIN_SAMPLE:
            report.observations.append(
                Observation(
                    "tuning", f"holding all parameters -- {stats.n} trades is below "
                    f"the {MIN_SAMPLE}-trade minimum for any change", stats.n
                )
            )

        journal.save_review(report.to_dict())
        return report

    # ----------------------------------------------------------- observations

    def _observe(self, trades: list[JournalEntry], stats: Stats) -> list[Observation]:
        out: list[Observation] = []
        if stats.n == 0:
            return [Observation("sample", "no closed trades in the window", 0)]

        # Sessions: only call one out with a real sample behind it.
        for sess, v in stats.by_session.items():
            n = int(v["n"])
            if n < 10:
                out.append(Observation(
                    f"session:{sess}",
                    f"{n} trades, too few to judge (win {v['win_rate'] * 100:.0f}%)", n))
                continue
            if v["expectancy_r"] < -0.05:
                out.append(Observation(
                    f"session:{sess}",
                    f"losing money here: expectancy {v['expectancy_r']:+.3f}R over "
                    f"{n} trades, win rate {v['win_rate'] * 100:.0f}%",
                    n, actionable=True))
            elif v["expectancy_r"] > 0.25:
                out.append(Observation(
                    f"session:{sess}",
                    f"strongest session: expectancy {v['expectancy_r']:+.3f}R, "
                    f"win rate {v['win_rate'] * 100:.0f}%", n))

        # Does the confluence score actually predict anything? If high scores do
        # not outperform low scores, the scoring model is decorative.
        buckets = [(k, v) for k, v in stats.by_score_bucket.items() if int(v["n"]) >= 8]
        if len(buckets) >= 2:
            lo_k, lo_v = buckets[0]
            hi_k, hi_v = buckets[-1]
            if hi_v["expectancy_r"] <= lo_v["expectancy_r"]:
                out.append(Observation(
                    "score_validity",
                    f"higher confluence is NOT producing better results "
                    f"({hi_k}: {hi_v['expectancy_r']:+.3f}R vs {lo_k}: "
                    f"{lo_v['expectancy_r']:+.3f}R) -- the scoring weights need a "
                    f"human review, not an auto-tune",
                    int(hi_v["n"]) + int(lo_v["n"]), actionable=True))
            else:
                out.append(Observation(
                    "score_validity",
                    f"confluence score is predictive ({hi_k}: "
                    f"{hi_v['expectancy_r']:+.3f}R vs {lo_k}: "
                    f"{lo_v['expectancy_r']:+.3f}R)",
                    int(hi_v["n"]) + int(lo_v["n"])))

        # Stop placement: are we being wicked out just before the move?
        if stats.avg_mae_r > 0.75:
            out.append(Observation(
                "stops",
                f"average adverse excursion {stats.avg_mae_r:.2f}R -- winners are "
                f"coming within a whisker of the stop, so the stop is too tight "
                f"for current volatility", stats.n, actionable=True))

        # Targets: are we leaving money behind, or reaching too far?
        if stats.avg_mfe_r > self.cfg.strategy.tp2_r * 1.35:
            out.append(Observation(
                "targets",
                f"average favourable excursion {stats.avg_mfe_r:.2f}R exceeds the "
                f"{self.cfg.strategy.tp2_r:.2f}R target -- money is being left on "
                f"the table", stats.n, actionable=True))
        elif stats.n >= 20 and stats.avg_mfe_r < self.cfg.strategy.tp2_r * 0.7:
            out.append(Observation(
                "targets",
                f"price rarely reaches TP2 (avg MFE {stats.avg_mfe_r:.2f}R vs target "
                f"{self.cfg.strategy.tp2_r:.2f}R) -- the target is optimistic",
                stats.n, actionable=True))

        return out

    # ------------------------------------------------------------- discipline

    def _discipline_check(self, trades: list[JournalEntry]) -> list[str]:
        """Audit the human, not the strategy.

        The system can only enforce its rules on the signals it generates. If the
        journal shows more trades in a day than the limit allows, or trades
        clustered seconds after a loss, that is a human overriding the plan --
        and it is the single most likely cause of failure.
        """
        out: list[str] = []
        lim = self.cfg.risk

        by_day: dict[str, list[JournalEntry]] = {}
        for t in trades:
            by_day.setdefault(t.signal_time.date().isoformat(), []).append(t)

        over = {d: len(v) for d, v in by_day.items() if len(v) > lim.max_trades_per_day}
        if over:
            worst = max(over.items(), key=lambda kv: kv[1])
            out.append(
                f"OVERTRADING: {len(over)} day(s) exceeded the "
                f"{lim.max_trades_per_day}-trade limit (worst: {worst[0]} with "
                f"{worst[1]}). These were not system signals."
            )

        revenge = 0
        for day_trades in by_day.values():
            ordered = sorted(day_trades, key=lambda t: t.signal_time)
            for a, b in zip(ordered, ordered[1:]):
                if a.outcome == "loss" and a.exit_time:
                    gap = (b.signal_time - a.exit_time).total_seconds() / 60
                    if 0 <= gap < lim.cooloff_minutes_after_loss:
                        revenge += 1
        if revenge:
            out.append(
                f"REVENGE TRADING: {revenge} entry/entries taken inside the "
                f"{lim.cooloff_minutes_after_loss}-minute cool-off after a loss."
            )

        breached = [
            d for d, v in by_day.items()
            if sum(t.pnl for t in v) < -abs(lim.daily_loss_limit) * 1.15
        ]
        if breached:
            out.append(
                f"DAILY LOSS CAP BREACHED on {len(breached)} day(s) "
                f"({', '.join(breached[:3])}) -- trading continued past the stop point."
            )

        if not out:
            out.append(
                "Clean: no overtrading, no revenge entries, no loss-cap breaches."
            )
        return out

    # -------------------------------------------------------------- proposals

    def _validate_proposals(
        self, entry_bars: list[Bar], zone_bars: list[Bar], bias_bars: list[Bar]
    ) -> list[Proposal]:
        """Grid-search a few tunables, keeping only walk-forward survivors."""
        from .backtest import Backtester  # local import avoids a cycle

        split = int(len(entry_bars) * 0.65)
        train_e, test_e = entry_bars[:split], entry_bars[split:]
        if len(train_e) < 500 or len(test_e) < 300:
            return []

        def run(bars: list[Bar], params: StrategyParams) -> Stats:
            cfg = AgentConfig(risk=self.cfg.risk, strategy=params)
            bt = Backtester(cfg, mode=Mode.BACKTEST)
            t0, t1 = bars[0].t, bars[-1].t
            z = [b for b in zone_bars if t0 <= b.t <= t1]
            bi = [b for b in bias_bars if b.t <= t1]
            if len(z) < 100 or len(bi) < 250:
                return Stats()
            return bt.run(bars, z, bi).stats

        base = self.cfg.strategy
        base_test = run(test_e, base)

        candidates: list[tuple[str, Any]] = [
            ("min_confluence_score", base.min_confluence_score + 5),
            ("min_confluence_score", max(50, base.min_confluence_score - 5)),
            ("sl_atr_buffer", round(base.sl_atr_buffer + 0.2, 2)),
            ("sl_atr_buffer", round(max(0.2, base.sl_atr_buffer - 0.15), 2)),
            ("tp2_r", round(base.tp2_r + 0.4, 2)),
            ("tp2_r", round(max(base.tp1_r + 0.3, base.tp2_r - 0.4), 2)),
            ("ob_min_volume_ratio", round(base.ob_min_volume_ratio + 0.2, 2)),
            ("ha_confirm_bars", base.ha_confirm_bars + 1),
        ]

        out: list[Proposal] = []
        for name, value in candidates:
            try:
                trial = base.with_overrides(**{name: value})
            except ValueError:
                continue

            train_stats = run(train_e, trial)
            if train_stats.n < 15:
                continue
            base_train = run(train_e, base)
            if train_stats.expectancy_r <= base_train.expectancy_r + MIN_EDGE_IMPROVEMENT:
                continue  # no improvement even in-sample; discard immediately

            test_stats = run(test_e, trial)
            if test_stats.n < 10:
                continue
            gain = test_stats.expectancy_r - base_test.expectancy_r
            if gain < MIN_EDGE_IMPROVEMENT:
                continue  # in-sample only -- this is the curve fit we are filtering out

            out.append(Proposal(
                param=name,
                current=getattr(base, name),
                proposed=value,
                train_expectancy=train_stats.expectancy_r,
                test_expectancy=test_stats.expectancy_r,
                baseline_test_expectancy=base_test.expectancy_r,
                rationale=(
                    f"improved expectancy on both the fitted window "
                    f"({base_train.expectancy_r:+.3f} -> "
                    f"{train_stats.expectancy_r:+.3f}R, n={train_stats.n}) and the "
                    f"unseen window (n={test_stats.n}); survived walk-forward"
                ),
            ))

        out.sort(key=lambda p: -p.improvement)
        return out[:3]  # never propose a wholesale rewrite in one night
