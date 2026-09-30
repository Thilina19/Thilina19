"""Trade journal, performance statistics, and the live-promotion gate.

The gate is the most important thing in this file. You asked for a 60% win rate
before the agent marks up live entries. The honest way to hold that promise is
to refuse to believe a win rate until the sample supports it.

A 6-from-10 run is a 60% win rate and means nothing: the 95% confidence interval
on that is roughly 26%-88%. So the gate tests the *lower bound* of the Wilson
score interval, not the point estimate. In practice that means roughly 40+ trades
before the system will promote itself out of paper mode, and it will demote
itself again if performance decays.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .config import AgentConfig
from .risk import ClosedTrade, SessionState


class Mode(str, Enum):
    """The agent's operating mode. Only LIVE publishes actionable entries."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Outcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"


@dataclass
class JournalEntry:
    """One recorded trade. `r_multiple` is the unit that matters -- dollars
    change with account size, R does not."""

    signal_time: datetime
    side: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    score: int
    session: str
    mode: str
    lots: float = 0.0
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    outcome: str = Outcome.OPEN.value
    mae_r: float = 0.0   # worst adverse excursion, in R -- how close to stopped
    mfe_r: float = 0.0   # best favourable excursion, in R -- did we leave money
    notes: str = ""
    params_hash: str = ""
    id: int | None = None

    def to_row(self) -> tuple:
        return (
            self.signal_time.isoformat(), self.side, self.entry, self.stop,
            self.tp1, self.tp2, self.score, self.session, self.mode, self.lots,
            self.exit_price,
            self.exit_time.isoformat() if self.exit_time else None,
            self.pnl, self.r_multiple, self.outcome, self.mae_r, self.mfe_r,
            self.notes, self.params_hash,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_time  TEXT NOT NULL,
    side         TEXT NOT NULL,
    entry        REAL NOT NULL,
    stop         REAL NOT NULL,
    tp1          REAL NOT NULL,
    tp2          REAL NOT NULL,
    score        INTEGER NOT NULL,
    session      TEXT NOT NULL,
    mode         TEXT NOT NULL,
    lots         REAL NOT NULL DEFAULT 0,
    exit_price   REAL,
    exit_time    TEXT,
    pnl          REAL NOT NULL DEFAULT 0,
    r_multiple   REAL NOT NULL DEFAULT 0,
    outcome      TEXT NOT NULL DEFAULT 'open',
    mae_r        REAL NOT NULL DEFAULT 0,
    mfe_r        REAL NOT NULL DEFAULT 0,
    notes        TEXT DEFAULT '',
    params_hash  TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(signal_time);
CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode);

-- Rejections are logged too. Knowing what we declined is how we tell
-- "the filter is working" from "the filter is broken".
CREATE TABLE IF NOT EXISTS rejections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    t           TEXT NOT NULL,
    score       INTEGER NOT NULL,
    threshold   INTEGER NOT NULL,
    reason      TEXT NOT NULL
);

-- Session state must survive a restart, or the daily loss cap resets on crash.
CREATE TABLE IF NOT EXISTS session_state (
    trading_day         TEXT PRIMARY KEY,
    realised_pnl        REAL NOT NULL,
    trades_today        INTEGER NOT NULL,
    consecutive_losses  INTEGER NOT NULL,
    last_loss_at        TEXT,
    open_positions      INTEGER NOT NULL,
    week_pnl            REAL NOT NULL,
    month_pnl           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at    TEXT NOT NULL,
    payload   TEXT NOT NULL
);
"""


class Journal:
    """SQLite-backed trade journal."""

    def __init__(self, path: str | Path = "data/journal.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- writes

    def record(self, entry: JournalEntry) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (signal_time, side, entry, stop, tp1, tp2, score,
                   session, mode, lots, exit_price, exit_time, pnl, r_multiple,
                   outcome, mae_r, mfe_r, notes, params_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            entry.to_row(),
        )
        self.conn.commit()
        entry.id = cur.lastrowid
        return cur.lastrowid

    def record_many(self, entries: Iterable[JournalEntry]) -> int:
        rows = [e.to_row() for e in entries]
        self.conn.executemany(
            """INSERT INTO trades (signal_time, side, entry, stop, tp1, tp2, score,
                   session, mode, lots, exit_price, exit_time, pnl, r_multiple,
                   outcome, mae_r, mfe_r, notes, params_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def record_rejection(self, t: datetime, score: int, threshold: int, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO rejections (t, score, threshold, reason) VALUES (?,?,?,?)",
            (t.isoformat(), score, threshold, reason),
        )
        self.conn.commit()

    def save_state(self, s: SessionState) -> None:
        self.conn.execute(
            """INSERT INTO session_state
               (trading_day, realised_pnl, trades_today, consecutive_losses,
                last_loss_at, open_positions, week_pnl, month_pnl)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(trading_day) DO UPDATE SET
                 realised_pnl=excluded.realised_pnl,
                 trades_today=excluded.trades_today,
                 consecutive_losses=excluded.consecutive_losses,
                 last_loss_at=excluded.last_loss_at,
                 open_positions=excluded.open_positions,
                 week_pnl=excluded.week_pnl,
                 month_pnl=excluded.month_pnl""",
            (
                s.trading_day.isoformat(), s.realised_pnl, s.trades_today,
                s.consecutive_losses,
                s.last_loss_at.isoformat() if s.last_loss_at else None,
                s.open_positions, s.week_pnl, s.month_pnl,
            ),
        )
        self.conn.commit()

    def load_state(self, day: date) -> SessionState:
        """Load the day's state, or carry week/month totals into a fresh day."""
        row = self.conn.execute(
            "SELECT * FROM session_state WHERE trading_day = ?", (day.isoformat(),)
        ).fetchone()
        if row:
            return SessionState(
                trading_day=date.fromisoformat(row["trading_day"]),
                realised_pnl=row["realised_pnl"],
                trades_today=row["trades_today"],
                consecutive_losses=row["consecutive_losses"],
                last_loss_at=(
                    datetime.fromisoformat(row["last_loss_at"])
                    if row["last_loss_at"] else None
                ),
                open_positions=row["open_positions"],
                week_pnl=row["week_pnl"],
                month_pnl=row["month_pnl"],
            )

        prev = self.conn.execute(
            "SELECT * FROM session_state WHERE trading_day < ? "
            "ORDER BY trading_day DESC LIMIT 1", (day.isoformat(),)
        ).fetchone()
        state = SessionState(trading_day=day)
        if prev:
            pd = date.fromisoformat(prev["trading_day"])
            if pd.isocalendar()[:2] == day.isocalendar()[:2]:
                state.week_pnl = prev["week_pnl"]
            if (pd.year, pd.month) == (day.year, day.month):
                state.month_pnl = prev["month_pnl"]
        return state

    def save_review(self, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO reviews (run_at, payload) VALUES (?,?)",
            (datetime.now(tz=timezone.utc).isoformat(), json.dumps(payload, default=str)),
        )
        self.conn.commit()

    # ----------------------------------------------------------------- reads

    def trades(self, *, mode: str | None = None, since: datetime | None = None,
               closed_only: bool = True) -> list[JournalEntry]:
        q = "SELECT * FROM trades WHERE 1=1"
        args: list[Any] = []
        if mode:
            q += " AND mode = ?"
            args.append(mode)
        if since:
            q += " AND signal_time >= ?"
            args.append(since.isoformat())
        if closed_only:
            q += " AND outcome != 'open'"
        q += " ORDER BY signal_time ASC"
        return [self._to_entry(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _to_entry(r: sqlite3.Row) -> JournalEntry:
        return JournalEntry(
            id=r["id"],
            signal_time=datetime.fromisoformat(r["signal_time"]),
            side=r["side"], entry=r["entry"], stop=r["stop"],
            tp1=r["tp1"], tp2=r["tp2"], score=r["score"], session=r["session"],
            mode=r["mode"], lots=r["lots"], exit_price=r["exit_price"],
            exit_time=datetime.fromisoformat(r["exit_time"]) if r["exit_time"] else None,
            pnl=r["pnl"], r_multiple=r["r_multiple"], outcome=r["outcome"],
            mae_r=r["mae_r"], mfe_r=r["mfe_r"], notes=r["notes"] or "",
            params_hash=r["params_hash"] or "",
        )

    def rejection_summary(self, since: datetime | None = None) -> dict[str, int]:
        q = "SELECT reason, COUNT(*) n FROM rejections"
        args: list[Any] = []
        if since:
            q += " WHERE t >= ?"
            args.append(since.isoformat())
        q += " GROUP BY reason ORDER BY n DESC"
        return {r["reason"]: r["n"] for r in self.conn.execute(q, args)}


# --------------------------------------------------------------- statistics


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because it behaves correctly
    for small n and for proportions near 0 or 1, which is exactly the regime a
    new strategy lives in.
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class Stats:
    n: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    win_rate_ci: tuple[float, float] = (0.0, 0.0)
    total_r: float = 0.0
    expectancy_r: float = 0.0
    total_pnl: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_r: float = 0.0
    max_consecutive_losses: int = 0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    by_session: dict[str, dict[str, float]] = field(default_factory=dict)
    by_score_bucket: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lo, hi = self.win_rate_ci
        return (
            f"{self.n} trades | win rate {self.win_rate * 100:.1f}% "
            f"(95% CI {lo * 100:.1f}-{hi * 100:.1f}%) | "
            f"expectancy {self.expectancy_r:+.3f}R | total {self.total_r:+.2f}R | "
            f"PF {self.profit_factor:.2f} | maxDD {self.max_drawdown_r:.2f}R"
        )


def compute_stats(trades: list[JournalEntry]) -> Stats:
    closed = [t for t in trades if t.outcome != Outcome.OPEN.value]
    s = Stats(n=len(closed))
    if not closed:
        return s

    s.wins = sum(1 for t in closed if t.outcome == Outcome.WIN.value)
    s.losses = sum(1 for t in closed if t.outcome == Outcome.LOSS.value)
    s.breakeven = sum(1 for t in closed if t.outcome == Outcome.BREAKEVEN.value)

    # Breakeven trades are excluded from the win-rate denominator: they are
    # neither a win nor a loss and including them understates the edge.
    decided = s.wins + s.losses
    s.win_rate = s.wins / decided if decided else 0.0
    s.win_rate_ci = wilson_interval(s.wins, decided)

    rs = [t.r_multiple for t in closed]
    s.total_r = sum(rs)
    s.expectancy_r = s.total_r / len(rs)
    s.total_pnl = sum(t.pnl for t in closed)

    win_rs = [t.r_multiple for t in closed if t.r_multiple > 0]
    loss_rs = [t.r_multiple for t in closed if t.r_multiple < 0]
    s.avg_win_r = sum(win_rs) / len(win_rs) if win_rs else 0.0
    s.avg_loss_r = sum(loss_rs) / len(loss_rs) if loss_rs else 0.0
    gross_win, gross_loss = sum(win_rs), abs(sum(loss_rs))
    s.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown on the R-multiple equity curve.
    peak = cum = 0.0
    dd = 0.0
    run = worst_run = 0
    for t in closed:
        cum += t.r_multiple
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        if t.outcome == Outcome.LOSS.value:
            run += 1
            worst_run = max(worst_run, run)
        elif t.outcome == Outcome.WIN.value:
            run = 0
    s.max_drawdown_r = dd
    s.max_consecutive_losses = worst_run

    s.avg_mae_r = sum(t.mae_r for t in closed) / len(closed)
    s.avg_mfe_r = sum(t.mfe_r for t in closed) / len(closed)

    s.by_session = _group(closed, lambda t: t.session)
    s.by_score_bucket = _group(closed, lambda t: f"{(t.score // 10) * 10}-{(t.score // 10) * 10 + 9}")
    return s


def _group(trades: list[JournalEntry], key) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[JournalEntry]] = {}
    for t in trades:
        buckets.setdefault(key(t), []).append(t)
    out: dict[str, dict[str, float]] = {}
    for k, group in sorted(buckets.items()):
        w = sum(1 for t in group if t.outcome == Outcome.WIN.value)
        l = sum(1 for t in group if t.outcome == Outcome.LOSS.value)
        decided = w + l
        out[k] = {
            "n": len(group),
            "wins": w,
            "win_rate": (w / decided) if decided else 0.0,
            "total_r": sum(t.r_multiple for t in group),
            "expectancy_r": sum(t.r_multiple for t in group) / len(group),
        }
    return out


# ------------------------------------------------------------ promotion gate


@dataclass
class GateResult:
    mode: Mode
    passed: bool
    reasons: list[str] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)

    def report(self) -> str:
        head = (
            f"MODE: {self.mode.value.upper()}  "
            f"({'cleared for live entries' if self.passed else 'not cleared'})"
        )
        return "\n".join([head, f"  {self.stats.summary()}"] +
                         [f"  - {r}" for r in self.reasons])


class PerformanceGate:
    """Decides whether the agent may publish live, actionable entries.

    Every condition must hold. The agent starts in PAPER and is demoted back to
    PAPER automatically the moment any condition stops holding -- promotion is
    not a one-way door.
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def evaluate(self, trades: list[JournalEntry]) -> GateResult:
        cfg = self.cfg
        paper = [t for t in trades
                 if t.mode in (Mode.PAPER.value, Mode.LIVE.value)
                 and t.outcome != Outcome.OPEN.value]
        stats = compute_stats(paper)
        reasons: list[str] = []

        if stats.n < cfg.min_trades_for_live:
            reasons.append(
                f"need {cfg.min_trades_for_live} forward-tested trades, have "
                f"{stats.n} -- a win rate on a smaller sample is not measurable"
            )

        lo, _ = stats.win_rate_ci
        if lo < cfg.live_win_rate_floor:
            reasons.append(
                f"win-rate 95% CI lower bound {lo * 100:.1f}% is below the "
                f"{cfg.live_win_rate_floor * 100:.0f}% floor (point estimate "
                f"{stats.win_rate * 100:.1f}%) -- not yet distinguishable from luck"
            )

        if stats.expectancy_r < cfg.min_expectancy_r:
            reasons.append(
                f"expectancy {stats.expectancy_r:+.3f}R below required "
                f"{cfg.min_expectancy_r:+.2f}R"
            )

        if stats.max_drawdown_r > cfg.max_drawdown_r_for_live:
            reasons.append(
                f"peak drawdown {stats.max_drawdown_r:.1f}R exceeds the "
                f"{cfg.max_drawdown_r_for_live:.0f}R comfort limit for this "
                f"account size"
            )

        passed = not reasons
        if passed:
            reasons.append(
                f"win rate {stats.win_rate * 100:.1f}% (CI lower bound "
                f"{lo * 100:.1f}%) over {stats.n} trades, expectancy "
                f"{stats.expectancy_r:+.3f}R -- entries will be published with "
                f"SL and TP"
            )
            # Clearing the gate is not the same as having proven 60%. Say so.
            if stats.n < cfg.trades_to_prove_target or lo < cfg.target_win_rate:
                reasons.append(
                    f"note: the {cfg.target_win_rate * 100:.0f}% target is not yet "
                    f"statistically established -- that needs roughly "
                    f"{cfg.trades_to_prove_target} trades with the CI lower bound "
                    f"above {cfg.target_win_rate * 100:.0f}%. Currently "
                    f"{stats.n} trades, lower bound {lo * 100:.1f}%."
                )
        return GateResult(Mode.LIVE if passed else Mode.PAPER, passed, reasons, stats)
