"""Risk management and trading guardrails.

This module is the reason the rest of the system is allowed to be clever. It
has one job: make it structurally impossible to overtrade, revenge trade, or
size a position larger than the plan allows.

Design rules:
  - Nothing here consults the strategy. A perfect setup and a terrible setup
    are sized identically and blocked identically.
  - `check_can_trade` returns every reason it said no, not just the first, so
    the journal records the full picture.
  - The learning loop cannot reach these limits (see config.RiskLimits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from .config import OZ_PER_LOT, RiskLimits


class BlockReason(str, Enum):
    DAILY_LOSS_LIMIT = "daily_loss_limit_hit"
    DAILY_TARGET_MET = "daily_profit_target_met"
    MAX_TRADES = "max_trades_per_day_reached"
    CONSECUTIVE_LOSSES = "consecutive_loss_shutdown"
    COOLOFF = "cooloff_after_loss_active"
    POSITION_OPEN = "position_already_open"
    WEEKLY_DRAWDOWN = "weekly_drawdown_limit"
    MONTHLY_DRAWDOWN = "monthly_drawdown_limit"
    REWARD_RISK_TOO_LOW = "reward_to_risk_below_minimum"
    INVALID_STOP = "invalid_stop_distance"
    SIZE_ROUNDS_TO_ZERO = "position_size_rounds_to_zero"


@dataclass(frozen=True)
class PositionSize:
    lots: float
    ounces: float
    risk_dollars: float
    stop_distance: float

    @property
    def dollars_per_point(self) -> float:
        """USD gained/lost per 1.00 move in the gold price."""
        return self.ounces


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reasons: list[BlockReason] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.allowed:
            return "allowed"
        return "blocked: " + ", ".join(r.value for r in self.reasons)


@dataclass
class ClosedTrade:
    """A finished trade, as the risk manager needs to see it."""

    opened_at: datetime
    closed_at: datetime
    side: str
    entry: float
    exit: float
    pnl: float
    r_multiple: float

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class SessionState:
    """Mutable state the risk manager tracks across a trading day.

    Persisted by journal.py so a restart does not reset the daily loss cap --
    that reset would be the single most dangerous bug this system could have.
    """

    trading_day: date
    realised_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    last_loss_at: datetime | None = None
    open_positions: int = 0
    week_pnl: float = 0.0
    month_pnl: float = 0.0

    def register_close(self, trade: ClosedTrade) -> None:
        self.realised_pnl += trade.pnl
        self.week_pnl += trade.pnl
        self.month_pnl += trade.pnl
        self.trades_today += 1
        self.open_positions = max(0, self.open_positions - 1)
        if trade.is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.last_loss_at = trade.closed_at

    def register_open(self) -> None:
        self.open_positions += 1

    def roll_to_day(self, day: date) -> None:
        """Start a new trading day, carrying week/month totals forward."""
        if day == self.trading_day:
            return
        if day.isocalendar()[:2] != self.trading_day.isocalendar()[:2]:
            self.week_pnl = 0.0
        if (day.year, day.month) != (self.trading_day.year, self.trading_day.month):
            self.month_pnl = 0.0
        self.trading_day = day
        self.realised_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_loss_at = None


class RiskManager:
    """Enforces RiskLimits. Stateless with respect to strategy opinion."""

    def __init__(self, limits: RiskLimits, *, lot_step: float = 0.01, min_lot: float = 0.01):
        limits.validate()
        self.limits = limits
        self.lot_step = lot_step
        self.min_lot = min_lot

    # ------------------------------------------------------------------ sizing

    def size_position(self, entry: float, stop: float) -> PositionSize | None:
        """Lots such that a stop-out loses exactly the per-trade risk budget.

        Returns None when the stop distance is unusable or the resulting size
        rounds below the broker's minimum -- in both cases there is no valid
        trade, and forcing one by widening risk is how accounts die.
        """
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return None

        risk = self.limits.risk_dollars()
        ounces = risk / stop_distance
        lots = ounces / OZ_PER_LOT

        # Round DOWN to the broker's lot step. Rounding up would breach the
        # risk budget, which is never acceptable.
        steps = int(lots / self.lot_step)
        lots = steps * self.lot_step
        if lots < self.min_lot:
            return None

        ounces = lots * OZ_PER_LOT
        return PositionSize(
            lots=round(lots, 2),
            ounces=ounces,
            risk_dollars=ounces * stop_distance,
            stop_distance=stop_distance,
        )

    # ---------------------------------------------------------------- gating

    def check_can_trade(
        self,
        state: SessionState,
        *,
        now: datetime,
        entry: float | None = None,
        stop: float | None = None,
        target: float | None = None,
    ) -> RiskDecision:
        """Collect every reason this trade should not be taken."""
        reasons: list[BlockReason] = []
        notes: list[str] = []
        lim = self.limits

        if state.open_positions >= lim.max_concurrent_positions:
            reasons.append(BlockReason.POSITION_OPEN)

        if state.realised_pnl <= -abs(lim.daily_loss_limit):
            reasons.append(BlockReason.DAILY_LOSS_LIMIT)
            notes.append(
                f"day P&L {state.realised_pnl:+.2f} at or beyond "
                f"-{lim.daily_loss_limit:.2f}; done until tomorrow"
            )

        if state.realised_pnl >= lim.daily_profit_target:
            reasons.append(BlockReason.DAILY_TARGET_MET)
            notes.append(
                f"target met ({state.realised_pnl:+.2f}); protecting the day"
            )

        if state.trades_today >= lim.max_trades_per_day:
            reasons.append(BlockReason.MAX_TRADES)

        if state.consecutive_losses >= lim.max_consecutive_losses:
            reasons.append(BlockReason.CONSECUTIVE_LOSSES)
            notes.append(
                f"{state.consecutive_losses} losses in a row; the edge is not "
                f"present today"
            )

        if state.last_loss_at is not None:
            elapsed = now - state.last_loss_at
            if elapsed < timedelta(minutes=lim.cooloff_minutes_after_loss):
                left = timedelta(minutes=lim.cooloff_minutes_after_loss) - elapsed
                reasons.append(BlockReason.COOLOFF)
                notes.append(f"cool-off active, {int(left.total_seconds() // 60)}m left")

        if state.week_pnl <= -abs(lim.account_equity * lim.weekly_loss_limit_pct):
            reasons.append(BlockReason.WEEKLY_DRAWDOWN)
        if state.month_pnl <= -abs(lim.account_equity * lim.monthly_loss_limit_pct):
            reasons.append(BlockReason.MONTHLY_DRAWDOWN)

        # Trade-specific geometry checks.
        if entry is not None and stop is not None:
            if abs(entry - stop) <= 0:
                reasons.append(BlockReason.INVALID_STOP)
            elif self.size_position(entry, stop) is None:
                reasons.append(BlockReason.SIZE_ROUNDS_TO_ZERO)

            if target is not None:
                rr = self.reward_risk(entry, stop, target)
                if rr < lim.min_reward_risk:
                    reasons.append(BlockReason.REWARD_RISK_TOO_LOW)
                    notes.append(
                        f"R:R {rr:.2f} below minimum {lim.min_reward_risk:.2f}"
                    )

        return RiskDecision(allowed=not reasons, reasons=reasons, notes=notes)

    # ------------------------------------------------------------- utilities

    @staticmethod
    def reward_risk(entry: float, stop: float, target: float) -> float:
        risk = abs(entry - stop)
        if risk <= 0:
            return 0.0
        return abs(target - entry) / risk

    def remaining_risk_budget(self, state: SessionState) -> float:
        """Dollars of loss still permitted today before the cap trips."""
        return max(0.0, self.limits.daily_loss_limit + min(0.0, state.realised_pnl))

    def trades_remaining(self, state: SessionState) -> int:
        return max(0, self.limits.max_trades_per_day - state.trades_today)

    def progress_note(self, state: SessionState) -> str:
        lim = self.limits
        return (
            f"day P&L {state.realised_pnl:+.2f} / target {lim.daily_profit_target:.0f} "
            f"| trades {state.trades_today}/{lim.max_trades_per_day} "
            f"| loss budget left {self.remaining_risk_budget(state):.2f}"
        )


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)
