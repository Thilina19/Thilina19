"""Tests for the risk guardrails.

These are the most important tests in the project. A bug in the strategy costs
a trade; a bug here costs the account.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

from xau_agent.config import OZ_PER_LOT, RiskLimits
from xau_agent.risk import BlockReason, ClosedTrade, RiskManager, SessionState


def dt(h: int = 12, m: int = 0, d: int = 15) -> datetime:
    return datetime(2024, 5, d, h, m, tzinfo=timezone.utc)


def fresh_state() -> SessionState:
    return SessionState(trading_day=date(2024, 5, 15))


def loss(amount: float, at: datetime) -> ClosedTrade:
    return ClosedTrade(at - timedelta(minutes=30), at, "long", 2000.0, 1990.0,
                       -abs(amount), -1.0)


def win(amount: float, at: datetime) -> ClosedTrade:
    return ClosedTrade(at - timedelta(minutes=30), at, "long", 2000.0, 2020.0,
                       abs(amount), 2.0)


class TestPositionSizing(unittest.TestCase):
    def setUp(self) -> None:
        self.rm = RiskManager(RiskLimits())

    def test_risk_budget_is_respected(self) -> None:
        """A stop-out must lose no more than the configured risk."""
        size = self.rm.size_position(entry=2000.0, stop=1990.0)
        assert size is not None
        # 250 USD risk / 10 USD stop = 25 oz = 0.25 lots
        self.assertAlmostEqual(size.lots, 0.25, places=2)
        self.assertLessEqual(size.risk_dollars, 250.0 + 0.01)

    def test_wider_stop_means_smaller_size(self) -> None:
        tight = self.rm.size_position(2000.0, 1995.0)
        wide = self.rm.size_position(2000.0, 1980.0)
        assert tight and wide
        self.assertGreater(tight.lots, wide.lots)
        # Both target the same risk budget. They cannot match exactly because
        # lots are quantized to 0.01 (= 1 oz), so the achievable risk moves in
        # steps of one ounce times the stop distance. Each must land within one
        # such step BELOW the budget -- never above it.
        budget = self.rm.limits.risk_dollars()
        for size in (tight, wide):
            step_value = 1.0 * size.stop_distance  # 1 oz of exposure
            self.assertLessEqual(size.risk_dollars, budget + 0.01)
            self.assertGreater(size.risk_dollars, budget - step_value)

    def test_rounding_never_increases_risk(self) -> None:
        """Lot rounding must go down, never up, or the budget is breached."""
        for stop_dist in [3.3, 7.7, 11.1, 13.9, 23.7, 41.3]:
            size = self.rm.size_position(2000.0, 2000.0 - stop_dist)
            if size is None:
                continue
            self.assertLessEqual(
                size.risk_dollars, self.rm.limits.risk_dollars() + 0.01,
                f"stop distance {stop_dist} breached the risk budget",
            )

    def test_zero_stop_distance_rejected(self) -> None:
        self.assertIsNone(self.rm.size_position(2000.0, 2000.0))

    def test_enormous_stop_rejected_rather_than_undersized(self) -> None:
        """If the stop is so wide that size rounds below the minimum lot, there
        is no trade -- we do not widen risk to make one fit."""
        self.assertIsNone(self.rm.size_position(2000.0, 200.0))

    def test_ounces_match_lots(self) -> None:
        size = self.rm.size_position(2000.0, 1990.0)
        assert size is not None
        self.assertAlmostEqual(size.ounces, size.lots * OZ_PER_LOT, places=6)
        self.assertAlmostEqual(size.dollars_per_point, size.ounces, places=6)


class TestDailyGuardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.rm = RiskManager(RiskLimits())
        self.state = fresh_state()

    def test_clean_state_allows_trade(self) -> None:
        d = self.rm.check_can_trade(self.state, now=dt(), entry=2000.0,
                                    stop=1990.0, target=2025.0)
        self.assertTrue(d.allowed, d.summary)

    def test_daily_loss_limit_blocks(self) -> None:
        self.state.realised_pnl = -500.0
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertFalse(d.allowed)
        self.assertIn(BlockReason.DAILY_LOSS_LIMIT, d.reasons)

    def test_daily_target_stops_trading(self) -> None:
        """Hitting +500 ends the day. Giving profit back is a real failure mode."""
        self.state.realised_pnl = 505.0
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertFalse(d.allowed)
        self.assertIn(BlockReason.DAILY_TARGET_MET, d.reasons)

    def test_max_trades_blocks(self) -> None:
        self.state.trades_today = 3
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertIn(BlockReason.MAX_TRADES, d.reasons)

    def test_two_consecutive_losses_ends_session(self) -> None:
        t = dt(9)
        self.state.register_close(loss(250, t))
        self.state.register_close(loss(250, t + timedelta(hours=1)))
        # Far past cool-off, so this isolates the consecutive-loss rule.
        d = self.rm.check_can_trade(self.state, now=t + timedelta(hours=5))
        self.assertFalse(d.allowed)
        self.assertIn(BlockReason.CONSECUTIVE_LOSSES, d.reasons)

    def test_win_resets_consecutive_losses(self) -> None:
        t = dt(9)
        self.state.register_close(loss(250, t))
        self.state.register_close(win(300, t + timedelta(hours=1)))
        self.assertEqual(self.state.consecutive_losses, 0)

    def test_cooloff_blocks_immediate_reentry(self) -> None:
        """The anti-revenge rule: no re-entry for 30 minutes after a loss."""
        t = dt(9)
        self.state.register_close(loss(250, t))
        d = self.rm.check_can_trade(self.state, now=t + timedelta(minutes=5))
        self.assertIn(BlockReason.COOLOFF, d.reasons)

    def test_cooloff_expires(self) -> None:
        t = dt(9)
        self.state.register_close(loss(250, t))
        d = self.rm.check_can_trade(self.state, now=t + timedelta(minutes=31),
                                    entry=2000.0, stop=1990.0, target=2025.0)
        self.assertNotIn(BlockReason.COOLOFF, d.reasons)
        self.assertTrue(d.allowed, d.summary)

    def test_open_position_blocks_second_entry(self) -> None:
        self.state.register_open()
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertIn(BlockReason.POSITION_OPEN, d.reasons)

    def test_weekly_drawdown_blocks(self) -> None:
        self.state.week_pnl = -1600.0  # beyond 3% of 50k
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertIn(BlockReason.WEEKLY_DRAWDOWN, d.reasons)

    def test_monthly_drawdown_blocks(self) -> None:
        self.state.month_pnl = -3100.0  # beyond 6% of 50k
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertIn(BlockReason.MONTHLY_DRAWDOWN, d.reasons)

    def test_all_reasons_reported_not_just_first(self) -> None:
        self.state.realised_pnl = -600.0
        self.state.trades_today = 3
        self.state.register_open()
        d = self.rm.check_can_trade(self.state, now=dt())
        self.assertGreaterEqual(len(d.reasons), 3)

    def test_poor_reward_risk_blocked(self) -> None:
        # 10 risk for 10 reward = 1.0 R:R, below the 1.8 minimum.
        d = self.rm.check_can_trade(self.state, now=dt(), entry=2000.0,
                                    stop=1990.0, target=2010.0)
        self.assertIn(BlockReason.REWARD_RISK_TOO_LOW, d.reasons)


class TestSessionRollover(unittest.TestCase):
    def test_new_day_resets_daily_but_keeps_week(self) -> None:
        s = SessionState(trading_day=date(2024, 5, 15))
        s.realised_pnl = -400.0
        s.week_pnl = -400.0
        s.trades_today = 2
        s.consecutive_losses = 1
        s.roll_to_day(date(2024, 5, 16))  # same ISO week
        self.assertEqual(s.realised_pnl, 0.0)
        self.assertEqual(s.trades_today, 0)
        self.assertEqual(s.consecutive_losses, 0)
        self.assertEqual(s.week_pnl, -400.0)

    def test_new_week_resets_week_pnl(self) -> None:
        s = SessionState(trading_day=date(2024, 5, 17))  # Friday
        s.week_pnl = -900.0
        s.month_pnl = -900.0
        s.roll_to_day(date(2024, 5, 20))  # following Monday
        self.assertEqual(s.week_pnl, 0.0)
        self.assertEqual(s.month_pnl, -900.0)  # same month, carried

    def test_new_month_resets_month_pnl(self) -> None:
        s = SessionState(trading_day=date(2024, 5, 31))
        s.month_pnl = -1200.0
        s.roll_to_day(date(2024, 6, 3))
        self.assertEqual(s.month_pnl, 0.0)


class TestLimitValidation(unittest.TestCase):
    def test_reckless_risk_refused(self) -> None:
        with self.assertRaises(ValueError):
            RiskManager(RiskLimits(risk_per_trade_pct=0.05))

    def test_absurd_max_risk_refused(self) -> None:
        with self.assertRaises(ValueError):
            RiskManager(RiskLimits(max_risk_per_trade_pct=0.10,
                                   risk_per_trade_pct=0.05))

    def test_sub_one_reward_risk_refused(self) -> None:
        with self.assertRaises(ValueError):
            RiskManager(RiskLimits(min_reward_risk=0.5))

    def test_risk_capped_by_max(self) -> None:
        """Even if risk_per_trade_pct is raised, max_risk_per_trade_pct wins."""
        lim = RiskLimits(risk_per_trade_pct=0.01, max_risk_per_trade_pct=0.01)
        self.assertAlmostEqual(lim.risk_dollars(), 500.0)

    def test_multiple_positions_refused(self) -> None:
        with self.assertRaises(ValueError):
            RiskManager(RiskLimits(max_concurrent_positions=3))


class TestBudgetHelpers(unittest.TestCase):
    def test_remaining_budget_shrinks_with_losses(self) -> None:
        rm = RiskManager(RiskLimits())
        s = fresh_state()
        self.assertAlmostEqual(rm.remaining_risk_budget(s), 500.0)
        s.realised_pnl = -200.0
        self.assertAlmostEqual(rm.remaining_risk_budget(s), 300.0)
        s.realised_pnl = -700.0
        self.assertAlmostEqual(rm.remaining_risk_budget(s), 0.0)

    def test_profit_does_not_expand_loss_budget(self) -> None:
        """Being up money must not license a larger loss -- that is how a good
        day turns into a bad one."""
        rm = RiskManager(RiskLimits())
        s = fresh_state()
        s.realised_pnl = 400.0
        self.assertAlmostEqual(rm.remaining_risk_budget(s), 500.0)

    def test_reward_risk_math(self) -> None:
        rm = RiskManager(RiskLimits())
        self.assertAlmostEqual(rm.reward_risk(2000, 1990, 2020), 2.0)
        self.assertAlmostEqual(rm.reward_risk(2000, 2010, 1980), 2.0)
        self.assertEqual(rm.reward_risk(2000, 2000, 2020), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
