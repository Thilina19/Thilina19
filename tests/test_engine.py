"""Tests for indicators, market structure, strategy and the backtester.

The lookahead tests matter most. A backtest that can see the future produces
beautiful numbers and loses real money.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from xau_agent.backtest import Backtester, align_index
from xau_agent.config import AgentConfig, RiskLimits, StrategyParams
from xau_agent.data import resample, synthetic_bars
from xau_agent.indicators import (
    Bar, atr, ema, find_swings, heikin_ashi, rsi, sma, volume_ratio,
)
from xau_agent.journal import (
    Journal, JournalEntry, Mode, Outcome, PerformanceGate, compute_stats,
    wilson_interval,
)
from xau_agent.strategy import MarketView, Rejection, Side, Strategy
from xau_agent.structure import (
    Bias, confirmed_swings, find_order_blocks, market_bias,
    nearest_unmitigated_block,
)


def mk(seq: list[tuple[float, float, float, float]], *, step: int = 900,
       vol: float = 1000.0) -> list[Bar]:
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    return [Bar(t0 + i * step, o, h, l, c, vol) for i, (o, h, l, c) in enumerate(seq)]


def _with_volume(b: Bar, v: float) -> Bar:
    """Copy of a bar with its volume replaced."""
    return Bar(b.t, b.o, b.h, b.l, b.c, v)


# A 4h EMA200 needs 250+ 4h bars, which is 250 * 16 = 4000 bars of 15m data
# (about 42 days). Anything less and every evaluation is skipped during warm-up.
# Tests that expect signals must use at least this much.
MIN_15M_BARS_FOR_SIGNALS = 4800


def dataset(n: int = MIN_15M_BARS_FOR_SIGNALS, seed: int = 11
            ) -> tuple[list[Bar], list[Bar], list[Bar]]:
    """A 15m series plus its consistent 1h and 4h resamples."""
    bars = synthetic_bars(n, interval="15m", seed=seed)
    return bars, resample(bars, "1h"), resample(bars, "4h")


class TestIndicators(unittest.TestCase):
    def test_sma_basic(self) -> None:
        out = sma([1, 2, 3, 4, 5], 3)
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_seeded_with_sma(self) -> None:
        vals = [float(i) for i in range(1, 21)]
        out = ema(vals, 5)
        self.assertIsNone(out[3])
        self.assertAlmostEqual(out[4], 3.0)  # SMA of 1..5
        self.assertGreater(out[-1], out[4])

    def test_ema_shorter_than_period_is_all_none(self) -> None:
        self.assertTrue(all(v is None for v in ema([1.0, 2.0], 10)))

    def test_atr_positive_and_warms_up(self) -> None:
        bars = synthetic_bars(80, interval="15m")
        a = atr(bars, 14)
        self.assertIsNone(a[0])
        self.assertIsNone(a[13])
        self.assertIsNotNone(a[14])
        self.assertTrue(all(v > 0 for v in a[14:]))

    def test_atr_on_constant_range_equals_range(self) -> None:
        """Ten identical bars: ATR should converge to the bar range."""
        bars = mk([(100.0, 102.0, 98.0, 100.0)] * 40)
        a = atr(bars, 14)
        self.assertAlmostEqual(a[-1], 4.0, places=6)

    def test_rsi_bounds(self) -> None:
        bars = synthetic_bars(120)
        for v in rsi(bars, 14):
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_rsi_all_up_is_100(self) -> None:
        bars = mk([(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(40)])
        self.assertAlmostEqual(rsi(bars, 14)[-1], 100.0, places=6)

    def test_volume_ratio_excludes_current_bar(self) -> None:
        bars = mk([(100, 101, 99, 100)] * 25, vol=100.0)
        bars[24] = Bar(bars[24].t, 100, 101, 99, 100, 500.0)
        vr = volume_ratio(bars, 20)
        # 500 against a 100 average of the PRIOR 20 bars.
        self.assertAlmostEqual(vr[24], 5.0, places=6)

    def test_heikin_ashi_length_and_smoothing(self) -> None:
        bars = synthetic_bars(50)
        ha = heikin_ashi(bars)
        self.assertEqual(len(ha), len(bars))
        # HA close is the OHLC average of the same bar.
        self.assertAlmostEqual(
            ha[10].c, (bars[10].o + bars[10].h + bars[10].l + bars[10].c) / 4
        )

    def test_heikin_ashi_wick_ratio_doji(self) -> None:
        ha = heikin_ashi(mk([(100.0, 105.0, 95.0, 100.0)] * 3))
        self.assertGreater(ha[-1].wick_ratio, 0.5)

    def test_swings_found_on_a_clear_peak(self) -> None:
        # The peak bar's high must strictly exceed its neighbours on both
        # sides, so no adjacent bar may also reach 110.
        seq = [(100, 101, 99, 100), (101, 102, 100, 101), (102, 110, 101, 109),
               (109, 109.5, 104, 105), (105, 106, 100, 101), (101, 102, 99, 100)]
        swings = find_swings(mk(seq), lookback=2)
        self.assertTrue(any(s.is_high and s.price == 110 for s in swings))

    def test_equal_highs_are_not_swings(self) -> None:
        """A double top is not a fractal peak -- strict inequality is intended."""
        seq = [(100, 101, 99, 100), (101, 102, 100, 101), (102, 110, 101, 109),
               (109, 110, 104, 105), (105, 106, 100, 101), (101, 102, 99, 100)]
        swings = find_swings(mk(seq), lookback=2)
        self.assertFalse(any(s.is_high and s.price == 110 for s in swings))


class TestNoLookahead(unittest.TestCase):
    """The backtest's credibility rests entirely on these."""

    def test_confirmed_swings_respect_the_confirmation_delay(self) -> None:
        bars = synthetic_bars(200)
        lookback = 2
        for upto in (50, 80, 120, 199):
            for s in confirmed_swings(bars, upto, lookback):
                self.assertLessEqual(
                    s.index + lookback, upto,
                    "a swing was reported before it could be confirmed",
                )

    def test_confirmed_swings_never_index_beyond_upto(self) -> None:
        bars = synthetic_bars(200)
        for s in confirmed_swings(bars, 100, 2):
            self.assertLessEqual(s.index, 100)

    def test_bias_is_stable_when_future_bars_are_appended(self) -> None:
        """Bias at bar 150 must not change because bars 151+ exist.

        This is the decisive lookahead test: same index, different amounts of
        future data, identical answer.
        """
        bars = synthetic_bars(400)
        early = market_bias(bars[:151], 150)
        late = market_bias(bars, 150)
        self.assertEqual(early.bias, late.bias)
        self.assertEqual(early.reason, late.reason)

    def test_order_blocks_stable_when_future_bars_appended(self) -> None:
        bars = synthetic_bars(500)
        a = find_order_blocks(bars[:301], 300)
        b = find_order_blocks(bars, 300)
        self.assertEqual([x.index for x in a], [x.index for x in b])
        self.assertEqual([x.top for x in a], [x.top for x in b])

    def test_align_index_excludes_the_still_open_bar(self) -> None:
        # 4h bars opening at 0, 4h, 8h...
        times = [0, 14400, 28800, 43200]
        # At exactly 28800 the 28800 bar has just opened, so the newest CLOSED
        # bar is the one at 14400.
        self.assertEqual(align_index(times, 28800), 1)
        # Mid-way through the 28800 bar, still the same answer.
        self.assertEqual(align_index(times, 30000), 2)
        self.assertEqual(align_index(times, -1), -1)

    def test_align_index_never_returns_future_bar(self) -> None:
        times = [i * 14400 for i in range(50)]
        for target in (0, 100, 14400, 20000, 43200, 700000):
            i = align_index(times, target)
            if i >= 0:
                self.assertLess(times[i], target + 1)


class TestStructure(unittest.TestCase):
    def test_uptrend_detected(self) -> None:
        seq: list[tuple[float, float, float, float]] = []
        price = 100.0
        for leg in range(12):  # rising zig-zag: HH and HL
            for k in range(4):
                price += 2.0
                seq.append((price - 1, price + 1.5, price - 2, price))
            for k in range(2):
                price -= 1.0
                seq.append((price + 1, price + 1.5, price - 1.5, price))
        bars = mk(seq)
        st = market_bias(bars, len(bars) - 1, ema_fast=5, ema_slow=10, lookback=1)
        self.assertEqual(st.bias, Bias.BULL, st.reason)

    def test_ema_veto_blocks_counter_trend_bias(self) -> None:
        """Structure may say bull, but if price is under both EMAs we stand aside."""
        seq = [(100 + i * 0.5, 101 + i * 0.5, 99 + i * 0.5, 100.2 + i * 0.5)
               for i in range(60)]
        seq += [(130 - i * 3, 131 - i * 3, 128 - i * 3, 129 - i * 3) for i in range(15)]
        bars = mk(seq)
        st = market_bias(bars, len(bars) - 1, ema_fast=10, ema_slow=30, lookback=2)
        self.assertIn(st.bias, (Bias.NONE, Bias.BEAR))

    @staticmethod
    def _ob_scenario(disp_volume: float, block_volume: float = 1000.0) -> list[Bar]:
        """25 quiet bars, a small bearish candle, then a bullish displacement.

        The bearish candle is the order block; the last bar is the displacement.
        """
        seq = [(100, 100.5, 99.5, 100)] * 25
        seq += [(100, 100.4, 99.0, 99.2)]      # the order block candle
        seq += [(99.2, 112.0, 99.0, 111.5)]    # the displacement leg
        bars = mk(seq, vol=1000.0)
        bars[-2] = _with_volume(bars[-2], block_volume)
        bars[-1] = _with_volume(bars[-1], disp_volume)
        return bars

    def test_volume_filter_applies_to_the_displacement_leg(self) -> None:
        """Heavy volume on the displacement leg qualifies the block.

        Measured on real gold, the displacement candle carries ~2x normal
        volume while the block candle averages 1.0x -- so this is where the
        filter belongs.
        """
        bars = self._ob_scenario(disp_volume=4000.0)
        blocks = find_order_blocks(bars, len(bars) - 1, min_volume_ratio=1.4,
                                   volume_lookback=20)
        self.assertTrue(blocks, "a high-volume displacement block was missed")
        self.assertEqual(blocks[-1].side, Side.LONG)
        self.assertGreater(blocks[-1].volume_ratio, 1.4)

    def test_thin_displacement_volume_is_rejected(self) -> None:
        bars = self._ob_scenario(disp_volume=900.0)
        blocks = find_order_blocks(bars, len(bars) - 1, min_volume_ratio=1.4,
                                   volume_lookback=20)
        self.assertEqual(blocks, [], "a block was accepted on a thin displacement")

    def test_quiet_block_candle_does_not_disqualify(self) -> None:
        """Regression guard.

        An earlier version required heavy volume on the BLOCK candle. Real
        order-block candles are small and quiet by construction (median volume
        ratio 0.84 on gold), so that filter rejected essentially every genuine
        setup and the strategy produced no signals at all. The block candle's
        own volume must not be a gate.
        """
        bars = self._ob_scenario(disp_volume=4000.0, block_volume=200.0)
        blocks = find_order_blocks(bars, len(bars) - 1, min_volume_ratio=1.4,
                                   volume_lookback=20)
        self.assertTrue(
            blocks,
            "a valid setup was rejected because the block candle was quiet -- "
            "the volume filter has drifted back onto the wrong candle",
        )

    def test_block_is_invalidated_only_by_a_close_through_it(self) -> None:
        """Regression guard for the bug that made the setup impossible.

        A tap of the zone is the entry trigger. If a touch marked the block
        mitigated, then "price is at the zone" and "the zone is still valid"
        could never both hold, and no signal could ever fire.
        """
        # Displacement up, then price returns and WICKS into the zone without
        # closing below it.
        seq = [(100, 100.5, 99.5, 100)] * 25
        seq += [(100, 100.4, 99.0, 99.2)]     # block: 99.0 - 100.4
        seq += [(99.2, 112.0, 99.0, 111.5)]   # displacement
        seq += [(111.0, 111.5, 99.5, 101.0)]  # taps the zone, closes above it
        bars = mk(seq, vol=1000.0)
        bars[26] = _with_volume(bars[26], 4000.0)   # the displacement leg
        blocks = find_order_blocks(bars, len(bars) - 1, min_volume_ratio=1.4,
                                   volume_lookback=20)
        self.assertTrue(blocks)
        ob = blocks[-1]
        self.assertTrue(ob.touched, "the tap was not recorded")
        self.assertFalse(
            ob.mitigated,
            "a mere tap invalidated the block -- the setup can never trigger",
        )
        # And it must still be selectable for entry at the tapped price.
        self.assertIsNotNone(nearest_unmitigated_block([ob], Side.LONG, 100.0))

    def test_close_through_the_zone_does_invalidate(self) -> None:
        seq = [(100, 100.5, 99.5, 100)] * 25
        seq += [(100, 100.4, 99.0, 99.2)]     # block: 99.0 - 100.4
        seq += [(99.2, 112.0, 99.0, 111.5)]   # displacement
        seq += [(111.0, 111.5, 97.0, 97.5)]   # closes BELOW the zone
        bars = mk(seq, vol=1000.0)
        bars[26] = _with_volume(bars[26], 4000.0)   # the displacement leg
        blocks = find_order_blocks(bars, len(bars) - 1, min_volume_ratio=1.4,
                                   volume_lookback=20)
        self.assertTrue(blocks)
        self.assertTrue(blocks[-1].mitigated,
                        "a close through the zone should invalidate it")

    def test_block_containing_current_price_is_selectable(self) -> None:
        """Regression guard: the zone we are trading into contains the price."""
        from xau_agent.structure import OrderBlock
        inside = OrderBlock(1, 0, Side.LONG, 101.0, 99.0, 2.0, 2.0)
        self.assertIsNotNone(
            nearest_unmitigated_block([inside], Side.LONG, price=100.0),
            "a block price is trading inside was excluded -- that is the tap",
        )

    def test_nearest_block_ignores_wrong_side_of_price(self) -> None:
        from xau_agent.structure import OrderBlock
        below = OrderBlock(1, 0, Side.LONG, 99.0, 98.0, 2.0, 2.0)
        above = OrderBlock(2, 0, Side.LONG, 120.0, 119.0, 2.0, 2.0)
        got = nearest_unmitigated_block([below, above], Side.LONG, price=110.0)
        self.assertEqual(got, below)  # demand must sit below price

    def test_mitigated_blocks_skipped(self) -> None:
        from xau_agent.structure import OrderBlock
        used = OrderBlock(1, 0, Side.LONG, 99.0, 98.0, 2.0, 2.0, mitigated=True)
        self.assertIsNone(nearest_unmitigated_block([used], Side.LONG, 110.0))


class TestResample(unittest.TestCase):
    def test_15m_to_1h_aggregation(self) -> None:
        bars = synthetic_bars(400, interval="15m")
        h1 = resample(bars, "1h")
        self.assertAlmostEqual(len(h1), 100, delta=2)
        # First hourly bar must summarise its four 15m children exactly.
        first4 = bars[:4]
        self.assertAlmostEqual(h1[0].o, first4[0].o)
        self.assertAlmostEqual(h1[0].c, first4[-1].c)
        self.assertAlmostEqual(h1[0].h, max(b.h for b in first4))
        self.assertAlmostEqual(h1[0].l, min(b.l for b in first4))
        self.assertAlmostEqual(h1[0].v, sum(b.v for b in first4))

    def test_4h_buckets_align_to_epoch(self) -> None:
        h4 = resample(synthetic_bars(400, interval="15m"), "4h")
        for b in h4:
            self.assertEqual(b.t % 14400, 0)

    def test_resample_preserves_extremes(self) -> None:
        bars = synthetic_bars(400, interval="15m")
        h4 = resample(bars, "4h")
        self.assertAlmostEqual(max(b.h for b in h4), max(b.h for b in bars))
        self.assertAlmostEqual(min(b.l for b in h4), min(b.l for b in bars))

    def test_unknown_interval_raises(self) -> None:
        with self.assertRaises(ValueError):
            resample(synthetic_bars(10), "7h")


class TestStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.p = StrategyParams()
        self.s = Strategy(self.p)

    def test_no_signal_during_warmup(self) -> None:
        bars = synthetic_bars(300, interval="15m")
        v = MarketView.build(bars, self.p)
        out = self.s.evaluate(entry_view=v, entry_idx=5, bias_view=v, bias_idx=5,
                              zone_view=v, zone_idx=5)
        self.assertIsInstance(out, Rejection)

    def test_news_blackout_rejects_immediately(self) -> None:
        bars = synthetic_bars(600, interval="15m")
        v = MarketView.build(bars, self.p)
        out = self.s.evaluate(entry_view=v, entry_idx=500, bias_view=v, bias_idx=500,
                              zone_view=v, zone_idx=500, news_blackout=True)
        self.assertIsInstance(out, Rejection)
        self.assertIn("news", out.reason)

    def test_signals_always_have_sane_geometry(self) -> None:
        """Any signal produced must have SL and TP on the correct sides."""
        bars, zone, bias = dataset()
        ev, zv, bv = (MarketView.build(b, self.p) for b in (bars, zone, bias))
        zt, bt = [b.t for b in zone], [b.t for b in bias]

        found = 0
        for i in range(250, len(bars)):
            zi, bi = align_index(zt, bars[i].t), align_index(bt, bars[i].t)
            if zi < 50 or bi < self.p.ema_slow:
                continue
            out = self.s.evaluate(entry_view=ev, entry_idx=i, bias_view=bv,
                                  bias_idx=bi, zone_view=zv, zone_idx=zi)
            if isinstance(out, Rejection):
                continue
            found += 1
            if out.side is Side.LONG:
                self.assertLess(out.stop, out.entry)
                self.assertLess(out.entry, out.tp1)
                self.assertLess(out.tp1, out.tp2)
            else:
                self.assertGreater(out.stop, out.entry)
                self.assertGreater(out.entry, out.tp1)
                self.assertGreater(out.tp1, out.tp2)
            self.assertGreaterEqual(out.score, self.p.min_confluence_score)
            self.assertGreater(out.reward_risk, 1.0)
        self.assertGreater(found, 0, "strategy produced no signals at all on 2500 bars")

    def test_raising_threshold_reduces_signals(self) -> None:
        """The overtrading dial must actually work."""
        bars, zone, bias = dataset(seed=5)
        zt, bt = [b.t for b in zone], [b.t for b in bias]

        def count(threshold: int) -> int:
            p = StrategyParams(min_confluence_score=threshold)
            s = Strategy(p)
            ev, zv, bv = (MarketView.build(b, p) for b in (bars, zone, bias))
            n = 0
            for i in range(250, len(bars)):
                zi, bi = align_index(zt, bars[i].t), align_index(bt, bars[i].t)
                if zi < 50 or bi < p.ema_slow:
                    continue
                if not isinstance(s.evaluate(entry_view=ev, entry_idx=i,
                                             bias_view=bv, bias_idx=bi,
                                             zone_view=zv, zone_idx=zi), Rejection):
                    n += 1
            return n

        self.assertGreaterEqual(count(55), count(85))

    def test_invalid_params_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Strategy(StrategyParams(ema_fast=200, ema_slow=50))
        with self.assertRaises(ValueError):
            Strategy(StrategyParams(tp1_r=2.0, tp2_r=1.0))

    def test_only_tunable_params_can_be_overridden(self) -> None:
        p = StrategyParams()
        p.with_overrides(min_confluence_score=75)  # allowed
        with self.assertRaises(ValueError):
            p.with_overrides(symbol="OANDA:EURUSD")   # structural
        with self.assertRaises(ValueError):
            p.with_overrides(tp1_close_fraction=0.9)  # structural


class TestDataSufficiency(unittest.TestCase):
    """The silent-failure guard.

    Without this, too little history means every bar is skipped during warm-up
    and the system reports no setups indefinitely -- indistinguishable from
    patience, but actually a dead pipeline.
    """

    def test_short_history_raises_rather_than_silently_finding_nothing(self) -> None:
        from xau_agent.backtest import InsufficientData
        bars = synthetic_bars(2500, interval="15m")
        with self.assertRaises(InsufficientData) as ctx:
            Backtester(AgentConfig()).run(bars, resample(bars, "1h"),
                                          resample(bars, "4h"))
        msg = str(ctx.exception)
        self.assertIn("4h", msg)
        self.assertIn("days", msg)  # tells the user how much data is needed

    def test_adequate_history_passes_the_check(self) -> None:
        from xau_agent.backtest import check_data_sufficiency
        bars, zone, bias = dataset()
        check_data_sufficiency(bars, zone, bias, AgentConfig())  # must not raise

    def test_guard_can_be_bypassed_explicitly(self) -> None:
        """Opting out is allowed, but must be deliberate."""
        bars = synthetic_bars(2500, interval="15m")
        res = Backtester(AgentConfig()).run(
            bars, resample(bars, "1h"), resample(bars, "4h"),
            require_sufficient_data=False)
        self.assertEqual(res.stats.n, 0)  # confirms it really would find nothing


def rescale_volatility(bars: list[Bar], k: float, base: float = 4000.0) -> list[Bar]:
    """Rebuild a series with 1/k the bar-to-bar volatility, same structure.

    This stands in for viewing the same market on a lower timeframe: the shape
    of the path is preserved but every excursion is smaller.
    """
    out: list[Bar] = []
    price = base
    for b in bars:
        o = price
        c = o + (b.c - b.o) / k
        h = o + (b.h - b.o) / k
        l = o + (b.l - b.o) / k
        out.append(Bar(b.t, o, max(o, c, h), min(o, c, l), c, b.v))
        price = c
    return out


class TestVolatilityGateIsTimeframeIndependent(unittest.TestCase):
    """Regression guard for the bug that produced zero trades on 5m.

    The volatility filter was an absolute band on ATR as a percentage of price,
    tuned at 15m. Gold's ATR is ~0.38% of price on 1h but only ~0.11% on 5m, so
    the 0.12% floor rejected every single 5m bar. Because the gate is mandatory,
    the strategy silently produced no trades at all on that timeframe -- which
    looks identical to "no setups occurred".

    The gate is now a ratio to the instrument's own rolling average ATR, which
    is invariant to how large the bars are.
    """

    @staticmethod
    def pass_rate(bars: list[Bar], p: StrategyParams) -> float:
        view = MarketView.build(bars, p)
        checked = passed = 0
        for i in range(len(bars)):
            a, avg = view.atr[i], view.atr_avg[i]
            if a is None or avg is None or avg <= 0:
                continue
            checked += 1
            if p.min_vol_mult <= a / avg <= p.max_vol_mult:
                passed += 1
        return passed / checked if checked else 0.0

    def test_same_pass_rate_at_one_quarter_the_volatility(self) -> None:
        p = StrategyParams()
        loud = synthetic_bars(2000, interval="15m", seed=17)
        quiet = rescale_volatility(loud, 4.0)

        loud_rate = self.pass_rate(loud, p)
        quiet_rate = self.pass_rate(quiet, p)

        self.assertGreater(loud_rate, 0.5, "the gate rejects most normal bars")
        self.assertAlmostEqual(
            loud_rate, quiet_rate, delta=0.05,
            msg=f"the volatility gate is timeframe dependent: {loud_rate:.1%} "
                f"of bars pass at full volatility but {quiet_rate:.1%} at a "
                f"quarter of it. This is exactly the failure that produced "
                f"zero trades on 5m.",
        )

    def test_still_passes_at_one_twentieth_the_volatility(self) -> None:
        """A 5m chart is roughly this much quieter than a daily one."""
        p = StrategyParams()
        bars = rescale_volatility(synthetic_bars(2000, interval="5m", seed=23), 20.0)
        self.assertGreater(
            self.pass_rate(bars, p), 0.5,
            "a low-volatility timeframe is being filtered out wholesale",
        )

    def test_gate_still_rejects_genuine_extremes(self) -> None:
        """Relative must not mean toothless: a real volatility spike is caught."""
        p = StrategyParams()
        bars = list(synthetic_bars(1200, interval="15m", seed=31))
        # Blow out the last 20 bars to many times the prevailing range.
        for i in range(len(bars) - 20, len(bars)):
            b = bars[i]
            mid = (b.h + b.l) / 2
            span = (b.h - b.l) * 12
            bars[i] = Bar(b.t, b.o, mid + span / 2, mid - span / 2, b.c, b.v)
        view = MarketView.build(bars, p)
        i = len(bars) - 1
        a, avg = view.atr[i], view.atr_avg[i]
        self.assertIsNotNone(a)
        self.assertIsNotNone(avg)
        self.assertGreater(a / avg, p.max_vol_mult,
                           "a 12x volatility spike should breach the ceiling")

    def test_atr_avg_series_aligns_with_bars(self) -> None:
        p = StrategyParams()
        bars = synthetic_bars(600, interval="15m")
        view = MarketView.build(bars, p)
        self.assertEqual(len(view.atr_avg), len(bars))
        self.assertIsNone(view.atr_avg[0], "baseline cannot exist on bar 0")
        self.assertIsNotNone(view.atr_avg[-1])

    def test_absolute_atr_percent_band_is_gone(self) -> None:
        """The old timeframe-dependent parameters must not come back."""
        p = StrategyParams()
        self.assertFalse(hasattr(p, "min_atr_pct"))
        self.assertFalse(hasattr(p, "max_atr_pct"))


class TestBacktester(unittest.TestCase):
    def test_runs_and_respects_daily_trade_cap(self) -> None:
        cfg = AgentConfig()
        bars, zone, bias = dataset(seed=3)
        res = Backtester(cfg).run(bars, zone, bias)

        self.assertGreater(res.bars_tested, 0)
        by_day: dict[str, int] = {}
        for t in res.trades:
            k = t.signal_time.date().isoformat()
            by_day[k] = by_day.get(k, 0) + 1
        for day, n in by_day.items():
            self.assertLessEqual(
                n, cfg.risk.max_trades_per_day,
                f"{day} took {n} trades, above the cap",
            )

    def test_never_two_positions_at_once(self) -> None:
        bars, zone, bias = dataset(seed=9)
        res = Backtester(AgentConfig()).run(bars, zone, bias)
        trades = sorted(res.trades, key=lambda t: t.signal_time)
        for a, b in zip(trades, trades[1:]):
            if a.exit_time:
                self.assertGreaterEqual(
                    b.signal_time, a.exit_time,
                    "a trade opened while another was still running",
                )

    def test_loss_never_exceeds_risk_budget_materially(self) -> None:
        """No single simulated loss should blow well past the per-trade budget.

        Slippage and spread mean it can exceed slightly; a large overshoot means
        the sizing or fill logic is wrong.
        """
        cfg = AgentConfig()
        bars, zone, bias = dataset(seed=21)
        res = Backtester(cfg).run(bars, zone, bias)
        budget = cfg.risk.risk_dollars()
        for t in res.trades:
            if t.pnl < 0:
                self.assertLess(
                    abs(t.pnl), budget * 1.6,
                    f"loss {t.pnl:.2f} far exceeds the {budget:.2f} budget",
                )

    def test_r_multiple_consistent_with_pnl_sign(self) -> None:
        bars, zone, bias = dataset(seed=13)
        res = Backtester(AgentConfig()).run(bars, zone, bias)
        for t in res.trades:
            if t.outcome == Outcome.WIN.value:
                self.assertGreater(t.r_multiple, 0)
            elif t.outcome == Outcome.LOSS.value:
                self.assertLess(t.r_multiple, 0)

    def test_report_renders(self) -> None:
        bars, zone, bias = dataset(seed=4)
        res = Backtester(AgentConfig()).run(bars, zone, bias)
        self.assertIn("BACKTEST RESULT", res.report())


class TestStatsAndGate(unittest.TestCase):
    @staticmethod
    def entries(pattern: list[float], mode: str = Mode.PAPER.value) -> list[JournalEntry]:
        out = []
        for i, r in enumerate(pattern):
            out.append(JournalEntry(
                signal_time=datetime(2024, 5, 1, 10, tzinfo=timezone.utc).replace(
                    day=1 + (i % 28)),
                side="long", entry=2000.0, stop=1990.0, tp1=2010.0, tp2=2022.0,
                score=75, session="london", mode=mode, lots=0.25,
                exit_price=2000.0 + r * 10, pnl=r * 250.0, r_multiple=r,
                outcome=(Outcome.WIN.value if r > 0 else
                         Outcome.LOSS.value if r < 0 else Outcome.BREAKEVEN.value),
            ))
        return out

    def test_wilson_interval_widens_on_small_samples(self) -> None:
        lo_small, hi_small = wilson_interval(6, 10)
        lo_big, hi_big = wilson_interval(600, 1000)
        self.assertLess(hi_small - lo_small, 1.01)
        self.assertGreater(hi_small - lo_small, hi_big - lo_big)
        # 6/10 really does not establish a 60% win rate.
        self.assertLess(lo_small, 0.5)

    def test_wilson_empty(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_breakeven_excluded_from_win_rate(self) -> None:
        st = compute_stats(self.entries([2.0, -1.0, 0.0, 2.0]))
        self.assertEqual(st.n, 4)
        self.assertAlmostEqual(st.win_rate, 2 / 3)  # 2 wins of 3 decided

    def test_expectancy_and_profit_factor(self) -> None:
        st = compute_stats(self.entries([2.0, -1.0, 2.0, -1.0]))
        self.assertAlmostEqual(st.expectancy_r, 0.5)
        self.assertAlmostEqual(st.profit_factor, 2.0)

    def test_max_drawdown_r(self) -> None:
        st = compute_stats(self.entries([2.0, -1.0, -1.0, -1.0, 2.0]))
        self.assertAlmostEqual(st.max_drawdown_r, 3.0)
        self.assertEqual(st.max_consecutive_losses, 3)

    def test_gate_refuses_tiny_sample_even_at_60_percent(self) -> None:
        """The headline requirement: 6/10 winners does NOT unlock live mode."""
        gate = PerformanceGate(AgentConfig())
        res = gate.evaluate(self.entries([2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                          -1.0, -1.0, -1.0, -1.0]))
        self.assertFalse(res.passed)
        self.assertEqual(res.mode, Mode.PAPER)
        self.assertTrue(any("forward-tested trades" in r for r in res.reasons))

    @staticmethod
    def interleaved(wins: int, losses: int, win_r: float = 1.6,
                    loss_r: float = -1.0) -> list[float]:
        """Wins and losses spread evenly, as a real equity curve would be.

        Stacking all wins then all losses creates an artificial drawdown that
        has nothing to do with the strategy.
        """
        total = wins + losses
        out: list[float] = []
        w = l = 0
        for i in range(total):
            if w * losses <= l * wins and w < wins:
                out.append(win_r)
                w += 1
            else:
                out.append(loss_r)
                l += 1
        return out

    def test_gate_passes_on_a_real_sample(self) -> None:
        """60 trades at 60% with realistic sequencing clears the gate."""
        pattern = self.interleaved(wins=36, losses=24)
        res = PerformanceGate(AgentConfig()).evaluate(self.entries(pattern))
        self.assertTrue(res.passed, res.report())
        self.assertEqual(res.mode, Mode.LIVE)

    def test_gate_warns_that_60_percent_is_not_yet_proven(self) -> None:
        """Passing the gate must not be reported as having proven 60%."""
        res = PerformanceGate(AgentConfig()).evaluate(
            self.entries(self.interleaved(wins=36, losses=24)))
        self.assertTrue(res.passed)
        self.assertTrue(
            any("not yet statistically established" in r for r in res.reasons),
            "the gate passed without flagging that 60% is still unproven",
        )

    def test_gate_blocks_on_excessive_drawdown(self) -> None:
        """A long losing run demotes to paper even with a good win rate."""
        pattern = [1.6] * 40 + [-1.0] * 20  # 20 losses in a row = 20R drawdown
        res = PerformanceGate(AgentConfig()).evaluate(self.entries(pattern))
        self.assertFalse(res.passed)
        self.assertTrue(any("drawdown" in r for r in res.reasons))

    def test_gate_blocks_just_below_sample_threshold(self) -> None:
        cfg = AgentConfig()
        n = cfg.min_trades_for_live - 1
        res = PerformanceGate(cfg).evaluate(
            self.entries(self.interleaved(wins=int(n * 0.6), losses=n - int(n * 0.6))))
        self.assertFalse(res.passed)

    def test_gate_demotes_on_poor_expectancy(self) -> None:
        """Enough trades and a 60% win rate, but the winners are too small."""
        entries = self.entries([0.3] * 27 + [-1.0] * 18)
        res = PerformanceGate(AgentConfig()).evaluate(entries)
        self.assertFalse(res.passed)
        self.assertTrue(any("expectancy" in r for r in res.reasons))

    def test_gate_ignores_backtest_trades(self) -> None:
        """Backtest results must never unlock live mode on their own."""
        entries = self.entries(self.interleaved(wins=60, losses=40),
                               mode=Mode.BACKTEST.value)
        res = PerformanceGate(AgentConfig()).evaluate(entries)
        self.assertFalse(res.passed)

    def test_config_rejects_incoherent_gate_settings(self) -> None:
        """A floor at or above the target makes the gate unopenable."""
        with self.assertRaises(ValueError):
            AgentConfig(min_trades_for_live=60).__class__(
                live_win_rate_floor=0.65, target_win_rate=0.60).validate()


class TestJournalPersistence(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.j = Journal(f"{self.tmp.name}/j.db")

    def tearDown(self) -> None:
        self.j.close()
        self.tmp.cleanup()

    def test_round_trip(self) -> None:
        e = JournalEntry(
            signal_time=datetime(2024, 5, 1, 10, tzinfo=timezone.utc),
            side="long", entry=2000.0, stop=1990.0, tp1=2010.0, tp2=2022.0,
            score=78, session="london", mode=Mode.PAPER.value, lots=0.25,
            exit_price=2022.0, exit_time=datetime(2024, 5, 1, 14, tzinfo=timezone.utc),
            pnl=550.0, r_multiple=2.2, outcome=Outcome.WIN.value,
        )
        self.j.record(e)
        got = self.j.trades()
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0].pnl, 550.0)
        self.assertEqual(got[0].outcome, Outcome.WIN.value)

    def test_session_state_survives_reload(self) -> None:
        """The daily loss cap must not reset when the process restarts."""
        from datetime import date
        from xau_agent.risk import SessionState
        s = SessionState(trading_day=date(2024, 5, 15), realised_pnl=-420.0,
                         trades_today=2, consecutive_losses=1, week_pnl=-420.0)
        self.j.save_state(s)
        back = self.j.load_state(date(2024, 5, 15))
        self.assertAlmostEqual(back.realised_pnl, -420.0)
        self.assertEqual(back.trades_today, 2)
        self.assertEqual(back.consecutive_losses, 1)

    def test_new_day_carries_week_total(self) -> None:
        from datetime import date
        from xau_agent.risk import SessionState
        self.j.save_state(SessionState(trading_day=date(2024, 5, 15),
                                       realised_pnl=-300.0, week_pnl=-300.0,
                                       month_pnl=-300.0))
        nxt = self.j.load_state(date(2024, 5, 16))
        self.assertEqual(nxt.realised_pnl, 0.0)      # fresh day
        self.assertAlmostEqual(nxt.week_pnl, -300.0)  # same week, carried

    def test_rejections_recorded(self) -> None:
        self.j.record_rejection(datetime.now(tz=timezone.utc), 55, 70, "score too low")
        self.j.record_rejection(datetime.now(tz=timezone.utc), 40, 70, "score too low")
        self.assertEqual(self.j.rejection_summary()["score too low"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
