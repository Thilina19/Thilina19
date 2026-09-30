"""The XAUUSD confluence strategy.

The thesis in one line: trade *with* the 4h structure, *from* a volume-backed
1h order block, *timed* by a 15m Heikin Ashi flip.

Nothing here is original -- it is the standard SMC continuation trade. The value
is not the pattern, it is that every condition is explicit, scored, and logged,
so when it stops working the journal can tell us which condition decayed.

Scoring rather than boolean AND is deliberate. A hard AND of eight conditions
either never triggers or gets loosened until it triggers on noise. A score with
a published threshold lets us tighten or loosen one dial (`min_confluence_score`)
and measure the effect on win rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import StrategyParams
from .indicators import Bar, HABar, atr, closes, ema, heikin_ashi, rsi, volume_ratio
from .structure import (
    Bias,
    OrderBlock,
    Side,
    StructureState,
    find_order_blocks,
    market_bias,
    nearest_unmitigated_block,
    structural_stop_level,
)


@dataclass(frozen=True)
class ScoreItem:
    name: str
    points: int
    max_points: int
    passed: bool
    detail: str


@dataclass(frozen=True)
class Signal:
    """A complete, executable trade plan. All prices are real, not Heikin Ashi."""

    t: int
    side: Side
    entry: float
    stop: float
    tp1: float
    tp2: float
    score: int
    atr: float
    session: str
    items: list[ScoreItem] = field(default_factory=list)
    narrative: str = ""

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.t, tz=timezone.utc)

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_risk(self) -> float:
        r = self.risk_distance
        return abs(self.tp2 - self.entry) / r if r > 0 else 0.0

    def explain(self) -> str:
        lines = [
            f"{self.side.value.upper()} XAUUSD @ {self.entry:.2f}",
            f"  SL {self.stop:.2f}   TP1 {self.tp1:.2f}   TP2 {self.tp2:.2f}",
            f"  score {self.score}/100   R:R {self.reward_risk:.2f}   "
            f"session {self.session}   ATR {self.atr:.2f}",
        ]
        for it in self.items:
            mark = "+" if it.passed else "-"
            lines.append(f"   {mark} {it.name}: {it.points}/{it.max_points}  {it.detail}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Rejection:
    """Why no signal was produced. Logged so we can audit what we skipped."""

    t: int
    score: int
    threshold: int
    items: list[ScoreItem] = field(default_factory=list)
    reason: str = ""


def session_for(dt: datetime, p: StrategyParams) -> str:
    h = dt.hour
    if h in p.ny_hours:
        return "newyork"
    if h in p.london_hours:
        return "london"
    if h in p.asian_hours:
        return "asian"
    return "offhours"


@dataclass
class MarketView:
    """Pre-computed indicator series for one timeframe, to avoid recomputing
    them for every bar during a backtest."""

    bars: list[Bar]
    atr: list[float | None]
    rsi: list[float | None]
    vol_ratio: list[float | None]
    ha: list[HABar]
    ema_fast: list[float | None]
    ema_slow: list[float | None]

    @classmethod
    def build(cls, bars: list[Bar], p: StrategyParams) -> "MarketView":
        cl = closes(bars)
        return cls(
            bars=bars,
            atr=atr(bars, p.atr_period),
            rsi=rsi(bars, 14),
            vol_ratio=volume_ratio(bars, p.ob_volume_lookback),
            ha=heikin_ashi(bars),
            ema_fast=ema(cl, p.ema_fast),
            ema_slow=ema(cl, p.ema_slow),
        )


class Strategy:
    """Scores the setup at a given entry-timeframe bar.

    The scoring weights sum to 100:
        HTF bias aligned .......... 25   (non-negotiable: 0 here means no trade)
        Order block quality ....... 20
        Price at the zone ......... 15
        Heikin Ashi trigger ....... 15   (non-negotiable: 0 here means no trade)
        Volume confirmation ....... 10
        Volatility regime ......... 10
        Momentum not exhausted .....5
    """

    # Conditions that cannot be compensated for by other confluence.
    MANDATORY = ("htf_bias", "ha_trigger")

    def __init__(self, params: StrategyParams):
        params.validate()
        self.p = params

    def evaluate(
        self,
        *,
        entry_view: MarketView,
        entry_idx: int,
        bias_view: MarketView,
        bias_idx: int,
        zone_view: MarketView,
        zone_idx: int,
        news_blackout: bool = False,
    ) -> Signal | Rejection:
        """Evaluate the setup at `entry_idx` on the entry timeframe.

        All three views are indexed independently because the timeframes are
        not aligned; the caller is responsible for passing indices that refer
        to the same wall-clock moment (see backtest.align_index).
        """
        p = self.p
        bar = entry_view.bars[entry_idx]
        dt = bar.dt
        items: list[ScoreItem] = []

        if news_blackout:
            return Rejection(bar.t, 0, p.min_confluence_score, items,
                             "high-impact news blackout")

        a = entry_view.atr[entry_idx]
        if a is None or a <= 0:
            return Rejection(bar.t, 0, p.min_confluence_score, items,
                             "ATR unavailable (warm-up)")

        # ---------------------------------------------------- 1. HTF bias (25)
        struct: StructureState = market_bias(
            bias_view.bars, bias_idx,
            ema_fast=p.ema_fast, ema_slow=p.ema_slow, lookback=p.swing_lookback,
        )
        if struct.bias is Bias.NONE:
            items.append(ScoreItem("htf_bias", 0, 25, False, struct.reason))
            return Rejection(bar.t, 0, p.min_confluence_score, items,
                             f"no {p.bias_tf} bias: {struct.reason}")

        side = Side.LONG if struct.bias is Bias.BULL else Side.SHORT
        bias_pts = 25 if (struct.bos and not struct.choch) else 18
        items.append(
            ScoreItem("htf_bias", bias_pts, 25, True,
                      f"{p.bias_tf} {struct.bias.value} ({struct.reason})")
        )

        # --------------------------------------------- 2. order block quality (20)
        blocks = find_order_blocks(
            zone_view.bars, zone_idx,
            atr_values=zone_view.atr, vol_ratios=zone_view.vol_ratio,
            displacement_atr=p.ob_displacement_atr,
            min_volume_ratio=p.ob_min_volume_ratio,
            volume_lookback=p.ob_volume_lookback,
            max_age_bars=p.ob_max_age_bars,
        )
        block = nearest_unmitigated_block(blocks, side, bar.c)
        if block is None:
            items.append(ScoreItem("order_block", 0, 20, False,
                                   f"no unmitigated {side.value} block on {p.zone_tf}"))
            return Rejection(bar.t, sum(i.points for i in items),
                             p.min_confluence_score, items, "no valid order block")

        # Strong displacement and heavy volume both raise confidence.
        ob_pts = 10
        if block.displacement_atr >= p.ob_displacement_atr * 1.5:
            ob_pts += 5
        if block.volume_ratio >= p.ob_min_volume_ratio * 1.4:
            ob_pts += 5
        items.append(
            ScoreItem("order_block", ob_pts, 20, True,
                      f"{block.bottom:.2f}-{block.top:.2f}, "
                      f"disp {block.displacement_atr:.2f}atr, "
                      f"vol {block.volume_ratio:.2f}x")
        )

        # ------------------------------------------------ 3. price at zone (15)
        tol = p.ob_tap_tolerance_atr * a
        at_zone = block.contains(bar.l if side is Side.LONG else bar.h, tol)
        dist = abs(bar.c - block.mid)
        if at_zone:
            zone_pts = 15
            zdetail = "price tapped the zone"
        elif dist <= 2.0 * a:
            zone_pts = 7
            zdetail = f"approaching zone ({dist / a:.2f} atr away)"
        else:
            zone_pts = 0
            zdetail = f"too far from zone ({dist / a:.2f} atr)"
        items.append(ScoreItem("at_zone", zone_pts, 15, at_zone, zdetail))

        # -------------------------------------------- 4. Heikin Ashi trigger (15)
        ha_pts, ha_ok, ha_detail = self._ha_trigger(entry_view, entry_idx, side)
        items.append(ScoreItem("ha_trigger", ha_pts, 15, ha_ok, ha_detail))
        if not ha_ok:
            return Rejection(bar.t, sum(i.points for i in items),
                             p.min_confluence_score, items,
                             f"Heikin Ashi trigger absent: {ha_detail}")

        # ------------------------------------------------- 5. entry volume (10)
        evr = entry_view.vol_ratio[entry_idx]
        if evr is None:
            vol_pts, vdetail = 0, "volume unavailable"
        elif evr >= 1.2:
            vol_pts, vdetail = 10, f"entry volume {evr:.2f}x average"
        elif evr >= 0.9:
            vol_pts, vdetail = 5, f"entry volume {evr:.2f}x (adequate)"
        else:
            vol_pts, vdetail = 0, f"entry volume {evr:.2f}x (thin)"
        items.append(ScoreItem("entry_volume", vol_pts, 10, vol_pts > 0, vdetail))

        # -------------------------------------------- 6. volatility regime (10)
        atr_pct = a / bar.c
        if p.min_atr_pct <= atr_pct <= p.max_atr_pct:
            vol_regime_pts = 10
            rdetail = f"ATR {atr_pct * 100:.3f}% of price (in band)"
            regime_ok = True
        else:
            vol_regime_pts = 0
            regime_ok = False
            rdetail = (
                f"ATR {atr_pct * 100:.3f}% outside "
                f"{p.min_atr_pct * 100:.3f}-{p.max_atr_pct * 100:.3f}%"
            )
        items.append(ScoreItem("volatility", vol_regime_pts, 10, regime_ok, rdetail))
        if not regime_ok:
            return Rejection(bar.t, sum(i.points for i in items),
                             p.min_confluence_score, items,
                             f"volatility regime unsuitable: {rdetail}")

        # ---------------------------------------------------- 7. momentum (5)
        r = entry_view.rsi[entry_idx]
        if r is None:
            mom_pts, mdetail = 0, "RSI unavailable"
        elif side is Side.LONG and r < 68:
            mom_pts, mdetail = 5, f"RSI {r:.1f}, room above"
        elif side is Side.SHORT and r > 32:
            mom_pts, mdetail = 5, f"RSI {r:.1f}, room below"
        else:
            mom_pts, mdetail = 0, f"RSI {r:.1f} already extended"
        items.append(ScoreItem("momentum", mom_pts, 5, mom_pts > 0, mdetail))

        # -------------------------------------------------------- session bar
        sess = session_for(dt, p)
        score = sum(i.points for i in items)
        if sess == "asian":
            score -= p.asian_score_penalty
            items.append(
                ScoreItem("session", -p.asian_score_penalty, 0, False,
                          f"asian session penalty (chop risk)")
            )
        score = max(0, min(100, score))

        if score < p.min_confluence_score:
            return Rejection(bar.t, score, p.min_confluence_score, items,
                             f"score {score} below threshold {p.min_confluence_score}")

        # ------------------------------------------------ build the trade plan
        entry, stop, tp1, tp2 = self._levels(entry_view, entry_idx, side, block, a)
        if (side is Side.LONG and not (stop < entry < tp1 < tp2)) or (
            side is Side.SHORT and not (stop > entry > tp1 > tp2)
        ):
            return Rejection(bar.t, score, p.min_confluence_score, items,
                             "level geometry invalid")

        return Signal(
            t=bar.t, side=side, entry=entry, stop=stop, tp1=tp1, tp2=tp2,
            score=score, atr=a, session=sess, items=items,
            narrative=self._narrate(side, struct, block, sess, score),
        )

    # ------------------------------------------------------------- internals

    def _ha_trigger(
        self, view: MarketView, idx: int, side: Side
    ) -> tuple[int, bool, str]:
        """Require `ha_confirm_bars` clean Heikin Ashi candles in our direction,
        with the flip having happened recently rather than long ago.

        The wick filter matters: an HA candle that is mostly wick is
        indecision. Trading it is how you get stopped out by the very move you
        were trying to join.
        """
        p = self.p
        n = p.ha_confirm_bars
        if idx < n:
            return 0, False, "insufficient HA history"

        recent = view.ha[idx - n + 1 : idx + 1]
        want_bull = side is Side.LONG
        if not all(h.is_bull == want_bull for h in recent):
            return 0, False, f"last {n} HA candles not uniformly {side.value}"

        worst_wick = max(h.wick_ratio for h in recent)
        if worst_wick > p.ha_max_wick_ratio:
            return 0, False, f"HA wick ratio {worst_wick:.2f} too indecisive"

        # Was this an actual flip, or are we chasing a mature run?
        before = view.ha[idx - n]
        flipped = before.is_bull != want_bull
        pts = 15 if flipped else 9
        detail = (
            f"{n} clean {side.value} HA candles"
            + (" on a fresh flip" if flipped else " (continuation, not a flip)")
            + f", wick {worst_wick:.2f}"
        )
        return pts, True, detail

    def _levels(
        self, view: MarketView, idx: int, side: Side, block: OrderBlock, a: float
    ) -> tuple[float, float, float, float]:
        """Entry at the close, stop beyond structure *and* the order block.

        Taking the further of the two protects against the common failure where
        a stop sits just inside the zone that is supposed to defend it.
        """
        p = self.p
        entry = view.bars[idx].c
        swing = structural_stop_level(view.bars, idx, side, lookback=12)
        buf = p.sl_atr_buffer * a

        if side is Side.LONG:
            stop = min(swing, block.bottom) - buf
            r = entry - stop
            return entry, stop, entry + p.tp1_r * r, entry + p.tp2_r * r

        stop = max(swing, block.top) + buf
        r = stop - entry
        return entry, stop, entry - p.tp1_r * r, entry - p.tp2_r * r

    def _narrate(
        self, side: Side, struct: StructureState, block: OrderBlock,
        session: str, score: int,
    ) -> str:
        d = "bullish" if side is Side.LONG else "bearish"
        return (
            f"{self.p.bias_tf} structure is {d} ({struct.reason}). Price returned "
            f"to a {self.p.zone_tf} {d} order block at "
            f"{block.bottom:.2f}-{block.top:.2f} that formed on "
            f"{block.volume_ratio:.2f}x average volume with a "
            f"{block.displacement_atr:.2f} ATR displacement leg. Heikin Ashi on "
            f"{self.p.entry_tf} has turned {d}. Confluence {score}/100 during the "
            f"{session} session."
        )
