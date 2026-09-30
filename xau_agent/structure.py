"""Smart-money-concepts market structure: trend, BOS/CHoCH, order blocks, FVG.

A note on lookahead, because this is where backtests quietly lie:

`find_swings` can only confirm a swing `lookback` bars after it printed. Every
function here that is called during a bar-by-bar simulation takes an explicit
`upto` index and refuses to look at any bar beyond it. A swing at index i is
treated as unknown until i + lookback. If you ever see a suspiciously good
backtest, this is the first place to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .indicators import Bar, Swing, atr, closes, ema, find_swings, volume_ratio


class Bias(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NONE = "none"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class OrderBlock:
    """The last opposing candle before an impulsive displacement leg.

    `side` is the direction we would trade *from* this block: a bullish order
    block is a demand zone we buy from.
    """

    index: int
    t: int
    side: Side
    top: float
    bottom: float
    # Volume ratio of the DISPLACEMENT leg, not of the block candle. See
    # find_order_blocks for the measurement that decided this.
    volume_ratio: float
    displacement_atr: float
    # Invalidated: price CLOSED through the far edge of the zone, so the level
    # failed to hold and we no longer trade from it.
    #
    # Note this is deliberately NOT "price touched the zone". A tap is the
    # entry trigger we are waiting for -- treating a touch as invalidation
    # makes the setup impossible to ever take, since the conditions "price is
    # at the zone" and "price has never reached the zone" cannot both hold.
    mitigated: bool = False
    # Informational: price has traded into the zone at least once without
    # closing through it. This is a valid, live setup.
    touched: bool = False

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.bottom - tolerance) <= price <= (self.top + tolerance)


@dataclass(frozen=True)
class FairValueGap:
    """A three-bar imbalance where price moved so fast it left an untraded gap."""

    index: int
    t: int
    side: Side
    top: float
    bottom: float

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(frozen=True)
class StructureState:
    bias: Bias
    last_swing_high: Swing | None
    last_swing_low: Swing | None
    # True when the most recent structural event broke the prior range in the
    # direction of bias (break of structure) rather than against it (change of
    # character).
    bos: bool
    choch: bool
    reason: str


def confirmed_swings(bars: list[Bar], upto: int, lookback: int = 2) -> list[Swing]:
    """Swings that an observer standing at bar `upto` could actually know about.

    A swing at index i needs `lookback` bars after it to be confirmed, so we
    only report swings with i + lookback <= upto.
    """
    if upto < 0:
        return []
    visible = bars[: upto + 1]
    return [s for s in find_swings(visible, lookback) if s.index + lookback <= upto]


def market_bias(
    bars: list[Bar],
    upto: int,
    *,
    ema_fast: int = 50,
    ema_slow: int = 200,
    lookback: int = 2,
    mode: str = "ema_fallback",
) -> StructureState:
    """Decide directional bias from structure first, moving averages second.

    Structure is the primary signal: higher highs and higher lows is an uptrend
    regardless of what a moving average says. The EMAs act as a veto -- if
    structure says bull but price is below both EMAs, we call it NONE and stand
    aside rather than fight the larger trend.

    Modes:
      "strict"       -- structure only. Requires the last two confirmed swing
                        highs AND the last two confirmed swing lows to agree.
      "ema_fallback" -- as above, but when structure is a range, fall back to
                        the EMA trend (price above a rising fast EMA that is
                        itself above the slow EMA, or the mirror image).

    Why the fallback is the default: measured over 69 days of real 4h gold,
    strict structure produced a usable bias on only 45.5% of bars. Because bias
    is a mandatory gate, that alone roughly halves the signal rate before any
    other condition is considered, and in a ranging stretch it went to 25%. The
    fallback lifts availability to 69.7% on the same data without abandoning
    structure -- structure still wins wherever it has an opinion, and the EMA
    veto still applies to it. Demanding that both swings agree is a statement
    about clean trends, and gold is not in one most of the time.
    """
    if mode not in ("strict", "ema_fallback"):
        raise ValueError(f"unknown bias mode {mode!r}")
    swings = confirmed_swings(bars, upto, lookback)
    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if not s.is_high]

    last_h = highs[-1] if highs else None
    last_l = lows[-1] if lows else None

    if len(highs) < 2 or len(lows) < 2:
        return StructureState(Bias.NONE, last_h, last_l, False, False,
                              "not enough confirmed swings")

    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price

    struct_bias = Bias.NONE
    reason = "swings not aligned (range)"
    if hh and hl:
        struct_bias, reason = Bias.BULL, "higher high + higher low"
    elif lh and ll:
        struct_bias, reason = Bias.BEAR, "lower high + lower low"

    # Which structural event happened most recently, the high or the low?
    newest_is_high = last_h is not None and (
        last_l is None or last_h.index > last_l.index
    )
    bos = choch = False
    if struct_bias is Bias.BULL:
        bos, choch = (hh, ll) if newest_is_high else (hl, ll)
    elif struct_bias is Bias.BEAR:
        bos, choch = (ll, hh) if not newest_is_high else (lh, hh)

    # EMA veto.
    cl = closes(bars[: upto + 1])
    ef = ema(cl, ema_fast)[upto] if upto < len(cl) else None
    es = ema(cl, ema_slow)[upto] if upto < len(cl) else None
    price = bars[upto].c

    final = struct_bias
    if ef is not None and es is not None and struct_bias is not Bias.NONE:
        if struct_bias is Bias.BULL and price < min(ef, es):
            final = Bias.NONE
            reason += "; vetoed: price below both EMAs"
        elif struct_bias is Bias.BEAR and price > max(ef, es):
            final = Bias.NONE
            reason += "; vetoed: price above both EMAs"

    # Structure had no opinion (or was vetoed): fall back to the EMA trend.
    # This requires full alignment -- price on the correct side of the fast EMA
    # and the fast EMA on the correct side of the slow one -- so it is a trend
    # statement, not merely "price is above a line".
    if final is Bias.NONE and mode == "ema_fallback" and ef is not None and es is not None:
        if price > ef > es:
            final = Bias.BULL
            reason = f"EMA trend fallback (price > EMA{ema_fast} > EMA{ema_slow})"
            bos = choch = False
        elif price < ef < es:
            final = Bias.BEAR
            reason = f"EMA trend fallback (price < EMA{ema_fast} < EMA{ema_slow})"
            bos = choch = False

    return StructureState(final, last_h, last_l, bos, choch, reason)


def find_order_blocks(
    bars: list[Bar],
    upto: int,
    *,
    atr_values: list[float | None] | None = None,
    vol_ratios: list[float | None] | None = None,
    displacement_atr: float = 1.3,
    min_volume_ratio: float = 1.4,
    volume_lookback: int = 20,
    max_age_bars: int = 60,
) -> list[OrderBlock]:
    """Volume-weighted order blocks visible at bar `upto`, newest last.

    The recipe:
      1. Find a displacement leg -- a candle whose body covers at least
         `displacement_atr` multiples of ATR, AND whose volume exceeds
         `min_volume_ratio` times its rolling average. That combination is the
         institutional footprint.
      2. Step back to the last candle of the opposite colour before it. That
         candle is the order block, and its high/low define the zone.
      3. Discard blocks older than `max_age_bars`; mark blocks price has closed
         through as mitigated.

    WHERE THE VOLUME FILTER GOES, AND WHY IT MOVED
    ----------------------------------------------
    The usual retail formulation puts the volume requirement on the order-block
    candle, on the theory that it shows institutional accumulation. Measured
    against 400 real OANDA:XAUUSD 1h bars, that theory does not hold:

        order-block candle volume ratio:  mean 1.01, median 0.84
                                          only 3 of 14 above 1.4x
        displacement candle volume ratio: mean 2.15, median 1.83
                                          10 of 14 above 1.4x

    The block candle is by construction a small, quiet candle before the move,
    so demanding heavy volume on it rejects almost every real setup -- with the
    filter there, the detector found 3 blocks in 399 bars, and the strategy
    produced no signals at all. The displacement leg carries about twice the
    volume of a typical bar, which is exactly the confirmation we wanted.

    So the volume test is applied to the displacement leg. Same intent, placed
    where the evidence says the footprint actually is.
    """
    if upto < volume_lookback + 2:
        return []

    window = bars[: upto + 1]
    av = atr_values if atr_values is not None else atr(window)
    vr = vol_ratios if vol_ratios is not None else volume_ratio(window, volume_lookback)

    blocks: list[OrderBlock] = []
    start = max(1, upto - max_age_bars)

    for i in range(start, upto + 1):
        a = av[i] if i < len(av) else None
        if a is None or a <= 0:
            continue
        disp = bars[i]
        disp_mult = disp.body / a
        if disp_mult < displacement_atr:
            continue

        # Volume confirmation on the displacement leg itself.
        ratio = vr[i] if i < len(vr) else None
        if ratio is None or ratio < min_volume_ratio:
            continue

        # Walk back for the last candle of opposite colour -- that is the block.
        ob_idx = None
        for j in range(i - 1, max(start - 1, 0) - 1, -1):
            if bars[j].is_bull != disp.is_bull:
                ob_idx = j
                break
        if ob_idx is None:
            continue

        ob = bars[ob_idx]
        side = Side.LONG if disp.is_bull else Side.SHORT

        after = bars[i + 1 : upto + 1]
        # Invalidated only if a bar CLOSED beyond the far side of the zone. For
        # a demand block that means a close below its low: the level was
        # offered and rejected, so it is no longer support.
        if side is Side.LONG:
            mitigated = any(b.c < ob.l for b in after)
        else:
            mitigated = any(b.c > ob.h for b in after)
        # A touch without a close-through leaves the block live.
        touched = any(b.l <= ob.h and b.h >= ob.l for b in after)

        blocks.append(
            OrderBlock(
                index=ob_idx,
                t=ob.t,
                side=side,
                top=ob.h,
                bottom=ob.l,
                volume_ratio=ratio,
                displacement_atr=disp_mult,
                mitigated=mitigated,
                touched=touched,
            )
        )

    # De-duplicate: several displacement legs can point at the same block.
    unique: dict[int, OrderBlock] = {}
    for b in blocks:
        prior = unique.get(b.index)
        if prior is None or b.displacement_atr > prior.displacement_atr:
            unique[b.index] = b
    return sorted(unique.values(), key=lambda b: b.index)


def find_fvgs(bars: list[Bar], upto: int, *, max_age_bars: int = 60) -> list[FairValueGap]:
    """Three-bar fair value gaps visible at `upto`.

    Bullish FVG: bar[i-1].high < bar[i+1].low, leaving the middle bar's range
    partly untraded on the way up.
    """
    out: list[FairValueGap] = []
    start = max(1, upto - max_age_bars)
    for i in range(start, upto):
        prev, nxt = bars[i - 1], bars[i + 1]
        if nxt.l > prev.h:
            out.append(FairValueGap(i, bars[i].t, Side.LONG, nxt.l, prev.h))
        elif nxt.h < prev.l:
            out.append(FairValueGap(i, bars[i].t, Side.SHORT, prev.l, nxt.h))
    return out


def nearest_unmitigated_block(
    blocks: list[OrderBlock], side: Side, price: float
) -> OrderBlock | None:
    """Closest still-valid order block on the given side, or None.

    For longs we want demand at or below price; for shorts, supply at or above
    it. "At" matters: a block price is currently trading inside is the one we
    most want, because that is the tap we are waiting for. Requiring the zone
    to sit strictly beyond price would exclude exactly the setup we trade.

    A block on the far wrong side of price has been left behind, and one that
    price closed through is invalidated -- both are dropped.
    """
    candidates = [
        b
        for b in blocks
        if b.side is side
        and not b.mitigated
        and (
            (side is Side.LONG and b.bottom <= price)
            or (side is Side.SHORT and b.top >= price)
        )
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda b: abs(price - b.mid))


def structural_stop_level(bars: list[Bar], upto: int, side: Side, lookback: int = 12) -> float:
    """The swing extreme a stop should sit beyond.

    Uses the raw low/high of the recent window rather than a confirmed swing,
    because for stop placement we want the actual worst price printed, not a
    fractal pattern.
    """
    window = bars[max(0, upto - lookback) : upto + 1]
    if not window:
        return bars[upto].c
    return min(b.l for b in window) if side is Side.LONG else max(b.h for b in window)
