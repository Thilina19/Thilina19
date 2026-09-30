"""Indicator math. Pure standard library -- no numpy, no pandas.

Every function takes a list of Bar and returns a list the same length as the
input, with None in positions where there is not enough history yet. Keeping
the lengths aligned means callers can always index by bar position without
off-by-one bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Bar:
    """One OHLCV candle. `t` is a unix timestamp in seconds, UTC."""

    t: int
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.t, tz=timezone.utc)

    @property
    def range(self) -> float:
        return self.h - self.l

    @property
    def body(self) -> float:
        return abs(self.c - self.o)

    @property
    def is_bull(self) -> bool:
        return self.c >= self.o

    @classmethod
    def from_dict(cls, d: dict) -> "Bar":
        return cls(
            t=int(d["t"]),
            o=float(d["o"]),
            h=float(d["h"]),
            l=float(d["l"]),
            c=float(d["c"]),
            v=float(d.get("v", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class HABar:
    """A Heikin Ashi candle derived from real bars.

    Heikin Ashi smooths noise by averaging, which is exactly why we use it as
    a *trigger* and never as a price source: HA open/close are synthetic and
    are not tradeable prices. Stops and targets always come from real OHLC.
    """

    t: int
    o: float
    h: float
    l: float
    c: float

    @property
    def is_bull(self) -> bool:
        return self.c >= self.o

    @property
    def body(self) -> float:
        return abs(self.c - self.o)

    @property
    def wick_ratio(self) -> float:
        """Fraction of the candle's range that is wick rather than body.

        A high ratio means indecision. 1.0 is a pure doji.
        """
        rng = self.h - self.l
        if rng <= 0:
            return 1.0
        return 1.0 - (self.body / rng)


def heikin_ashi(bars: list[Bar]) -> list[HABar]:
    """Standard Heikin Ashi transform.

    HA close = mean(o,h,l,c); HA open = mean(prev HA open, prev HA close).
    The first candle seeds HA open from the real open/close midpoint.
    """
    out: list[HABar] = []
    for i, b in enumerate(bars):
        ha_c = (b.o + b.h + b.l + b.c) / 4.0
        if i == 0:
            ha_o = (b.o + b.c) / 2.0
        else:
            prev = out[-1]
            ha_o = (prev.o + prev.c) / 2.0
        out.append(
            HABar(
                t=b.t,
                o=ha_o,
                c=ha_c,
                h=max(b.h, ha_o, ha_c),
                l=min(b.l, ha_o, ha_c),
            )
        )
    return out


def sma(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    """EMA seeded with an SMA of the first `period` values.

    Seeding with an SMA rather than the first value avoids a long warm-up
    distortion that would otherwise skew early backtest bars.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def true_range(prev_close: float, bar: Bar) -> float:
    return max(
        bar.h - bar.l,
        abs(bar.h - prev_close),
        abs(bar.l - prev_close),
    )


def atr(bars: list[Bar], period: int = 14) -> list[float | None]:
    """Wilder's ATR. Index 0 is always None since TR needs a previous close."""
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period + 1:
        return out
    trs: list[float] = [0.0]
    for i in range(1, len(bars)):
        trs.append(true_range(bars[i - 1].c, bars[i]))

    seed = sum(trs[1 : period + 1]) / period
    out[period] = seed
    prev = seed
    for i in range(period + 1, len(bars)):
        # Wilder smoothing
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


def rsi(bars: list[Bar], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period + 1:
        return out
    gains: list[float] = [0.0]
    losses: list[float] = [0.0]
    for i in range(1, len(bars)):
        d = bars[i].c - bars[i - 1].c
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))

    avg_g = sum(gains[1 : period + 1]) / period
    avg_l = sum(losses[1 : period + 1]) / period

    def rsi_from(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = rsi_from(avg_g, avg_l)
    for i in range(period + 1, len(bars)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        out[i] = rsi_from(avg_g, avg_l)
    return out


def volume_ratio(bars: list[Bar], lookback: int = 20) -> list[float | None]:
    """Each bar's volume divided by the average of the `lookback` bars before it.

    Deliberately excludes the current bar from its own average, so a single
    huge bar does not dilute the spike it is supposed to signal.
    """
    out: list[float | None] = [None] * len(bars)
    for i in range(lookback, len(bars)):
        window = [b.v for b in bars[i - lookback : i]]
        avg = sum(window) / lookback
        if avg > 0:
            out[i] = bars[i].v / avg
    return out


@dataclass(frozen=True)
class Swing:
    """A confirmed fractal swing point."""

    index: int
    t: int
    price: float
    is_high: bool


def find_swings(bars: list[Bar], lookback: int = 2) -> list[Swing]:
    """Fractal swing highs/lows: a bar whose high (low) exceeds `lookback`
    bars on both sides.

    Note these are only confirmed `lookback` bars after the fact. Callers
    doing bar-by-bar simulation must respect that delay or they leak future
    information -- structure.py handles this.
    """
    out: list[Swing] = []
    n = len(bars)
    for i in range(lookback, n - lookback):
        left = bars[i - lookback : i]
        right = bars[i + 1 : i + 1 + lookback]
        hi, lo = bars[i].h, bars[i].l
        if all(b.h < hi for b in left) and all(b.h < hi for b in right):
            out.append(Swing(i, bars[i].t, hi, True))
        if all(b.l > lo for b in left) and all(b.l > lo for b in right):
            out.append(Swing(i, bars[i].t, lo, False))
    out.sort(key=lambda s: s.index)
    return out


def closes(bars: list[Bar]) -> list[float]:
    return [b.c for b in bars]
