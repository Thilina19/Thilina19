"""Loading and resampling OHLCV bars.

Three sources, in order of preference for backtesting:

  1. `load_csv` -- history exported from YOUR broker (MT5: Tools > History
     Center, or a right-click Export on the chart). This is the best source
     because it is the feed you will actually be filled on. OANDA data with
     IC Markets execution is a subtle mismatch that flatters the backtest.
  2. `load_json` -- bars cached from the TradingView MCP by cli.py.
  3. `resample` -- build 1h/4h series from a 15m series so one fetch covers
     all three timeframes and they are guaranteed mutually consistent.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .indicators import Bar

# Interval label -> seconds.
INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400,
    "1D": 86400, "1W": 604800,
}


def load_json(path: str | Path) -> list[Bar]:
    """Load bars from a JSON file: either a bare list or {"bars": [...]}."""
    data = json.loads(Path(path).read_text())
    raw = data["bars"] if isinstance(data, dict) else data
    bars = [Bar.from_dict(d) for d in raw]
    return _clean(bars)


def save_json(bars: list[Bar], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"count": len(bars),
         "bars": [{"t": b.t, "o": b.o, "h": b.h, "l": b.l, "c": b.c, "v": b.v}
                  for b in bars]},
        indent=None,
    ))


def load_csv(path: str | Path, *, tz_offset_hours: float = 0.0) -> list[Bar]:
    """Load a broker CSV export.

    Handles the common MT5/TradingView layouts: a header naming the columns, or
    headerless `date,time,open,high,low,close,volume`. Most MT5 servers stamp
    bars in broker time (often UTC+2/+3), so pass `tz_offset_hours` to shift
    them to UTC -- getting this wrong silently misplaces every session filter.
    """
    rows = list(csv.reader(Path(path).open(newline="", encoding="utf-8-sig")))
    if not rows:
        return []

    header: list[str] | None = None
    first = [c.strip().lower() for c in rows[0]]
    if any(k in first for k in ("open", "close", "time", "date", "<open>")):
        header = [c.strip().lower().strip("<>") for c in rows[0]]
        rows = rows[1:]

    def col(names: tuple[str, ...], default: int | None) -> int | None:
        if header:
            for n in names:
                if n in header:
                    return header.index(n)
        return default

    i_date = col(("date", "datetime", "time", "timestamp"), 0)
    i_time = col(("time",), 1) if header is None else (
        header.index("time") if ("time" in header and "date" in header) else None
    )
    i_o = col(("open",), 2)
    i_h = col(("high",), 3)
    i_l = col(("low",), 4)
    i_c = col(("close",), 5)
    i_v = col(("volume", "vol", "tickvol", "tickvolume"), 6)

    shift = int(tz_offset_hours * 3600)
    out: list[Bar] = []
    for r in rows:
        if not r or len(r) <= max(x for x in (i_o, i_h, i_l, i_c) if x is not None):
            continue
        try:
            stamp = r[i_date].strip()
            if i_time is not None and i_time < len(r) and i_time != i_date:
                stamp = f"{stamp} {r[i_time].strip()}"
            t = _parse_time(stamp) - shift
            out.append(Bar(
                t=t,
                o=float(r[i_o]), h=float(r[i_h]),
                l=float(r[i_l]), c=float(r[i_c]),
                v=float(r[i_v]) if (i_v is not None and i_v < len(r) and r[i_v]) else 0.0,
            ))
        except (ValueError, IndexError):
            continue  # skip malformed rows rather than abort the import
    return _clean(out)


_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
)


def _parse_time(s: str) -> int:
    s = s.strip().replace("T", " ").rstrip("Z")
    if s.isdigit() and len(s) >= 10:
        v = int(s)
        return v // 1000 if len(s) >= 13 else v
    for fmt in _TIME_FORMATS:
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp: {s!r}")


def _clean(bars: list[Bar]) -> list[Bar]:
    """Sort by time, drop duplicate timestamps and impossible bars."""
    seen: set[int] = set()
    out: list[Bar] = []
    for b in sorted(bars, key=lambda x: x.t):
        if b.t in seen:
            continue
        if not (b.h >= b.l and b.h >= max(b.o, b.c) and b.l <= min(b.o, b.c)):
            continue
        if b.h <= 0 or b.l <= 0:
            continue
        seen.add(b.t)
        out.append(b)
    return out


def resample(bars: list[Bar], target: str) -> list[Bar]:
    """Aggregate bars up to a longer interval.

    Buckets are aligned to the unix epoch, which puts 4h boundaries at 00/04/08
    /12/16/20 UTC. That matches how TradingView draws 4h candles on a 24h
    instrument like gold.
    """
    secs = INTERVAL_SECONDS.get(target)
    if secs is None:
        raise ValueError(f"unsupported interval {target!r}")
    if not bars:
        return []

    out: list[Bar] = []
    bucket: list[Bar] = []
    current = bars[0].t - (bars[0].t % secs)

    def flush() -> None:
        if not bucket:
            return
        out.append(Bar(
            t=current,
            o=bucket[0].o,
            h=max(b.h for b in bucket),
            l=min(b.l for b in bucket),
            c=bucket[-1].c,
            v=sum(b.v for b in bucket),
        ))

    for b in bars:
        start = b.t - (b.t % secs)
        if start != current:
            flush()
            bucket = []
            current = start
        bucket.append(b)
    flush()
    return out


def describe(bars: list[Bar], label: str = "") -> str:
    if not bars:
        return f"{label}: empty"
    a, z = bars[0].dt, bars[-1].dt
    span_days = (z - a).days
    return (
        f"{label}: {len(bars)} bars, {a:%Y-%m-%d %H:%M} -> {z:%Y-%m-%d %H:%M} UTC "
        f"({span_days}d), price {min(b.l for b in bars):.2f}-{max(b.h for b in bars):.2f}"
    )


def synthetic_bars(
    n: int, *, start_price: float = 2000.0, seed: int = 7,
    interval: str = "15m", trend_strength: float = 0.35,
) -> list[Bar]:
    """Deterministic pseudo-random bars for testing the plumbing.

    IMPORTANT: this is a test fixture, not a market model. Any backtest metric
    produced on it is meaningless. Its only job is to exercise the engine.

    That said, a plain Gaussian random walk is useless even for that: it has no
    fat tails, so displacement candles above 1.3 ATR occur in ~1.5% of bars and
    volume spikes in ~1.6%, and jointly essentially never. The order-block
    detector then finds nothing and the tests pass vacuously while proving
    nothing. So this generator reproduces three properties real gold has:

      1. Volatility clustering (GARCH-like): quiet periods and violent periods
         persist rather than alternating randomly.
      2. Jumps: occasional news-driven displacement candles several ATR wide.
      3. Volume correlated with volatility -- big candles print big volume,
         which is the whole premise of a volume-weighted order block.
    """
    import random

    rng = random.Random(seed)
    secs = INTERVAL_SECONDS[interval]
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    price = start_price
    out: list[Bar] = []

    base_vol = start_price * 0.0009
    vol = base_vol          # current volatility level
    drift = 0.0
    trend_len = 0

    for i in range(n):
        # --- regime: trends persist for a while, then flip
        if trend_len <= 0:
            drift = rng.gauss(0.0, 1.0) * trend_strength
            trend_len = rng.randint(40, 160)
        trend_len -= 1

        # --- volatility clustering: mean-reverting but persistent
        vol = 0.93 * vol + 0.07 * base_vol * abs(rng.gauss(1.0, 0.45))
        vol = max(base_vol * 0.25, min(vol, base_vol * 6.0))

        # --- jumps: roughly 1 in 70 bars gets a news-style displacement
        jump = 0.0
        is_jump = rng.random() < 0.014
        if is_jump:
            jump = rng.choice([-1, 1]) * rng.uniform(2.5, 6.0) * vol

        step = rng.gauss(drift * vol * 0.5, vol) + jump
        o = price
        c = max(1.0, o + step)

        # Wicks scale with volatility; jump candles close near their extreme,
        # which is what makes them read as displacement rather than indecision.
        wick_scale = 0.25 if is_jump else 0.85
        up_wick = abs(rng.gauss(0, 1)) * vol * wick_scale
        dn_wick = abs(rng.gauss(0, 1)) * vol * wick_scale

        # Volume tracks the size of the move -- this is the correlation the
        # order-block logic depends on. The dispersion is calibrated against
        # 400 real OANDA:XAUUSD 1h bars, which show a volume/20-bar-average
        # ratio of mean 1.05, p90 1.90, max 6.5. A tighter spread (an early
        # version used sigma 0.28, giving p90 of only 1.24) never triggers the
        # volume filter and makes the order-block tests vacuous.
        move = abs(c - o) / max(vol, 1e-9)
        v = 700.0 * (0.4 + 1.1 * move) * math.exp(rng.gauss(0.0, 0.45)) + 50.0
        if is_jump:
            v *= rng.uniform(2.0, 4.5)

        out.append(Bar(
            t=t0 + i * secs, o=o, c=c,
            h=max(o, c) + up_wick, l=min(o, c) - dn_wick,
            v=v,
        ))
        price = c
    return out
