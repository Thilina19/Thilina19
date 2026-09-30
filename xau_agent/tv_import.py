"""Import TradingView Strategy Tester trade exports into the journal.

This closes the loop that makes "the agent improves daily" mean something. The
review in review.py can only learn from trades it can see, and until now the
only way in was logging each trade by hand. TradingView's Strategy Tester will
export its full List of Trades as CSV; this reads that file.

Export path: Strategy Tester -> List of Trades -> the download icon -> CSV.

The export pairs two rows per trade, an entry and an exit, joined by trade
number. Columns have been renamed across TradingView versions, so lookup is by
fuzzy header match rather than fixed position.

On R-multiples: the export does not carry the stop, so R cannot be derived from
prices. But a strategy that sizes every position to a fixed risk budget makes
P&L proportional to R by construction, so R = P&L / risk_per_trade. That is
exact for fixed-risk sizing and wrong for anything else, which is why
`risk_per_trade` is explicit rather than guessed.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .journal import JournalEntry, Mode, Outcome

# Header aliases seen across TradingView versions, lowercased.
_FIELDS = {
    "trade":  ("trade #", "trade#", "trade"),
    "type":   ("type",),
    "when":   ("date/time", "datetime", "date", "time"),
    "signal": ("signal",),
    "price":  ("price usd", "price", "price $"),
    "qty":    ("position size (qty)", "contracts", "quantity", "position size"),
    "pnl":    ("net p&l usd", "profit usd", "net profit usd", "p&l usd"),
    "runup":  ("run-up usd", "runup usd", "run up usd"),
    "draw":   ("drawdown usd", "drawdown $"),
}

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S",
)


def _index(header: list[str]) -> dict[str, int]:
    norm = [h.strip().strip('"').lower() for h in header]
    out: dict[str, int] = {}
    for key, aliases in _FIELDS.items():
        for i, h in enumerate(norm):
            if h in aliases or any(h.startswith(a) for a in aliases):
                out[key] = i
                break
    return out


def _num(s: str) -> float:
    """Parse a TradingView numeric cell: strips currency, commas, spaces, %."""
    if s is None:
        return 0.0
    t = str(s).strip().strip('"').replace(",", "").replace("$", "").replace("%", "")
    t = t.replace("−", "-").replace("–", "-")  # unicode minus/en-dash
    if t in ("", "-", "n/a", "N/A"):
        return 0.0
    # Parentheses denote a negative in some locales: (123.45)
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        return float(t)
    except ValueError:
        return 0.0


def _when(s: str) -> datetime:
    t = str(s).strip().strip('"').replace("T", " ")
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognised date/time: {s!r}")


def parse_tv_export(
    path: str | Path,
    *,
    risk_per_trade: float = 250.0,
    mode: str = Mode.PAPER.value,
    session: str = "unknown",
) -> list[JournalEntry]:
    """Read a TradingView List of Trades CSV into JournalEntry rows."""
    if risk_per_trade <= 0:
        raise ValueError("risk_per_trade must be positive to derive R-multiples")

    rows = list(csv.reader(Path(path).open(newline="", encoding="utf-8-sig")))
    if not rows:
        return []
    idx = _index(rows[0])
    missing = {"trade", "type", "when", "price"} - set(idx)
    if missing:
        raise ValueError(
            f"{path}: not a TradingView List of Trades export -- missing "
            f"column(s) {sorted(missing)}. Export via Strategy Tester -> "
            f"List of Trades -> download icon -> CSV."
        )

    def cell(row: list[str], key: str) -> str:
        i = idx.get(key, -1)
        return row[i] if 0 <= i < len(row) else ""

    grouped: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []
    for row in rows[1:]:
        if not row or not any(c.strip() for c in row):
            continue
        tid = cell(row, "trade").strip().strip('"')
        kind = cell(row, "type").strip().strip('"').lower()
        if not tid or not kind:
            continue
        slot = "entry" if "entry" in kind else "exit" if "exit" in kind else None
        if slot is None:
            continue
        if tid not in grouped:
            grouped[tid] = {}
            order.append(tid)
        grouped[tid][slot] = row
        if slot == "entry":
            grouped[tid]["_side"] = ["short" if "short" in kind else "long"]

    out: list[JournalEntry] = []
    for tid in order:
        g = grouped[tid]
        if "entry" not in g or "exit" not in g:
            continue                      # still-open trade: no exit row
        e, x = g["entry"], g["exit"]
        side = g.get("_side", ["long"])[0]
        entry_px = _num(cell(e, "price"))
        exit_px = _num(cell(x, "price"))
        if entry_px <= 0:
            continue

        # P&L is reported on the exit row in every version seen.
        pnl = _num(cell(x, "pnl")) or _num(cell(e, "pnl"))
        r = pnl / risk_per_trade
        runup = _num(cell(x, "runup"))
        draw = _num(cell(x, "draw"))

        outcome = (Outcome.WIN.value if pnl > 0.5 else
                   Outcome.LOSS.value if pnl < -0.5 else
                   Outcome.BREAKEVEN.value)

        out.append(JournalEntry(
            signal_time=_when(cell(e, "when")),
            side=side,
            entry=entry_px,
            # The export carries no stop or target. Recording the entry price
            # in those fields would be a fabricated level, so they are left
            # equal to entry and the honest signal is r_multiple.
            stop=entry_px, tp1=entry_px, tp2=entry_px,
            score=0, session=session, mode=mode,
            lots=_num(cell(e, "qty")),
            exit_price=exit_px,
            exit_time=_when(cell(x, "when")),
            pnl=pnl, r_multiple=r, outcome=outcome,
            mae_r=abs(draw) / risk_per_trade,
            mfe_r=abs(runup) / risk_per_trade,
            notes=f"tradingview import, trade #{tid}",
        ))
    return out
