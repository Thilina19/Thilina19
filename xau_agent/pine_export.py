"""Export journalled signals to a Pine Script overlay.

Two different things can be drawn on a TradingView chart, and they answer
different questions:

  pine/xau_agent_strategy.pine  -- re-implements the logic in Pine, so it shows
      what the rules WOULD have done across all history, with TradingView's own
      Strategy Tester statistics. Independent of the journal.

  this module                   -- draws the signals the Python engine ACTUALLY
      produced and you actually logged, at their real timestamps, with their
      real SL/TP and outcome. This is your trade history on the chart.

Use both. When they disagree, the journal is the truth and the disagreement is
worth understanding -- the Pine version runs on two timeframes rather than
three, so some divergence is expected.

Drawing objects are capped by TradingView at 500 of each type. Each trade costs
2 boxes, 1 line and 1 label, so roughly 160 trades fit. Newest are kept.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .journal import JournalEntry, Outcome

# Each trade draws 2 boxes; TradingView's ceiling is 500 per object type.
MAX_TRADES = 160


def _pine_float_array(name: str, values: list[float]) -> str:
    if not values:
        return f"{name} = array.new<float>(0)"
    inner = ", ".join(f"{v:.5f}" for v in values)
    return f"{name} = array.from({inner})"


def _pine_int_array(name: str, values: list[int]) -> str:
    if not values:
        return f"{name} = array.new<int>(0)"
    inner = ", ".join(str(v) for v in values)
    return f"{name} = array.from({inner})"


def _pine_string_array(name: str, values: list[str]) -> str:
    if not values:
        return f"{name} = array.new<string>(0)"
    inner = ", ".join('"' + v.replace('"', "'") + '"' for v in values)
    return f"{name} = array.from({inner})"


def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def build_pine(
    trades: list[JournalEntry],
    *,
    title: str = "XAU Agent -- Journalled Signals",
    symbol_note: str = "CAPITALCOM:XAUUSD",
) -> str:
    """Render a Pine v6 indicator that draws these trades on the chart."""
    kept = sorted(trades, key=lambda t: t.signal_time)[-MAX_TRADES:]

    t_open = [_ms(t.signal_time) for t in kept]
    t_close = [_ms(t.exit_time) if t.exit_time else _ms(t.signal_time) for t in kept]
    side = [1 if t.side == "long" else -1 for t in kept]
    entry = [t.entry for t in kept]
    stop = [t.stop for t in kept]
    tp1 = [t.tp1 for t in kept]
    tp2 = [t.tp2 for t in kept]
    exitpx = [t.exit_price if t.exit_price is not None else t.entry for t in kept]
    score = [int(t.score) for t in kept]
    rmult = [t.r_multiple for t in kept]
    pnl = [t.pnl for t in kept]
    # 1 win, -1 loss, 0 breakeven, 2 still open
    outcome = [
        1 if t.outcome == Outcome.WIN.value
        else -1 if t.outcome == Outcome.LOSS.value
        else 2 if t.outcome == Outcome.OPEN.value
        else 0
        for t in kept
    ]
    mode = [t.mode for t in kept]

    wins = sum(1 for o in outcome if o == 1)
    losses = sum(1 for o in outcome if o == -1)
    decided = wins + losses
    wr = (wins / decided * 100.0) if decided else 0.0
    total_r = sum(rmult)
    total_pnl = sum(pnl)
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    arrays = "\n".join([
        _pine_int_array("sigTime", t_open),
        _pine_int_array("endTime", t_close),
        _pine_int_array("sigSide", side),
        _pine_float_array("sigEntry", entry),
        _pine_float_array("sigStop", stop),
        _pine_float_array("sigTP1", tp1),
        _pine_float_array("sigTP2", tp2),
        _pine_float_array("sigExit", exitpx),
        _pine_int_array("sigScore", score),
        _pine_float_array("sigR", rmult),
        _pine_float_array("sigPnl", pnl),
        _pine_int_array("sigOutcome", outcome),
        _pine_string_array("sigMode", mode),
    ])

    return f'''//@version=6
// =============================================================================
// {title}
// =============================================================================
// GENERATED FILE -- do not edit by hand.
//   Regenerate with:  python3 -m xau_agent.cli export-pine
//   Generated:        {generated}
//   Symbol:           {symbol_note}
//
// These are the signals the Python engine actually produced and that were
// logged in the journal, drawn at their real timestamps with their real stop,
// targets and outcome. Unlike the strategy script, nothing here is recomputed
// -- it is your recorded history.
//
// Summary of what is drawn:
//   trades      {len(kept)}
//   win rate    {wr:.1f}%  ({wins}W / {losses}L{f" / {len(kept) - decided} open or breakeven" if len(kept) - decided else ""})
//   total       {total_r:+.2f}R   ${total_pnl:+,.2f}
//
// HOW TO USE
//   1. Open {symbol_note} on any timeframe.
//   2. Pine Editor -> paste -> Save -> Add to chart.
//   Boxes are anchored to absolute time, so the drawings land in the right
//   place on any timeframe.
//
// A green box is the reward leg (entry to TP2), a red box the risk leg (entry
// to stop). The marker at the entry shows the outcome: a filled triangle for a
// win, hollow for a loss.
// =============================================================================

indicator("{title}", overlay = true,
     max_boxes_count = 500, max_labels_count = 500, max_lines_count = 500)

showWins    = input.bool(true,  "Show winners")
showLosses  = input.bool(true,  "Show losers")
showOpen    = input.bool(true,  "Show open / breakeven")
showRisk    = input.bool(true,  "Shade the risk leg (entry to stop)")
showReward  = input.bool(true,  "Shade the reward leg (entry to TP2)")
showLabels  = input.bool(true,  "Show detail labels")
minScoreF   = input.int(0,      "Only show score >=", minval = 0, maxval = 100)
modeFilter  = input.string("all", "Mode", options = ["all", "paper", "live", "backtest"])

cWin  = input.color(color.new(#26a69a, 0),  "Win colour")
cLoss = input.color(color.new(#ef5350, 0),  "Loss colour")
cOpen = input.color(color.new(#b0bec5, 0),  "Open colour")

{arrays}

// Everything is drawn once, on the last bar, using absolute timestamps
// (xloc.bar_time) so it is timeframe independent.
if barstate.islast and array.size(sigTime) > 0
    for i = 0 to array.size(sigTime) - 1
        t0   = array.get(sigTime, i)
        t1   = array.get(endTime, i)
        sd   = array.get(sigSide, i)
        en   = array.get(sigEntry, i)
        sl   = array.get(sigStop, i)
        p1   = array.get(sigTP1, i)
        p2   = array.get(sigTP2, i)
        xp   = array.get(sigExit, i)
        sc   = array.get(sigScore, i)
        rm   = array.get(sigR, i)
        pl   = array.get(sigPnl, i)
        oc   = array.get(sigOutcome, i)
        md   = array.get(sigMode, i)

        passMode  = modeFilter == "all" or modeFilter == md
        passScore = sc >= minScoreF
        passKind  = (oc ==  1 and showWins) or (oc == -1 and showLosses) or
                    ((oc == 0 or oc == 2) and showOpen)

        if passMode and passScore and passKind
            col   = oc == 1 ? cWin : oc == -1 ? cLoss : cOpen
            // Give a still-open or zero-length trade a visible width.
            rightT = t1 > t0 ? t1 : t0 + 6 * 60 * 60 * 1000

            // box.new takes (top, bottom) in that order. On a short the stop is
            // ABOVE the entry and TP2 below, so order each pair by max/min
            // rather than by role or shorts draw inverted.
            if showRisk
                box.new(t0, math.max(en, sl), rightT, math.min(en, sl),
                     xloc = xloc.bar_time, border_color = color.new(cLoss, 55),
                     bgcolor = color.new(cLoss, 86), border_width = 1)
            if showReward
                box.new(t0, math.max(en, p2), rightT, math.min(en, p2),
                     xloc = xloc.bar_time, border_color = color.new(cWin, 55),
                     bgcolor = color.new(cWin, 86), border_width = 1)

            // TP1 partial level.
            line.new(t0, p1, rightT, p1, xloc = xloc.bar_time,
                 color = color.new(cWin, 30), style = line.style_dotted)
            // Entry level.
            line.new(t0, en, rightT, en, xloc = xloc.bar_time,
                 color = col, width = 2)
            // Where it actually closed.
            if oc != 2
                label.new(rightT, xp, "", xloc = xloc.bar_time,
                     style = label.style_xcross, color = col, size = size.tiny)

            // Direction / outcome marker at the entry.
            label.new(t0, en,
                 (sd == 1 ? "L" : "S") + " " + str.tostring(sc),
                 xloc  = xloc.bar_time,
                 style = sd == 1 ? label.style_triangleup : label.style_triangledown,
                 color = col, size = size.tiny,
                 tooltip = (sd == 1 ? "LONG" : "SHORT") +
                     "  score " + str.tostring(sc) + "/100" +
                     "\\nentry " + str.tostring(en, format.mintick) +
                     "\\nSL "    + str.tostring(sl, format.mintick) +
                     "\\nTP1 "   + str.tostring(p1, format.mintick) +
                     "\\nTP2 "   + str.tostring(p2, format.mintick) +
                     "\\nexit "  + str.tostring(xp, format.mintick) +
                     "\\nresult " + str.tostring(rm, "#.##") + "R" +
                     "  $" + str.tostring(pl, "#.##") +
                     "\\nmode " + md)

            if showLabels
                label.new(t0, sd == 1 ? sl : p2,
                     str.tostring(rm, "#.##") + "R",
                     xloc = xloc.bar_time,
                     style = sd == 1 ? label.style_label_up : label.style_label_down,
                     color = color.new(col, 25), textcolor = color.white,
                     size = size.tiny)

// Summary panel -- figures computed at export time from the journal.
if barstate.islast
    var table tb = table.new(position.bottom_right, 2, 4,
         border_width = 1, frame_width = 1, frame_color = color.gray)
    table.cell(tb, 0, 0, "Journalled signals", text_color = color.white,
         bgcolor = color.new(color.blue, 30), text_size = size.small)
    table.cell(tb, 1, 0, "{generated}", text_color = color.white,
         bgcolor = color.new(color.blue, 30), text_size = size.small)
    table.cell(tb, 0, 1, "trades",    text_size = size.small)
    table.cell(tb, 1, 1, "{len(kept)}", text_size = size.small)
    table.cell(tb, 0, 2, "win rate",  text_size = size.small)
    table.cell(tb, 1, 2, "{wr:.1f}%  ({wins}W/{losses}L)", text_size = size.small)
    table.cell(tb, 0, 3, "total",     text_size = size.small)
    table.cell(tb, 1, 3, "{total_r:+.2f}R   ${total_pnl:+,.2f}", text_size = size.small)
'''


def write_pine(trades: list[JournalEntry], path: str | Path,
               **kw: object) -> tuple[Path, int]:
    """Write the generated indicator. Returns (path, trades drawn)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    src = build_pine(trades, **kw)  # type: ignore[arg-type]
    p.write_text(src)
    return p, min(len(trades), MAX_TRADES)
