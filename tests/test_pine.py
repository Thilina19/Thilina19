"""Tests for the Pine Script exporter and a structural lint of the Pine files.

Pine cannot be executed here, so these tests cannot prove the scripts compile.
What they can do is catch the mistakes that are cheap to make and expensive to
find on a chart: unbalanced brackets, a reference used before it is defined,
inverted drawing coordinates, and the injection hazard of pushing journal text
straight into generated source.
"""

from __future__ import annotations

import re
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from xau_agent.journal import JournalEntry, Mode, Outcome
from xau_agent.pine_export import MAX_TRADES, build_pine

PINE_DIR = Path(__file__).resolve().parent.parent / "pine"
STRATEGY = PINE_DIR / "xau_agent_strategy.pine"


def mk_trade(
    *, side: str = "long", entry: float = 4200.0, stop: float = 4188.0,
    tp1: float = 4212.0, tp2: float = 4226.4, exit_price: float | None = 4226.4,
    pnl: float = 500.0, r: float = 2.2, outcome: Outcome = Outcome.WIN,
    score: int = 75, day: int = 1, mode: str = Mode.PAPER.value, notes: str = "",
) -> JournalEntry:
    t = datetime(2026, 9, day, 10, 0, tzinfo=timezone.utc)
    return JournalEntry(
        signal_time=t, side=side, entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        score=score, session="london", mode=mode, lots=0.2,
        exit_price=exit_price,
        exit_time=(t + timedelta(hours=4)) if exit_price is not None else None,
        pnl=pnl, r_multiple=r, outcome=outcome.value, notes=notes,
    )


def brackets_balanced(src: str) -> tuple[bool, str]:
    """Check (), [] balance outside of comments and string literals."""
    depth = {"(": 0, "[": 0}
    close_to_open = {")": "(", "]": "["}
    for lineno, raw in enumerate(src.splitlines(), 1):
        i = 0
        in_str = False
        while i < len(raw):
            ch = raw[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "/" and i + 1 < len(raw) and raw[i + 1] == "/":
                    break  # rest of the line is a comment
                elif ch in depth:
                    depth[ch] += 1
                elif ch in close_to_open:
                    o = close_to_open[ch]
                    depth[o] -= 1
                    if depth[o] < 0:
                        return False, f"line {lineno}: unmatched '{ch}'"
            i += 1
    for o, n in depth.items():
        if n != 0:
            return False, f"unbalanced '{o}': {n} left open"
    return True, "balanced"


#: Pine output calls that satisfy the CE10244 "script must produce output" check.
PINE_OUTPUT_CALLS = (
    "plot(", "plotshape(", "plotchar(", "plotcandle(", "plotarrow(", "plotbar(",
    "barcolor(", "bgcolor(", "hline(", "fill(",
)


def global_output_calls(src: str) -> list[tuple[int, str]]:
    """Output calls at column 0, i.e. at global scope.

    Pine raises CE10244 unless a script has at least one of these at GLOBAL
    scope. Calls nested inside `if` are local and do not count, and neither
    does a `var table t = table.new(...)` initialiser -- which is exactly how
    both of these scripts failed the first time.
    """
    found = []
    for n, line in enumerate(strip_comments(src).splitlines(), 1):
        if line.startswith((" ", "\t")):
            continue
        if any(line.startswith(c) for c in PINE_OUTPUT_CALLS):
            found.append((n, line))
    return found


def indented_output_calls(src: str) -> list[int]:
    """Output calls inside a local scope, which Pine does not allow at all."""
    bad = []
    for n, line in enumerate(strip_comments(src).splitlines(), 1):
        if not line.startswith((" ", "\t")):
            continue
        s = line.lstrip()
        if any(s.startswith(c) for c in PINE_OUTPUT_CALLS):
            bad.append(n)
    return bad


def typed_function_params(src: str) -> list[str]:
    """User function definitions with type annotations on parameters.

    Pine user-function signatures are untyped; annotating them is an error.
    """
    bad = []
    for line in strip_comments(src).splitlines():
        m = re.match(r"^(\w+)\(([^)]*)\)\s*=>", line)
        if m and re.search(
            r"\b(int|float|bool|string|color|series|simple)\s+\w+", m.group(2)
        ):
            bad.append(m.group(1))
    return bad


def comments_inside_calls(src: str) -> list[tuple[int, str]]:
    """Comment lines sitting inside an unclosed multi-line function call.

    Pine does not allow this. The comment terminates the call, the remaining
    arguments are orphaned, and the script mis-parses. The reported error is
    badly misleading -- the compiler blames line 1 with CE10244 ("a strategy
    must contain at least one ... plot") even though the script is full of
    plots and orders.

    This cost a full debugging round trip, so it is checked here.
    """
    bad: list[tuple[int, str]] = []
    prev_code: str | None = None
    for n, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("//"):
            # Indented comment directly continuing a line that left a call open.
            if (prev_code is not None
                    and prev_code.rstrip().endswith((",", "("))
                    and line.startswith((" ", "\t"))):
                bad.append((n, s[:60]))
            continue
        prev_code = s
    return bad


def reused_exit_ids(src: str) -> dict[str, set[str]]:
    """strategy.exit ids used with more than one from_entry."""
    ids: dict[str, set[str]] = {}
    for line in strip_comments(src).splitlines():
        m = re.search(
            r'strategy\.exit\(\s*"([^"]+)".*?from_entry\s*=\s*"([^"]+)"', line
        )
        if m:
            ids.setdefault(m.group(1), set()).add(m.group(2))
    return {k: v for k, v in ids.items() if len(v) > 1}


def strip_comments(src: str) -> str:
    out = []
    for raw in src.splitlines():
        in_str = False
        cut = len(raw)
        i = 0
        while i < len(raw):
            c = raw[i]
            if in_str:
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "/" and i + 1 < len(raw) and raw[i + 1] == "/":
                    cut = i
                    break
            i += 1
        out.append(raw[:cut])
    return "\n".join(out)


class TestStrategyPineFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = STRATEGY.read_text()
        cls.code = strip_comments(cls.src)

    def test_file_exists_and_declares_v6(self) -> None:
        self.assertTrue(self.src.startswith("//@version=6"),
                        "Pine files must open with the version tag")

    def test_brackets_balanced(self) -> None:
        ok, msg = brackets_balanced(self.src)
        self.assertTrue(ok, msg)

    def test_declares_a_strategy(self) -> None:
        self.assertIn("strategy(", self.code)

    def test_charges_slippage(self) -> None:
        """A tester with no costs reports fiction."""
        self.assertRegex(self.code, r"slippage\s*=\s*\d+")

    def test_uses_the_non_repainting_security_idiom(self) -> None:
        """f()[1] with lookahead_on is what makes history match realtime."""
        self.assertIn("f_bias()[1]", self.code)
        self.assertIn("barmerge.lookahead_on", self.code)

    def test_volume_filter_is_on_the_displacement_candle(self) -> None:
        """Regression guard mirroring the Python fix.

        The volume test must gate the displacement candle. If it drifts onto the
        block candle the strategy stops producing signals entirely.
        """
        m = re.search(r"isDisp\s*=(.+?)\n\S", self.code, re.S)
        self.assertIsNotNone(m, "could not locate the isDisp definition")
        self.assertIn("volRat", m.group(1),
                      "the displacement test must include the volume ratio")

    def test_zone_invalidated_by_close_not_touch(self) -> None:
        """Regression guard: a tap is the trigger, not invalidation."""
        self.assertRegex(self.code, r"close\s*<\s*dBot")
        self.assertRegex(self.code, r"close\s*>\s*sTop")

    def test_drawing_boxes_order_coordinates_safely(self) -> None:
        """box.new(top, bottom) must be ordered by max/min, not by role.

        On a short the stop is above the entry, so passing them by role draws
        every short's boxes inverted.
        """
        for call in re.findall(r"box\.new\((.*?)\)\n", self.code, re.S):
            if "math.max" not in call and "math.min" not in call:
                # Zone boxes use dTop/dBot and sTop/sBot, which are already
                # ordered by construction.
                self.assertTrue(
                    ("dTop" in call and "dBot" in call)
                    or ("sTop" in call and "sBot" in call),
                    f"box.new with unordered coordinates: {call[:80]}",
                )

    def test_enforces_the_hard_risk_rules(self) -> None:
        for token in ("dailyMaxLoss", "maxTrades", "maxConsecL", "cooloffMin",
                      "minRR", "dailyTarget"):
            self.assertIn(token, self.code, f"{token} is missing from the risk gate")

    def test_risk_gate_requires_flat_and_confirmed(self) -> None:
        m = re.search(r"riskOK\s*=(.+?)\n\n", self.code, re.S)
        self.assertIsNotNone(m)
        gate = m.group(1)
        self.assertIn("flat", gate, "must refuse a second concurrent position")
        self.assertIn("barstate.isconfirmed", gate,
                      "entries must be evaluated on bar close, not intrabar")

    def test_has_a_global_output_call(self) -> None:
        """Regression guard for Pine error CE10244.

        The first version of this file drew boxes, labels, a table and placed
        orders -- and still would not compile, because every one of those calls
        was inside an `if` (local scope) and a `var table t = table.new(...)`
        initialiser does not satisfy the check either. A script needs at least
        one unconditional output call at column 0.
        """
        found = global_output_calls(self.src)
        self.assertTrue(
            found,
            "no output call at global scope -- Pine will reject this with "
            "CE10244 regardless of how much the script draws",
        )

    def test_no_comments_inside_multiline_calls(self) -> None:
        """Regression guard for the bug that actually caused CE10244.

        Six comment lines were sitting inside the strategy() argument list,
        explaining slippage and fill assumptions. Pine ends the call at the
        first comment, so strategy() was truncated, the rest of its arguments
        were orphaned, the script mis-parsed, and the compiler reported "no
        plot" against line 1 -- while ten plots sat at global scope.
        """
        bad = comments_inside_calls(self.src)
        self.assertEqual(
            bad, [],
            "comment(s) inside a multi-line call will silently break parsing: "
            + "; ".join(f"line {n}: {t}" for n, t in bad),
        )

    def test_strategy_declaration_is_a_single_logical_call(self) -> None:
        """The strategy() call must not be interrupted by anything."""
        m = re.search(r"^strategy\(.*$", self.src, re.M)
        self.assertIsNotNone(m, "strategy() must start at column 0")
        self.assertTrue(
            m.group(0).rstrip().endswith(")"),
            "strategy() should be written on one line -- splitting it invites "
            "the comment-in-call parsing failure",
        )

    def test_no_output_calls_in_local_scope(self) -> None:
        bad = indented_output_calls(self.src)
        self.assertEqual(bad, [],
                         f"plot-family calls cannot be conditional; lines {bad}")

    def test_no_typed_function_parameters(self) -> None:
        bad = typed_function_params(self.src)
        self.assertEqual(bad, [],
                         f"Pine function params must be untyped; offenders: {bad}")

    def test_exit_ids_not_reused_across_directions(self) -> None:
        dupes = reused_exit_ids(self.src)
        self.assertEqual(dupes, {},
                         f"an exit id maps to two entries, which is ambiguous: {dupes}")

    def test_plots_the_working_stop_not_the_original(self) -> None:
        """After TP1 the stop moves to breakeven; the chart must show that."""
        self.assertIn("activeSL", self.code)
        self.assertRegex(self.code, r"plot\(inPos \? activeSL")

    def test_volatility_gate_is_relative_not_absolute(self) -> None:
        """Regression guard: the absolute ATR-percent band produced zero trades
        on 5m because gold's ATR is ~0.11% of price there against a 0.12%
        floor tuned for 15m. The gate must compare ATR to its own average."""
        self.assertNotIn("minAtrPct", self.code)
        self.assertNotIn("maxAtrPct", self.code)
        self.assertIn("atrAvg", self.code)
        self.assertRegex(self.code, r"volMult\s*>=\s*minVolMult")

    def test_reports_why_no_trade_fired(self) -> None:
        """An empty tester must say which gate is binding."""
        for token in ("nBias", "nZone", "nTap", "nHA", "nVol", "nScore", "nRisk"):
            self.assertIn(token, self.code,
                          f"diagnostic counter {token} is missing")
        self.assertIn("WHY NO TRADE", self.src)

    def test_exposes_alerts(self) -> None:
        self.assertIn("alertcondition(", self.code)
        self.assertIn("alert(", self.code)

    def test_heikin_ashi_used_only_as_trigger(self) -> None:
        """Levels must come from real OHLC. HA open/close are synthetic."""
        m = re.search(r"longSL\s*=(.+)", self.code)
        self.assertIsNotNone(m)
        self.assertNotIn("ha", m.group(1).lower().replace("math", ""),
                         "stop levels must not be derived from Heikin Ashi")


class TestJournalPineExport(unittest.TestCase):
    def test_empty_journal_still_renders(self) -> None:
        src = build_pine([])
        self.assertIn("//@version=6", src)
        self.assertIn("array.new<int>(0)", src)

    def test_arrays_match_trade_count(self) -> None:
        trades = [mk_trade(day=d) for d in range(1, 6)]
        src = build_pine(trades)
        m = re.search(r"sigEntry = array\.from\(([^)]*)\)", src)
        self.assertIsNotNone(m)
        self.assertEqual(len(m.group(1).split(",")), 5)

    def test_brackets_balanced(self) -> None:
        src = build_pine([mk_trade(day=d) for d in range(1, 4)])
        ok, msg = brackets_balanced(src)
        self.assertTrue(ok, msg)

    def test_outcome_encoding(self) -> None:
        trades = [
            mk_trade(day=1, outcome=Outcome.WIN, pnl=500, r=2.2),
            mk_trade(day=2, outcome=Outcome.LOSS, pnl=-250, r=-1.0),
            mk_trade(day=3, outcome=Outcome.BREAKEVEN, pnl=0.0, r=0.0),
            mk_trade(day=4, outcome=Outcome.OPEN, exit_price=None, pnl=0.0, r=0.0),
        ]
        src = build_pine(trades)
        m = re.search(r"sigOutcome = array\.from\(([^)]*)\)", src)
        self.assertIsNotNone(m)
        self.assertEqual([v.strip() for v in m.group(1).split(",")],
                         ["1", "-1", "0", "2"])

    def test_win_rate_excludes_breakeven_and_open(self) -> None:
        """Two wins, one loss, one breakeven, one open => 66.7%, not 40%."""
        trades = [
            mk_trade(day=1, outcome=Outcome.WIN),
            mk_trade(day=2, outcome=Outcome.WIN),
            mk_trade(day=3, outcome=Outcome.LOSS, pnl=-250, r=-1.0),
            mk_trade(day=4, outcome=Outcome.BREAKEVEN, pnl=0.0, r=0.0),
            mk_trade(day=5, outcome=Outcome.OPEN, exit_price=None),
        ]
        src = build_pine(trades)
        self.assertIn("66.7%", src)

    def test_caps_at_the_drawing_limit_keeping_newest(self) -> None:
        trades = [mk_trade(day=(d % 28) + 1, entry=4000.0 + d)
                  for d in range(MAX_TRADES + 40)]
        src = build_pine(trades)
        m = re.search(r"sigEntry = array\.from\(([^)]*)\)", src)
        self.assertIsNotNone(m)
        self.assertEqual(len(m.group(1).split(",")), MAX_TRADES)

    def test_timestamps_are_milliseconds(self) -> None:
        src = build_pine([mk_trade(day=1)])
        m = re.search(r"sigTime = array\.from\((\d+)\)", src)
        self.assertIsNotNone(m)
        ms = int(m.group(1))
        expected = int(datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(ms, expected)

    def test_naive_timestamps_treated_as_utc(self) -> None:
        """A journal row read back without a timezone must not shift the drawing."""
        t = mk_trade(day=1)
        t.signal_time = t.signal_time.replace(tzinfo=None)
        src = build_pine([t])
        self.assertIn(str(int(datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
                              .timestamp() * 1000)), src)

    def test_notes_cannot_break_out_of_the_generated_string(self) -> None:
        """Journal text is interpolated into source, so quotes must be neutralised."""
        t = mk_trade(day=1, mode='pa"per')
        src = build_pine([t])
        ok, msg = brackets_balanced(src)
        self.assertTrue(ok, f"a quote in journal text broke the output: {msg}")
        self.assertNotIn('"pa"per"', src)

    def test_generated_script_has_a_global_output_call(self) -> None:
        """Same CE10244 guard for the generated indicator.

        Every drawing call in the generated file sits inside
        `if barstate.islast`, so without an unconditional plot() it will not
        compile -- which is how it was first written.
        """
        for trades in ([], [mk_trade(day=1)], [mk_trade(day=d) for d in (1, 2, 3)]):
            src = build_pine(trades)
            self.assertTrue(
                global_output_calls(src),
                "generated Pine has no global output call -- CE10244",
            )

    def test_generated_script_has_no_local_output_calls(self) -> None:
        src = build_pine([mk_trade(day=1)])
        self.assertEqual(indented_output_calls(src), [])

    def test_anchor_plot_draws_nothing(self) -> None:
        """The compile-satisfying plot must not put a line on the chart."""
        src = build_pine([mk_trade(day=1)])
        self.assertIn('plot(na, "anchor")', src)

    def test_mode_filter_values_are_present(self) -> None:
        src = build_pine([mk_trade(day=1, mode=Mode.LIVE.value)])
        self.assertIn('"live"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
