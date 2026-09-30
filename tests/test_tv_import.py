"""Tests for the TradingView Strategy Tester import.

This is the path real trades take into the journal, so the review loop can only
learn from what this parses correctly. A silent mis-parse here would poison
every statistic downstream.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import timezone
from pathlib import Path

from xau_agent.journal import Mode, Outcome
from xau_agent.tv_import import parse_tv_export

HEADER = ("Trade #,Type,Date/Time,Signal,Price USD,Position size (qty),"
          "Net P&L USD,Net P&L %,Run-up USD,Run-up %,Drawdown USD,Drawdown %,"
          "Cumulative P&L USD,Cumulative P&L %")

BODY = """1,Entry long,2026-09-22 08:00,LONG,4180.50,25,,,,,,,,
1,Exit long,2026-09-22 14:00,TP,4193.20,25,317.50,0.64,340.00,0.68,-95.00,-0.19,317.50,0.64
2,Entry short,2026-09-23 11:00,SHORT,4210.75,25,,,,,,,,
2,Exit short,2026-09-23 15:30,SL,4220.75,25,-250.00,-0.50,60.00,0.12,-250.00,-0.50,67.50,0.14
3,Entry long,2026-09-26 13:00,LONG,4175.00,25,,,,,,,,"""


def write(text: str) -> str:
    d = tempfile.mkdtemp()
    p = Path(d) / "trades.csv"
    p.write_text(text)
    return str(p)


class TestTradingViewImport(unittest.TestCase):
    def setUp(self) -> None:
        self.path = write(HEADER + "\n" + BODY + "\n")

    def test_pairs_entry_and_exit_rows(self) -> None:
        t = parse_tv_export(self.path)
        self.assertEqual(len(t), 2, "should pair two completed trades")

    def test_open_trade_without_exit_is_skipped(self) -> None:
        """Trade 3 has an entry and no exit; including it would invent a result."""
        for tr in parse_tv_export(self.path):
            self.assertIsNotNone(tr.exit_price)
            self.assertNotIn("4175.00", str(tr.entry))

    def test_sides_read_from_the_entry_row(self) -> None:
        t = parse_tv_export(self.path)
        self.assertEqual([x.side for x in t], ["long", "short"])

    def test_prices_and_pnl(self) -> None:
        t = parse_tv_export(self.path)
        self.assertAlmostEqual(t[0].entry, 4180.50)
        self.assertAlmostEqual(t[0].exit_price, 4193.20)
        self.assertAlmostEqual(t[0].pnl, 317.50)
        self.assertAlmostEqual(t[1].pnl, -250.00)

    def test_r_multiple_is_pnl_over_risk(self) -> None:
        t = parse_tv_export(self.path, risk_per_trade=250.0)
        self.assertAlmostEqual(t[0].r_multiple, 317.50 / 250.0, places=4)
        self.assertAlmostEqual(t[1].r_multiple, -1.0, places=4)

    def test_risk_scales_r_multiples(self) -> None:
        a = parse_tv_export(self.path, risk_per_trade=250.0)
        b = parse_tv_export(self.path, risk_per_trade=500.0)
        self.assertAlmostEqual(a[0].r_multiple, 2 * b[0].r_multiple, places=4)

    def test_zero_risk_rejected(self) -> None:
        """R is undefined without a risk figure; guessing one would be worse."""
        with self.assertRaises(ValueError):
            parse_tv_export(self.path, risk_per_trade=0)

    def test_outcomes(self) -> None:
        t = parse_tv_export(self.path)
        self.assertEqual(t[0].outcome, Outcome.WIN.value)
        self.assertEqual(t[1].outcome, Outcome.LOSS.value)

    def test_excursions_recorded_in_r(self) -> None:
        t = parse_tv_export(self.path, risk_per_trade=250.0)
        self.assertAlmostEqual(t[0].mfe_r, 340.0 / 250.0, places=4)
        self.assertAlmostEqual(t[0].mae_r, 95.0 / 250.0, places=4)

    def test_timestamps_are_utc(self) -> None:
        t = parse_tv_export(self.path)
        self.assertEqual(t[0].signal_time.tzinfo, timezone.utc)
        self.assertLess(t[0].signal_time, t[0].exit_time)

    def test_stop_and_targets_not_fabricated(self) -> None:
        """The export carries no stop. Inventing one would create a fake level."""
        for tr in parse_tv_export(self.path):
            self.assertEqual(tr.stop, tr.entry)
            self.assertEqual(tr.tp2, tr.entry)

    def test_currency_and_thousands_separators(self) -> None:
        body = HEADER + "\n" + (
            '1,Entry long,2026-09-22 08:00,LONG,"$4,180.50",25,,,,,,,,\n'
            '1,Exit long,2026-09-22 14:00,TP,"$4,193.20",25,"$1,317.50",0.64,'
            '"$1,340.00",0.68,"-$95.00",-0.19,"$1,317.50",0.64\n')
        t = parse_tv_export(write(body))
        self.assertAlmostEqual(t[0].entry, 4180.50)
        self.assertAlmostEqual(t[0].pnl, 1317.50)

    def test_parenthesised_negative(self) -> None:
        body = HEADER + "\n" + (
            "1,Entry short,2026-09-23 11:00,SHORT,4210.75,25,,,,,,,,\n"
            "1,Exit short,2026-09-23 15:30,SL,4220.75,25,(250.00),-0.50,"
            "60.00,0.12,(250.00),-0.50,67.50,0.14\n")
        t = parse_tv_export(write(body))
        self.assertAlmostEqual(t[0].pnl, -250.0)
        self.assertEqual(t[0].outcome, Outcome.LOSS.value)

    def test_alternate_column_names(self) -> None:
        """Older exports use 'Contracts' and 'Profit USD'."""
        alt = ("Trade #,Type,Date/Time,Signal,Price USD,Contracts,Profit USD,"
               "Profit %,Run-up USD,Run-up %,Drawdown USD,Drawdown %\n"
               "1,Entry long,2026-09-22 08:00,LONG,4180.50,25,,,,,,\n"
               "1,Exit long,2026-09-22 14:00,TP,4193.20,25,317.50,0.64,"
               "340.00,0.68,-95.00,-0.19\n")
        t = parse_tv_export(write(alt))
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t[0].pnl, 317.50)

    def test_wrong_file_rejected_with_guidance(self) -> None:
        bad = write("date,open,high,low,close\n2026-01-01,1,2,0,1\n")
        with self.assertRaises(ValueError) as ctx:
            parse_tv_export(bad)
        self.assertIn("List of Trades", str(ctx.exception))

    def test_empty_file(self) -> None:
        self.assertEqual(parse_tv_export(write("")), [])

    def test_mode_is_paper_by_default(self) -> None:
        """Imported history must not count as live until a human says so."""
        self.assertEqual(parse_tv_export(self.path)[0].mode, Mode.PAPER.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
