"""Configuration for the XAUUSD signal agent.

Two kinds of settings live here and they are NOT equal:

  RiskLimits     -- hard guardrails. The self-learning loop may never change
                    these. They are what stops a losing streak from becoming
                    a blown account. Only a human edits this class.

  StrategyParams -- tunable. The review loop may propose changes to these,
                    but only with walk-forward evidence behind the proposal
                    (see review.py).

All prices are in USD per troy ounce, the way XAUUSD quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any


# Gold contract convention: 1.00 standard lot = 100 oz, so a $1.00 move in
# price is $100 per lot. A "pip" on gold is ambiguous between brokers, so this
# system works in dollars of price movement only and never in pips.
OZ_PER_LOT = 100.0


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk guardrails. IMMUTABLE BY THE LEARNING LOOP.

    Defaults are set for a 50,000 USD account with a 500 USD/day target.
    At 0.5% risk per trade (250 USD), a single 2R winner meets the target,
    which is the whole point: the target is reachable on one good trade,
    so there is never a reason to force a second one.
    """

    account_equity: float = 50_000.0

    # Per-trade risk as a fraction of equity. 0.005 = 0.5% = 250 USD.
    risk_per_trade_pct: float = 0.005

    # Absolute ceiling on per-trade risk regardless of any other setting.
    max_risk_per_trade_pct: float = 0.01

    # Daily profit target. Reaching it stops trading for the day -- giving
    # profits back is the most common way a good day becomes a bad one.
    daily_profit_target: float = 500.0

    # Daily loss cap. Hitting it stops trading for the day, no exceptions.
    daily_loss_limit: float = 500.0

    # Hard trade-count ceiling per day. Quality over quantity.
    max_trades_per_day: int = 3

    # Consecutive losses that end the session. This is the anti-revenge rule.
    max_consecutive_losses: int = 2

    # Cool-off after any loss. Blocks the immediate re-entry that is the
    # signature of revenge trading.
    cooloff_minutes_after_loss: int = 30

    # Weekly and monthly drawdown brakes, as fractions of starting equity.
    weekly_loss_limit_pct: float = 0.03
    monthly_loss_limit_pct: float = 0.06

    # Never hold more than one XAUUSD position. No grid, no averaging down.
    max_concurrent_positions: int = 1

    # Minimum reward:risk for a signal to be publishable.
    min_reward_risk: float = 1.8

    def risk_dollars(self) -> float:
        pct = min(self.risk_per_trade_pct, self.max_risk_per_trade_pct)
        return self.account_equity * pct

    def validate(self) -> None:
        """Sanity-check the limits themselves. Called at startup."""
        if not 0 < self.risk_per_trade_pct <= self.max_risk_per_trade_pct:
            raise ValueError(
                f"risk_per_trade_pct {self.risk_per_trade_pct} must be in "
                f"(0, {self.max_risk_per_trade_pct}]"
            )
        if self.max_risk_per_trade_pct > 0.02:
            raise ValueError(
                "max_risk_per_trade_pct above 2% is not survivable; refusing"
            )
        if self.daily_loss_limit <= 0 or self.max_trades_per_day <= 0:
            raise ValueError("daily_loss_limit and max_trades_per_day must be positive")
        if self.max_consecutive_losses <= 0:
            raise ValueError("max_consecutive_losses must be positive")
        if self.min_reward_risk < 1.0:
            raise ValueError(
                "min_reward_risk below 1.0 requires a win rate above 50% just to "
                "break even; refusing"
            )
        if self.max_concurrent_positions != 1:
            raise ValueError("this system trades one XAUUSD position at a time")


@dataclass(frozen=True)
class StrategyParams:
    """Tunable strategy parameters. The review loop may propose changes.

    Timeframe roles:
      bias_tf    -- 4h, decides direction. We only trade with it.
      zone_tf    -- 1h, locates the order block / imbalance we want price to reach.
      entry_tf   -- 15m, times the Heikin Ashi trigger.
    """

    # CAPITALCOM is the feed this account actually charts and alerts on, so it
    # is the default: backtesting on one provider's data while being filled on
    # another's is a quiet source of optimism. Verify any new feed first with
    # `python3 -m xau_agent.cli calibrate` -- volume conventions differ enough
    # between providers to disable the order-block filter entirely.
    symbol: str = "CAPITALCOM:XAUUSD"
    bias_tf: str = "4h"
    zone_tf: str = "1h"
    entry_tf: str = "15m"

    # --- Trend / bias ---
    ema_fast: int = 50
    ema_slow: int = 200
    swing_lookback: int = 2  # bars either side for a fractal swing point
    # "strict" = structure only; "ema_fallback" = structure, then EMA trend
    # when structure is a range. See structure.market_bias for the measurement
    # that made the fallback the default.
    bias_mode: str = "ema_fallback"

    # --- Volatility ---
    atr_period: int = 14
    # Skip when gold is unusually quiet (the move cannot pay for the spread) or
    # unusually wild (stops become meaningless).
    #
    # This is measured RELATIVE to the instrument's own recent average ATR, not
    # as an absolute percentage of price. An absolute band is timeframe
    # dependent and fails silently: gold's ATR is about 0.38% of price on 1h but
    # only ~0.11% on 5m, so a 0.12% floor tuned for 15m rejects every single 5m
    # bar and the strategy produces no trades at all. A ratio to the rolling
    # average ATR carries across timeframes unchanged.
    vol_avg_len: int = 100
    min_vol_mult: float = 0.55
    max_vol_mult: float = 2.50

    # --- Order blocks ---
    # A displacement leg must move at least this many ATR to qualify as the
    # impulse that makes the preceding candle an order block.
    ob_displacement_atr: float = 1.3
    # Volume on the order block candle relative to its rolling average.
    ob_min_volume_ratio: float = 1.4
    ob_volume_lookback: int = 20
    # How many bars an order block stays valid before it is considered stale.
    ob_max_age_bars: int = 60
    # How close price must come to the zone to count as a tap, in ATR.
    ob_tap_tolerance_atr: float = 0.35
    # How many bars a zone tap stays "live" while we wait for the trigger.
    # Requiring the tap and the Heikin Ashi flip on the SAME bar is not how the
    # setup is actually traded: price enters the zone, then you wait for the
    # turn. Measured on real gold, insisting on the same bar left the HA trigger
    # failing on 72.5% of bars and almost nothing ever reached scoring.
    ob_tap_grace_bars: int = 6

    # --- Heikin Ashi trigger ---
    # Consecutive HA candles in the signal direction required to confirm.
    ha_confirm_bars: int = 2
    # Reject HA candles that are mostly wick -- those are indecision, not a turn.
    ha_max_wick_ratio: float = 0.62

    # --- Confluence gate ---
    # Signal is published only at or above this score. Raising it means
    # fewer, better trades. This is the primary overtrading control.
    min_confluence_score: int = 70

    # --- Stops and targets ---
    sl_atr_buffer: float = 0.55   # ATR padding beyond the structural level
    tp1_r: float = 1.0            # take partial here
    tp1_close_fraction: float = 0.5
    tp2_r: float = 2.2            # remainder target
    move_sl_to_be_at_tp1: bool = True

    # --- Sessions (UTC hours). Gold's Asian session is chop-prone, so it
    # gets a higher confluence bar rather than an outright ban.
    asian_hours: tuple[int, ...] = tuple(range(0, 7))
    london_hours: tuple[int, ...] = tuple(range(7, 13))
    ny_hours: tuple[int, ...] = tuple(range(13, 21))
    asian_score_penalty: int = 12

    # --- News blackout ---
    # Minutes either side of a high-impact event where no new entry is taken.
    news_blackout_minutes: int = 30

    # --- Execution assumptions used by the backtest ---
    spread_usd: float = 0.25      # typical XAUUSD spread in price terms
    slippage_usd: float = 0.10

    def validate(self) -> None:
        if self.bias_mode not in ("strict", "ema_fallback"):
            raise ValueError(f"unknown bias_mode {self.bias_mode!r}")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be shorter than ema_slow")
        if not 0 < self.min_vol_mult < self.max_vol_mult:
            raise ValueError("require 0 < min_vol_mult < max_vol_mult")
        if self.vol_avg_len < 20:
            raise ValueError("vol_avg_len below 20 is too noisy to be a baseline")
        if self.tp2_r <= self.tp1_r:
            raise ValueError("tp2_r must exceed tp1_r")
        if not 0 < self.tp1_close_fraction < 1:
            raise ValueError("tp1_close_fraction must be strictly between 0 and 1")
        if not 0 <= self.min_confluence_score <= 100:
            raise ValueError("min_confluence_score must be within 0..100")
        if self.ha_confirm_bars < 1:
            raise ValueError("ha_confirm_bars must be at least 1")

    # Parameters the review loop is allowed to touch. Anything outside this
    # set is structural and needs a human.
    TUNABLE: tuple[str, ...] = (
        "ob_displacement_atr",
        "ob_min_volume_ratio",
        "ob_tap_tolerance_atr",
        "ob_tap_grace_bars",
        "ha_confirm_bars",
        "ha_max_wick_ratio",
        "min_confluence_score",
        "sl_atr_buffer",
        "tp2_r",
        "min_vol_mult",
        "max_vol_mult",
        "asian_score_penalty",
        "bias_mode",
    )

    def with_overrides(self, **kw: Any) -> "StrategyParams":
        """Return a copy with overrides, refusing non-tunable fields."""
        illegal = set(kw) - set(self.TUNABLE)
        if illegal:
            raise ValueError(
                f"these parameters are structural and not auto-tunable: "
                f"{sorted(illegal)}"
            )
        out = replace(self, **kw)
        out.validate()
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConfig:
    risk: RiskLimits = field(default_factory=RiskLimits)
    strategy: StrategyParams = field(default_factory=StrategyParams)

    # --- Promotion gate ---
    #
    # These numbers come from the statistics, not from preference.
    #
    # We test the LOWER BOUND of the 95% Wilson interval, never the point
    # estimate, because a point estimate on a small sample is noise: 6 wins
    # from 10 is "60%" with a real interval of 26-88%.
    #
    # Sample size: at a 60% observed win rate the CI lower bound reaches
    #   n=60  -> 47.4%
    #   n=100 -> 50.2%
    # So *proving* 60% against a 50% floor takes about 100 trades. At 1-2
    # trades a day that is 2-4 months, which is simply how long it takes to
    # know. We set the bar at 60 trades with a 45% floor instead, because
    # 45% is what actually matters economically:
    #
    #   With TP1 taking half off at 1R and the rest running to ~2.2R, an
    #   average winner is about 1.6R, so breakeven is 1/(1+1.6) = 38%.
    #   A 45% floor clears breakeven with real margin while being reachable
    #   in a realistic amount of forward testing.
    #
    # target_win_rate stays at 0.60 as the goal we report progress against;
    # live_win_rate_floor is the threshold that actually unlocks live entries.
    min_trades_for_live: int = 60
    target_win_rate: float = 0.60
    live_win_rate_floor: float = 0.45
    # Trades needed before the 60% target itself is statistically established.
    trades_to_prove_target: int = 100
    min_expectancy_r: float = 0.20
    # Peak drawdown in R that forces demotion back to paper.
    max_drawdown_r_for_live: float = 8.0

    def validate(self) -> None:
        self.risk.validate()
        self.strategy.validate()
        if self.min_trades_for_live < 30:
            raise ValueError(
                "fewer than 30 trades cannot support a win-rate claim; refusing"
            )
        if self.live_win_rate_floor >= self.target_win_rate:
            raise ValueError(
                "live_win_rate_floor must sit below target_win_rate, or the gate "
                "can never open"
            )


DEFAULT_CONFIG = AgentConfig()
