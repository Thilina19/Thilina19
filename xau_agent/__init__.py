"""XAUUSD disciplined signal agent.

A signal-generation, risk-management and self-review system for gold. It does
NOT place orders -- it produces trade plans (entry, stop, two targets, position
size) and refuses to produce them when its own statistics say it has not earned
the right to.
"""

from .config import DEFAULT_CONFIG, AgentConfig, RiskLimits, StrategyParams
from .indicators import Bar
from .journal import Journal, JournalEntry, Mode, PerformanceGate, compute_stats
from .risk import RiskManager, SessionState
from .pine_export import build_pine, write_pine
from .strategy import Rejection, Side, Signal, Strategy

__all__ = [
    "AgentConfig", "RiskLimits", "StrategyParams", "DEFAULT_CONFIG",
    "Bar", "Journal", "JournalEntry", "Mode", "PerformanceGate", "compute_stats",
    "RiskManager", "SessionState",
    "Strategy", "Signal", "Rejection", "Side",
    "build_pine", "write_pine",
]

__version__ = "0.1.0"
