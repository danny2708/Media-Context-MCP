"""Processor selection: pure, inspectable heuristics plus the dispatching router."""

from .heuristics import Capabilities, RoutingDecision, decide_route
from .router import ProcessorRouter

__all__ = ["Capabilities", "ProcessorRouter", "RoutingDecision", "decide_route"]
