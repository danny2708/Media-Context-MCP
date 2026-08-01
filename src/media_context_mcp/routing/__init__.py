"""Processor selection: pure, inspectable heuristics.

The dispatch itself (name -> processor instance) is a dict lookup in
``pipeline.py``; a separate router class would only add indirection.
"""

from .heuristics import Capabilities, RoutingDecision, classify_intent, decide_route

__all__ = ["Capabilities", "RoutingDecision", "classify_intent", "decide_route"]
