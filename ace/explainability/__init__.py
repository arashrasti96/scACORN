"""
Explainability module for ACE (Agentic Context Engineering).

This module provides comprehensive tools for understanding and explaining
how ACE systems learn, adapt, and improve over time.

Components:
- EvolutionTracker: Tracks playbook changes and strategy emergence
- AttributionAnalyzer: Analyzes bullet effectiveness and contribution patterns
- InteractionTracer: Traces role interactions and decision chains
- Visualizer: Creates interactive dashboards and explanatory visualizations
"""

from .evolution_tracker import EvolutionTracker, PlaybookSnapshot, StrategyEvolution
from .attribution_analyzer import AttributionAnalyzer, BulletAttribution, StrategyCorrelation
from .interaction_tracer import InteractionTracer, RoleInteraction, DecisionChain
from .visualizer import ExplainabilityVisualizer

__all__ = [
    "EvolutionTracker",
    "PlaybookSnapshot",
    "StrategyEvolution",
    "AttributionAnalyzer",
    "BulletAttribution",
    "StrategyCorrelation",
    "InteractionTracer",
    "RoleInteraction",
    "DecisionChain",
    "ExplainabilityVisualizer",
]