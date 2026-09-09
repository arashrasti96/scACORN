"""Agentic Context Engineering (ACE) reproduction framework."""

from .playbook import Bullet, Playbook
from .delta import DeltaOperation, DeltaBatch
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient
from .roles import (
    Generator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)

# Import explainability components
try:
    from .explainability import (
        EvolutionTracker,
        AttributionAnalyzer,
        InteractionTracer,
        ExplainabilityVisualizer,
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EvolutionTracker = None
    AttributionAnalyzer = None
    InteractionTracer = None
    ExplainabilityVisualizer = None
    EXPLAINABILITY_AVAILABLE = False

# Import production LLM clients if available
try:
    from .llm_providers import LiteLLMClient

    LITELLM_AVAILABLE = True
except ImportError:
    LiteLLMClient = None
    LITELLM_AVAILABLE = False

__all__ = [
    "Bullet",
    "Playbook",
    "DeltaOperation",
    "DeltaBatch",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "LiteLLMClient",
    "Generator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    "EvolutionTracker",
    "AttributionAnalyzer",
    "InteractionTracer",
    "ExplainabilityVisualizer",
    "LITELLM_AVAILABLE",
    "EXPLAINABILITY_AVAILABLE",
]
