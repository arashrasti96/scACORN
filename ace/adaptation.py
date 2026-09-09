"""Adaptation loops for offline and online ACE training."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .playbook import Playbook
from .roles import (
    Curator,
    CuratorOutput,
    Generator,
    GeneratorOutput,
    Reflector,
    ReflectorOutput,
)
from .explainability import EvolutionTracker, AttributionAnalyzer, InteractionTracer


@dataclass
class Sample:
    """Single task instance presented to ACE."""

    question: str
    context: str = ""
    ground_truth: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EnvironmentResult:
    """Feedback returned by the task environment after executing the generator output."""

    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class TaskEnvironment(ABC):
    """
    Abstract interface for evaluating generator outputs.

    Implement this class to define how your specific task evaluates
    the Generator's answers. The environment provides feedback that
    helps ACE learn what works and what doesn't.

    Example Implementation:
        >>> class MathEnvironment(TaskEnvironment):
        ...     def evaluate(self, sample, generator_output):
        ...         # Parse the answer
        ...         predicted = extract_number(generator_output.final_answer)
        ...         correct = str(predicted) == sample.ground_truth
        ...
        ...         # Provide feedback
        ...         if correct:
        ...             feedback = "Correct!"
        ...         else:
        ...             feedback = f"Incorrect. Expected {sample.ground_truth}"
        ...
        ...         return EnvironmentResult(
        ...             feedback=feedback,
        ...             ground_truth=sample.ground_truth,
        ...             metrics={'accuracy': 1.0 if correct else 0.0}
        ...         )
    """

    @abstractmethod
    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """
        Evaluate the generator's output for a given sample.

        Args:
            sample: The input sample with question and context
            generator_output: The Generator's produced answer

        Returns:
            EnvironmentResult with feedback and optional ground truth

        The feedback should be informative enough for the Reflector
        to understand what went right or wrong.
        """


@dataclass
class AdapterStepResult:
    sample: Sample
    generator_output: GeneratorOutput
    environment_result: EnvironmentResult
    reflection: ReflectorOutput
    curator_output: CuratorOutput
    playbook_snapshot: str

    # Explainability metadata
    epoch: int = 0
    step: int = 0
    bullet_metadata: Dict[str, Dict] = field(default_factory=dict)


class AdapterBase:
    """Shared orchestration logic for offline and online ACE adaptation."""

    def __init__(
        self,
        *,
        playbook: Optional[Playbook] = None,
        generator: Generator,
        reflector: Reflector,
        curator: Curator,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
        enable_explainability: bool = True,
    ) -> None:
        self.playbook = playbook or Playbook()
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window
        self._recent_reflections: List[str] = []

        # Explainability components
        self.enable_explainability = enable_explainability
        if enable_explainability:
            self.evolution_tracker = EvolutionTracker()
            self.attribution_analyzer = AttributionAnalyzer()
            self.interaction_tracer = InteractionTracer()
        else:
            self.evolution_tracker = None
            self.attribution_analyzer = None
            self.interaction_tracer = None

    # ------------------------------------------------------------------ #
    # Explainability tracking methods
    # ------------------------------------------------------------------ #
    def _track_explainability_data(
        self,
        sample: Sample,
        generator_output: GeneratorOutput,
        environment_result: EnvironmentResult,
        reflection: ReflectorOutput,
        curator_output: CuratorOutput,
        epoch: int,
        step: int
    ) -> None:
        """Track data for explainability analysis."""
        if not self.enable_explainability:
            return

        sample_id = sample.metadata.get('sample_id', f'sample_{step}')

        # Track evolution
        if self.evolution_tracker:
            # Take snapshot before delta application
            self.evolution_tracker.take_snapshot(
                playbook=self.playbook,
                epoch=epoch,
                step=step,
                performance_metrics=environment_result.metrics,
                context=f"Processing sample {sample_id}"
            )

            # Record delta operations
            self.evolution_tracker.record_delta(
                delta=curator_output.delta,
                epoch=epoch,
                step=step,
                context="after_curator"
            )

        # Track attribution
        if self.attribution_analyzer:
            # Determine success
            success = None
            if 'f1' in environment_result.metrics:
                success = environment_result.metrics['f1'] > 0.5
            elif any(v > 0.7 for v in environment_result.metrics.values()):
                success = True

            # Create bullet metadata
            bullet_metadata = {}
            for bullet in self.playbook.bullets():
                bullet_metadata[bullet.id] = {
                    'section': bullet.section,
                    'content': bullet.content
                }

            self.attribution_analyzer.record_bullet_usage(
                bullet_ids=generator_output.bullet_ids,
                performance_metrics=environment_result.metrics,
                sample_id=sample_id,
                epoch=epoch,
                step=step,
                success=success,
                bullet_metadata=bullet_metadata
            )

        # Track interactions
        if self.interaction_tracer:
            self.interaction_tracer.record_interaction(
                sample_id=sample_id,
                question=sample.question,
                context=sample.context,
                playbook_state=self.playbook.as_prompt(),
                generator_output=generator_output,
                reflector_output=reflection,
                curator_output=curator_output,
                environment_feedback=environment_result.feedback,
                performance_metrics=environment_result.metrics,
                epoch=epoch,
                step=step
            )

    def get_explainability_data(self) -> Dict[str, Any]:
        """Get all explainability data collected during adaptation."""
        if not self.enable_explainability:
            return {}

        data = {}

        if self.evolution_tracker:
            data['evolution'] = {
                'summary': self.evolution_tracker.get_evolution_summary(),
                'lifespans': self.evolution_tracker.analyze_strategy_lifespans(),
                'patterns': self.evolution_tracker.identify_learning_patterns()
            }

        if self.attribution_analyzer:
            data['attribution'] = self.attribution_analyzer.generate_attribution_report()

        if self.interaction_tracer:
            data['interactions'] = self.interaction_tracer.generate_interaction_report()

        return data

    def export_explainability_analysis(self, output_dir: str) -> Dict[str, str]:
        """Export all explainability analysis to files."""
        if not self.enable_explainability:
            return {}

        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        if self.evolution_tracker:
            evolution_file = output_path / "evolution_analysis.json"
            self.evolution_tracker.export_timeline(evolution_file)
            exported_files['evolution'] = str(evolution_file)

        if self.attribution_analyzer:
            attribution_file = output_path / "attribution_analysis.json"
            self.attribution_analyzer.export_analysis(attribution_file)
            exported_files['attribution'] = str(attribution_file)

        if self.interaction_tracer:
            interaction_file = output_path / "interaction_analysis.json"
            self.interaction_tracer.export_traces(interaction_file)
            exported_files['interactions'] = str(interaction_file)

        return exported_files

    # ------------------------------------------------------------------ #
    def _reflection_context(self) -> str:
        return "\n---\n".join(self._recent_reflections)

    def _update_recent_reflections(self, reflection: ReflectorOutput) -> None:
        serialized = json.dumps(reflection.raw, ensure_ascii=False)
        self._recent_reflections.append(serialized)
        if len(self._recent_reflections) > self.reflection_window:
            self._recent_reflections = self._recent_reflections[
                -self.reflection_window :
            ]

    def _apply_bullet_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.bullet_tags:
            try:
                self.playbook.tag_bullet(tag.id, tag.tag)
            except ValueError:
                continue

    def _question_context(
        self, sample: Sample, environment_result: EnvironmentResult
    ) -> str:
        parts = [
            f"question: {sample.question}",
            f"context: {sample.context}",
            f"metadata: {json.dumps(sample.metadata)}",
            f"feedback: {environment_result.feedback}",
            f"ground_truth: {environment_result.ground_truth}",
        ]
        return "\n".join(parts)

    def _progress_string(
        self, epoch: int, total_epochs: int, step: int, total_steps: int
    ) -> str:
        return f"epoch {epoch}/{total_epochs} · sample {step}/{total_steps}"

    def _process_sample(
        self,
        sample: Sample,
        environment: TaskEnvironment,
        *,
        epoch: int,
        total_epochs: int,
        step_index: int,
        total_steps: int,
    ) -> AdapterStepResult:
        generator_output = self.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=self.playbook,
            reflection=self._reflection_context(),
        )
        env_result = environment.evaluate(sample, generator_output)
        reflection = self.reflector.reflect(
            question=sample.question,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            max_refinement_rounds=self.max_refinement_rounds,
        )
        self._apply_bullet_tags(reflection)
        self._update_recent_reflections(reflection)
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=self._question_context(sample, env_result),
            progress=self._progress_string(
                epoch, total_epochs, step_index, total_steps
            ),
        )

        # Track explainability data if enabled
        if self.enable_explainability:
            self._track_explainability_data(
                sample, generator_output, env_result, reflection, curator_output,
                epoch, step_index
            )

        self.playbook.apply_delta(curator_output.delta)

        # Create bullet metadata for explainability
        bullet_metadata = {}
        if self.enable_explainability:
            for bullet in self.playbook.bullets():
                bullet_metadata[bullet.id] = {
                    'section': bullet.section,
                    'content': bullet.content,
                    'helpful': bullet.helpful,
                    'harmful': bullet.harmful,
                    'neutral': bullet.neutral
                }

        return AdapterStepResult(
            sample=sample,
            generator_output=generator_output,
            environment_result=env_result,
            reflection=reflection,
            curator_output=curator_output,
            playbook_snapshot=self.playbook.as_prompt(),
            epoch=epoch,
            step=step_index,
            bullet_metadata=bullet_metadata,
        )


class OfflineAdapter(AdapterBase):
    """
    Orchestrates offline ACE adaptation over multiple training epochs.

    The OfflineAdapter processes a fixed training set multiple times,
    allowing the playbook to evolve and improve through repeated exposure
    to the same examples. This is useful for building a robust initial
    playbook before deployment.

    Args:
        playbook: Initial playbook (creates empty one if None)
        generator: Generator instance for producing answers
        reflector: Reflector instance for analyzing outcomes
        curator: Curator instance for updating playbook
        max_refinement_rounds: Max reflection refinement attempts (default: 1)
        reflection_window: Number of recent reflections to maintain (default: 3)

    Example:
        >>> from ace import OfflineAdapter, Generator, Reflector, Curator
        >>> from ace.llm_providers import LiteLLMClient
        >>>
        >>> # Initialize components with same LLM
        >>> client = LiteLLMClient(model="gpt-4")
        >>> generator = Generator(client)
        >>> reflector = Reflector(client)
        >>> curator = Curator(client)
        >>>
        >>> # Create adapter
        >>> adapter = OfflineAdapter(
        ...     generator=generator,
        ...     reflector=reflector,
        ...     curator=curator
        ... )
        >>>
        >>> # Prepare training samples
        >>> samples = [
        ...     Sample(question="What is 2+2?", ground_truth="4"),
        ...     Sample(question="What is 5*3?", ground_truth="15")
        ... ]
        >>>
        >>> # Run adaptation for 3 epochs
        >>> results = adapter.run(samples, environment, epochs=3)
        >>>
        >>> # Access evolved playbook
        >>> print(adapter.playbook.as_prompt())

    The adapter will:
        1. Process each sample through Generator → Environment → Reflector → Curator
        2. Update the playbook after each sample
        3. Repeat for the specified number of epochs
        4. Return detailed results for analysis
    """

    def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
    ) -> List[AdapterStepResult]:
        """
        Run offline adaptation over training samples.

        Args:
            samples: Training samples to process
            environment: Environment for evaluating generator outputs
            epochs: Number of times to iterate over samples (default: 1)

        Returns:
            List of AdapterStepResult for each processed sample

        Note:
            The playbook is updated in-place during adaptation.
            Access the evolved playbook via adapter.playbook after running.
        """
        results: List[AdapterStepResult] = []
        total_steps = len(samples)
        for epoch_idx in range(1, epochs + 1):
            for step_idx, sample in enumerate(samples, start=1):
                result = self._process_sample(
                    sample,
                    environment,
                    epoch=epoch_idx,
                    total_epochs=epochs,
                    step_index=step_idx,
                    total_steps=total_steps,
                )
                results.append(result)
        return results


class OnlineAdapter(AdapterBase):
    """
    Orchestrates online ACE adaptation for continuous learning.

    The OnlineAdapter processes samples sequentially as they arrive,
    updating the playbook after each one. This enables continuous
    improvement during deployment, adapting to new patterns and
    correcting mistakes in real-time.

    Args:
        playbook: Initial playbook (creates empty one if None)
        generator: Generator instance for producing answers
        reflector: Reflector instance for analyzing outcomes
        curator: Curator instance for updating playbook
        max_refinement_rounds: Max reflection refinement attempts (default: 1)
        reflection_window: Number of recent reflections to maintain (default: 3)

    Example:
        >>> from ace import OnlineAdapter, Generator, Reflector, Curator
        >>> from ace.llm_providers import LiteLLMClient
        >>>
        >>> # Initialize with pre-trained playbook
        >>> playbook = Playbook.load_from_file("pretrained_playbook.json")
        >>>
        >>> client = LiteLLMClient(model="gpt-4")
        >>> adapter = OnlineAdapter(
        ...     playbook=playbook,
        ...     generator=Generator(client),
        ...     reflector=Reflector(client),
        ...     curator=Curator(client)
        ... )
        >>>
        >>> # Process streaming samples
        >>> def sample_stream():
        ...     while True:
        ...         yield get_next_sample()  # Your sample source
        >>>
        >>> # Run online adaptation
        >>> results = adapter.run(sample_stream(), environment)
        >>>
        >>> # Playbook evolves with each sample
        >>> print(f"Bullets: {len(adapter.playbook.bullets)}")

    Online vs Offline:
        - Online: Processes each sample once, adapts immediately
        - Offline: Processes fixed set multiple times for thorough learning
        - Online is ideal for production deployment with continuous improvement
        - Offline is ideal for initial training before deployment
    """

    def run(
        self,
        samples: Iterable[Sample],
        environment: TaskEnvironment,
    ) -> List[AdapterStepResult]:
        """
        Run online adaptation over a stream of samples.

        Args:
            samples: Iterable of samples (can be infinite stream)
            environment: Environment for evaluating generator outputs

        Returns:
            List of AdapterStepResult for each processed sample

        Note:
            - Processes samples sequentially, updating after each one
            - The playbook evolves continuously during processing
            - Can handle infinite streams for continuous deployment
        """
        results: List[AdapterStepResult] = []
        step_idx = 0
        for step_idx, sample in enumerate(samples, start=1):
            result = self._process_sample(
                sample,
                environment,
                epoch=1,
                total_epochs=1,
                step_index=step_idx,
                total_steps=step_idx,
            )
            results.append(result)
        return results
