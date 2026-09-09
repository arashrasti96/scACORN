"""
Role Interaction Tracer for ACE explainability.

Traces interactions between Generator, Reflector, and Curator roles,
capturing reasoning patterns, decision chains, and feedback loops.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..roles import GeneratorOutput, ReflectorOutput, CuratorOutput
from ..delta import DeltaBatch


@dataclass
class RoleInteraction:
    """Represents an interaction between ACE roles."""

    timestamp: str
    epoch: int
    step: int
    sample_id: str

    # Input data
    question: str
    context: str
    playbook_state: str  # Playbook content at this step

    # Role outputs
    generator_output: Dict[str, Any]
    reflector_output: Dict[str, Any]
    curator_output: Dict[str, Any]

    # Environmental feedback
    environment_feedback: str
    performance_metrics: Dict[str, float]

    # Analysis fields
    reasoning_chain: List[str] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    feedback_loops: List[str] = field(default_factory=list)

    @classmethod
    def from_ace_step(
        cls,
        sample_id: str,
        question: str,
        context: str,
        playbook_state: str,
        generator_output: GeneratorOutput,
        reflector_output: ReflectorOutput,
        curator_output: CuratorOutput,
        environment_feedback: str,
        performance_metrics: Dict[str, float],
        epoch: int,
        step: int
    ) -> RoleInteraction:
        """Create interaction from ACE step results."""
        interaction = cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            epoch=epoch,
            step=step,
            sample_id=sample_id,
            question=question,
            context=context,
            playbook_state=playbook_state,
            generator_output=generator_output.raw,
            reflector_output=reflector_output.raw,
            curator_output=curator_output.raw,
            environment_feedback=environment_feedback,
            performance_metrics=performance_metrics
        )

        # Extract reasoning chain
        interaction._extract_reasoning_chain(generator_output, reflector_output, curator_output)

        # Identify decision points
        interaction._identify_decision_points(generator_output, reflector_output, curator_output)

        # Trace feedback loops
        interaction._trace_feedback_loops(reflector_output, curator_output)

        return interaction

    def _extract_reasoning_chain(
        self,
        generator_output: GeneratorOutput,
        reflector_output: ReflectorOutput,
        curator_output: CuratorOutput
    ) -> None:
        """Extract the reasoning chain across all roles."""
        self.reasoning_chain = [
            f"Generator: {generator_output.reasoning}",
            f"Reflector - Error Analysis: {reflector_output.error_identification}",
            f"Reflector - Root Cause: {reflector_output.root_cause_analysis}",
            f"Reflector - Correct Approach: {reflector_output.correct_approach}",
            f"Reflector - Key Insight: {reflector_output.key_insight}",
            f"Curator: Applied {len(curator_output.delta.operations)} operations"
        ]

    def _identify_decision_points(
        self,
        generator_output: GeneratorOutput,
        reflector_output: ReflectorOutput,
        curator_output: CuratorOutput
    ) -> None:
        """Identify key decision points in the interaction."""
        self.decision_points = []

        # Generator decision: which bullets to use
        if generator_output.bullet_ids:
            self.decision_points.append({
                'role': 'Generator',
                'decision': 'bullet_selection',
                'details': {
                    'selected_bullets': generator_output.bullet_ids,
                    'reasoning': generator_output.reasoning[:200] + '...' if len(generator_output.reasoning) > 200 else generator_output.reasoning
                }
            })

        # Reflector decision: bullet tagging
        if reflector_output.bullet_tags:
            self.decision_points.append({
                'role': 'Reflector',
                'decision': 'bullet_evaluation',
                'details': {
                    'tagged_bullets': [{'id': tag.id, 'tag': tag.tag} for tag in reflector_output.bullet_tags],
                    'key_insight': reflector_output.key_insight
                }
            })

        # Curator decision: playbook modifications
        if curator_output.delta.operations:
            operation_summary = {}
            for op in curator_output.delta.operations:
                op_type = op.type.upper()
                operation_summary[op_type] = operation_summary.get(op_type, 0) + 1

            self.decision_points.append({
                'role': 'Curator',
                'decision': 'playbook_modification',
                'details': {
                    'operations': operation_summary,
                    'total_changes': len(curator_output.delta.operations)
                }
            })

    def _trace_feedback_loops(
        self,
        reflector_output: ReflectorOutput,
        curator_output: CuratorOutput
    ) -> None:
        """Trace feedback loops between roles."""
        self.feedback_loops = []

        # Reflection → Curation feedback
        if reflector_output.bullet_tags and curator_output.delta.operations:
            tagged_bullets = {tag.id for tag in reflector_output.bullet_tags}
            modified_bullets = {op.bullet_id for op in curator_output.delta.operations if op.bullet_id}

            if tagged_bullets.intersection(modified_bullets):
                self.feedback_loops.append(
                    f"Reflector tagged bullets {tagged_bullets} → "
                    f"Curator modified bullets {modified_bullets}"
                )

        # Error identification → Playbook updates
        if reflector_output.error_identification and curator_output.delta.operations:
            add_operations = [op for op in curator_output.delta.operations if op.type.upper() == 'ADD']
            if add_operations:
                self.feedback_loops.append(
                    f"Error '{reflector_output.error_identification[:50]}...' → "
                    f"Added {len(add_operations)} new strategies"
                )


@dataclass
class DecisionChain:
    """Represents a chain of decisions across multiple ACE steps."""

    chain_id: str
    start_epoch: int
    start_step: int
    end_epoch: Optional[int] = None
    end_step: Optional[int] = None

    # Chain components
    interactions: List[RoleInteraction] = field(default_factory=list)
    pattern_type: str = ""  # learning, refinement, exploration, etc.

    # Analysis
    effectiveness_progression: List[float] = field(default_factory=list)
    strategy_changes: List[str] = field(default_factory=list)
    outcome: str = ""  # successful, failed, ongoing

    @property
    def chain_length(self) -> int:
        """Number of interactions in this chain."""
        return len(self.interactions)

    @property
    def performance_trend(self) -> str:
        """Overall performance trend in this chain."""
        if len(self.effectiveness_progression) < 2:
            return "insufficient_data"

        start_perf = self.effectiveness_progression[0]
        end_perf = self.effectiveness_progression[-1]

        if end_perf > start_perf + 0.05:
            return "improving"
        elif end_perf < start_perf - 0.05:
            return "declining"
        else:
            return "stable"


class InteractionTracer:
    """
    Traces and analyzes interactions between ACE roles.

    This class captures the flow of information and decisions between
    Generator, Reflector, and Curator, providing insights into how
    the ACE system reasons and adapts.

    Example:
        >>> tracer = InteractionTracer()
        >>>
        >>> # During ACE adaptation
        >>> for step_result in adaptation_results:
        ...     interaction = tracer.record_interaction(
        ...         sample_id=step_result.sample.sample_id,
        ...         question=step_result.sample.question,
        ...         context=step_result.sample.context,
        ...         playbook_state=step_result.playbook_snapshot,
        ...         generator_output=step_result.generator_output,
        ...         reflector_output=step_result.reflection,
        ...         curator_output=step_result.curator_output,
        ...         environment_feedback=step_result.environment_result.feedback,
        ...         performance_metrics=step_result.environment_result.metrics,
        ...         epoch=epoch, step=step
        ...     )
        >>>
        >>> # Analyze interaction patterns
        >>> patterns = tracer.analyze_interaction_patterns()
        >>> decision_chains = tracer.identify_decision_chains()
        >>> feedback_analysis = tracer.analyze_feedback_loops()
    """

    def __init__(self):
        self.interactions: List[RoleInteraction] = []
        self.decision_chains: List[DecisionChain] = []
        self.interaction_patterns: Dict[str, List[RoleInteraction]] = {}

        # Analysis caches
        self._pattern_cache: Dict[str, Any] = {}
        self._chain_cache: List[DecisionChain] = []

    def record_interaction(
        self,
        sample_id: str,
        question: str,
        context: str,
        playbook_state: str,
        generator_output: GeneratorOutput,
        reflector_output: ReflectorOutput,
        curator_output: CuratorOutput,
        environment_feedback: str,
        performance_metrics: Dict[str, float],
        epoch: int,
        step: int
    ) -> RoleInteraction:
        """Record a complete ACE role interaction."""
        interaction = RoleInteraction.from_ace_step(
            sample_id=sample_id,
            question=question,
            context=context,
            playbook_state=playbook_state,
            generator_output=generator_output,
            reflector_output=reflector_output,
            curator_output=curator_output,
            environment_feedback=environment_feedback,
            performance_metrics=performance_metrics,
            epoch=epoch,
            step=step
        )

        self.interactions.append(interaction)

        # Clear caches
        self._pattern_cache.clear()
        self._chain_cache.clear()

        return interaction

    def analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in role interactions."""
        if 'patterns' in self._pattern_cache:
            return self._pattern_cache['patterns']

        patterns = {
            'bullet_selection_patterns': self._analyze_bullet_selection_patterns(),
            'reflection_patterns': self._analyze_reflection_patterns(),
            'curation_patterns': self._analyze_curation_patterns(),
            'feedback_loop_patterns': self._analyze_feedback_loop_patterns(),
            'reasoning_consistency': self._analyze_reasoning_consistency()
        }

        self._pattern_cache['patterns'] = patterns
        return patterns

    def _analyze_bullet_selection_patterns(self) -> Dict[str, Any]:
        """Analyze Generator bullet selection patterns."""
        selection_patterns = {
            'avg_bullets_used': 0.0,
            'bullet_reuse_rate': 0.0,
            'section_preferences': {},
            'performance_correlation': {}
        }

        if not self.interactions:
            return selection_patterns

        total_bullets = 0
        all_used_bullets = set()
        bullet_usage_count = {}
        section_usage = {}

        for interaction in self.interactions:
            generator_output = interaction.generator_output
            bullet_ids = generator_output.get('bullet_ids', [])

            total_bullets += len(bullet_ids)
            all_used_bullets.update(bullet_ids)

            for bullet_id in bullet_ids:
                bullet_usage_count[bullet_id] = bullet_usage_count.get(bullet_id, 0) + 1

            # Analyze section preferences (simplified)
            # Would need playbook metadata for accurate section mapping

        selection_patterns['avg_bullets_used'] = total_bullets / len(self.interactions)

        if all_used_bullets:
            reused_bullets = sum(1 for count in bullet_usage_count.values() if count > 1)
            selection_patterns['bullet_reuse_rate'] = reused_bullets / len(all_used_bullets)

        return selection_patterns

    def _analyze_reflection_patterns(self) -> Dict[str, Any]:
        """Analyze Reflector reasoning patterns."""
        reflection_patterns = {
            'avg_bullet_tags_per_step': 0.0,
            'insight_length_distribution': [],
            'error_identification_themes': {},
            'tagging_consistency': 0.0
        }

        if not self.interactions:
            return reflection_patterns

        total_tags = 0
        insight_lengths = []
        tag_patterns = {}

        for interaction in self.interactions:
            reflector_output = interaction.reflector_output

            # Count bullet tags
            bullet_tags = reflector_output.get('bullet_tags', [])
            total_tags += len(bullet_tags)

            # Analyze tag patterns
            for tag in bullet_tags:
                if isinstance(tag, dict) and 'tag' in tag:
                    tag_type = tag['tag']
                    tag_patterns[tag_type] = tag_patterns.get(tag_type, 0) + 1

            # Insight length
            key_insight = reflector_output.get('key_insight', '')
            if key_insight:
                insight_lengths.append(len(key_insight))

        reflection_patterns['avg_bullet_tags_per_step'] = total_tags / len(self.interactions)
        reflection_patterns['insight_length_distribution'] = insight_lengths
        reflection_patterns['tag_distribution'] = tag_patterns

        return reflection_patterns

    def _analyze_curation_patterns(self) -> Dict[str, Any]:
        """Analyze Curator operation patterns."""
        curation_patterns = {
            'avg_operations_per_step': 0.0,
            'operation_type_distribution': {},
            'playbook_growth_rate': 0.0,
            'modification_frequency': {}
        }

        if not self.interactions:
            return curation_patterns

        total_operations = 0
        operation_types = {}

        for interaction in self.interactions:
            curator_output = interaction.curator_output
            operations = curator_output.get('operations', [])

            total_operations += len(operations)

            for operation in operations:
                if isinstance(operation, dict) and 'type' in operation:
                    op_type = operation['type'].upper()
                    operation_types[op_type] = operation_types.get(op_type, 0) + 1

        curation_patterns['avg_operations_per_step'] = total_operations / len(self.interactions)
        curation_patterns['operation_type_distribution'] = operation_types

        return curation_patterns

    def _analyze_feedback_loop_patterns(self) -> Dict[str, Any]:
        """Analyze feedback loops between roles."""
        feedback_patterns = {
            'feedback_loop_frequency': 0.0,
            'loop_types': {},
            'effectiveness_correlation': 0.0
        }

        if not self.interactions:
            return feedback_patterns

        total_loops = 0
        loop_types = {}

        for interaction in self.interactions:
            loops = interaction.feedback_loops
            total_loops += len(loops)

            for loop in loops:
                # Categorize loop types
                if 'tagged' in loop and 'modified' in loop:
                    loop_types['reflection_to_curation'] = loop_types.get('reflection_to_curation', 0) + 1
                elif 'Error' in loop and 'Added' in loop:
                    loop_types['error_to_strategy'] = loop_types.get('error_to_strategy', 0) + 1

        feedback_patterns['feedback_loop_frequency'] = total_loops / len(self.interactions)
        feedback_patterns['loop_types'] = loop_types

        return feedback_patterns

    def _analyze_reasoning_consistency(self) -> Dict[str, Any]:
        """Analyze consistency in reasoning across roles."""
        consistency_analysis = {
            'generator_reasoning_diversity': 0.0,
            'reflector_insight_repetition': 0.0,
            'curator_operation_predictability': 0.0
        }

        # Simplified analysis - could be expanded with NLP techniques
        if not self.interactions:
            return consistency_analysis

        generator_reasonings = []
        reflector_insights = []

        for interaction in self.interactions:
            gen_reasoning = interaction.generator_output.get('reasoning', '')
            ref_insight = interaction.reflector_output.get('key_insight', '')

            if gen_reasoning:
                generator_reasonings.append(gen_reasoning)
            if ref_insight:
                reflector_insights.append(ref_insight)

        # Simple diversity measure: unique reasoning patterns
        if generator_reasonings:
            unique_reasonings = len(set(generator_reasonings))
            consistency_analysis['generator_reasoning_diversity'] = unique_reasonings / len(generator_reasonings)

        if reflector_insights:
            unique_insights = len(set(reflector_insights))
            consistency_analysis['reflector_insight_repetition'] = 1.0 - (unique_insights / len(reflector_insights))

        return consistency_analysis

    def identify_decision_chains(self, min_chain_length: int = 3) -> List[DecisionChain]:
        """Identify chains of related decisions across multiple steps."""
        if self._chain_cache:
            return self._chain_cache

        chains = []
        current_chain = None

        for i, interaction in enumerate(self.interactions):
            # Start a new chain if we have significant operations
            curator_ops = interaction.curator_output.get('operations', [])
            has_significant_changes = len(curator_ops) > 0

            if has_significant_changes:
                if current_chain is None:
                    # Start new chain
                    current_chain = DecisionChain(
                        chain_id=f"chain_{len(chains)}",
                        start_epoch=interaction.epoch,
                        start_step=interaction.step,
                        pattern_type="learning"
                    )

                # Add interaction to current chain
                current_chain.interactions.append(interaction)

                # Track performance
                f1_score = interaction.performance_metrics.get('f1', 0.0)
                current_chain.effectiveness_progression.append(f1_score)

                # Track strategy changes
                for op in curator_ops:
                    if isinstance(op, dict) and 'type' in op:
                        current_chain.strategy_changes.append(f"{op['type']}: {op.get('content', '')[:50]}")

            else:
                # End current chain if it's long enough
                if current_chain and len(current_chain.interactions) >= min_chain_length:
                    current_chain.end_epoch = interaction.epoch
                    current_chain.end_step = interaction.step
                    current_chain.outcome = current_chain.performance_trend
                    chains.append(current_chain)

                current_chain = None

        # Don't forget the last chain
        if current_chain and len(current_chain.interactions) >= min_chain_length:
            current_chain.end_epoch = self.interactions[-1].epoch
            current_chain.end_step = self.interactions[-1].step
            current_chain.outcome = current_chain.performance_trend
            chains.append(current_chain)

        self._chain_cache = chains
        return chains

    def analyze_feedback_loops(self) -> Dict[str, Any]:
        """Comprehensive analysis of feedback loops."""
        loops_analysis = {
            'total_loops_identified': 0,
            'loop_effectiveness': {},
            'common_loop_patterns': [],
            'role_interaction_strength': {}
        }

        all_loops = []
        for interaction in self.interactions:
            all_loops.extend(interaction.feedback_loops)

        loops_analysis['total_loops_identified'] = len(all_loops)

        # Analyze loop effectiveness
        loop_performance_map = {}
        for interaction in self.interactions:
            f1_score = interaction.performance_metrics.get('f1', 0.0)
            for loop in interaction.feedback_loops:
                if loop not in loop_performance_map:
                    loop_performance_map[loop] = []
                loop_performance_map[loop].append(f1_score)

        for loop, performances in loop_performance_map.items():
            if performances:
                loops_analysis['loop_effectiveness'][loop] = {
                    'avg_performance': sum(performances) / len(performances),
                    'occurrences': len(performances)
                }

        return loops_analysis

    def generate_interaction_report(self) -> Dict[str, Any]:
        """Generate comprehensive interaction analysis report."""
        patterns = self.analyze_interaction_patterns()
        decision_chains = self.identify_decision_chains()
        feedback_analysis = self.analyze_feedback_loops()

        return {
            'summary': {
                'total_interactions': len(self.interactions),
                'decision_chains_identified': len(decision_chains),
                'avg_chain_length': sum(chain.chain_length for chain in decision_chains) / len(decision_chains) if decision_chains else 0,
                'feedback_loops_total': feedback_analysis['total_loops_identified']
            },
            'interaction_patterns': patterns,
            'decision_chains': [
                {
                    'chain_id': chain.chain_id,
                    'length': chain.chain_length,
                    'performance_trend': chain.performance_trend,
                    'pattern_type': chain.pattern_type,
                    'outcome': chain.outcome
                }
                for chain in decision_chains
            ],
            'feedback_analysis': feedback_analysis,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def export_traces(self, file_path: Union[str, Path]) -> None:
        """Export complete interaction traces to JSON file."""
        trace_data = {
            'interactions': [asdict(interaction) for interaction in self.interactions],
            'decision_chains': [asdict(chain) for chain in self.decision_chains],
            'analysis_report': self.generate_interaction_report(),
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_traces(cls, file_path: Union[str, Path]) -> InteractionTracer:
        """Load interaction traces from JSON file."""
        file_path = Path(file_path)

        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        tracer = cls()

        # Restore interactions
        for interaction_data in data.get('interactions', []):
            tracer.interactions.append(RoleInteraction(**interaction_data))

        # Restore decision chains
        for chain_data in data.get('decision_chains', []):
            # Need to reconstruct interactions within chains
            chain = DecisionChain(**{k: v for k, v in chain_data.items() if k != 'interactions'})
            # Note: This would need more sophisticated reconstruction of nested objects
            tracer.decision_chains.append(chain)

        return tracer