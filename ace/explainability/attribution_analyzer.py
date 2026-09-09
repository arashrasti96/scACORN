"""
Bullet Attribution Analyzer for ACE explainability.

Analyzes which bullets contribute most to performance improvements,
tracks correlations between bullet usage and success metrics,
and identifies strategy patterns and dependencies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Set, Tuple, Union

import math


@dataclass
class BulletAttribution:
    """Attribution analysis for a specific bullet."""

    bullet_id: str
    section: str
    content: str

    # Usage statistics
    usage_count: int = 0
    success_count: int = 0  # Times used in successful predictions
    failure_count: int = 0  # Times used in failed predictions

    # Performance correlation
    performance_when_used: List[float] = field(default_factory=list)
    performance_when_not_used: List[float] = field(default_factory=list)

    # Co-occurrence analysis
    frequently_used_with: Dict[str, int] = field(default_factory=dict)  # bullet_id -> co-occurrence count
    rarely_used_with: Dict[str, int] = field(default_factory=dict)

    # Temporal patterns
    usage_by_epoch: Dict[int, int] = field(default_factory=dict)
    effectiveness_trend: List[Tuple[str, float]] = field(default_factory=list)  # (timestamp, effectiveness)

    @property
    def success_rate(self) -> float:
        """Calculate success rate when this bullet is used."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def performance_impact(self) -> float:
        """Calculate performance impact (used vs not used)."""
        if not self.performance_when_used or not self.performance_when_not_used:
            return 0.0

        avg_when_used = mean(self.performance_when_used)
        avg_when_not_used = mean(self.performance_when_not_used)
        return avg_when_used - avg_when_not_used

    @property
    def performance_correlation(self) -> float:
        """Calculate correlation between usage and performance."""
        if len(self.performance_when_used) < 2:
            return 0.0

        # Simple correlation: higher performance when used vs baseline
        return self.performance_impact

    @property
    def attribution_score(self) -> float:
        """Calculate overall attribution score for this bullet."""
        # Combine multiple factors: usage, success rate, performance impact
        usage_weight = min(self.usage_count / 10.0, 1.0)  # Cap at 10 uses
        success_weight = self.success_rate
        impact_weight = max(0, self.performance_impact)  # Only positive impact

        return (usage_weight * 0.3 + success_weight * 0.4 + impact_weight * 0.3)


@dataclass
class StrategyCorrelation:
    """Correlation analysis between different strategies."""

    bullet_a: str
    bullet_b: str
    co_occurrence_count: int
    total_a_usage: int
    total_b_usage: int
    joint_success_rate: float
    individual_a_success_rate: float
    individual_b_success_rate: float

    @property
    def correlation_strength(self) -> float:
        """Calculate correlation strength between strategies."""
        if self.total_a_usage == 0 or self.total_b_usage == 0:
            return 0.0

        # Jaccard similarity
        union_size = self.total_a_usage + self.total_b_usage - self.co_occurrence_count
        return self.co_occurrence_count / union_size if union_size > 0 else 0.0

    @property
    def synergy_score(self) -> float:
        """Calculate synergy score (joint performance vs individual)."""
        individual_avg = (self.individual_a_success_rate + self.individual_b_success_rate) / 2
        return self.joint_success_rate - individual_avg


@dataclass
class PerformanceAttribution:
    """Attribution of performance changes to specific factors."""

    timestamp: str
    epoch: int
    step: int
    performance_change: Dict[str, float]  # metric -> change value
    contributing_bullets: List[str]  # bullets used in this step
    bullet_attributions: Dict[str, float]  # bullet_id -> attribution weight
    context: str = ""


class AttributionAnalyzer:
    """
    Analyzes bullet attribution and strategy correlations in ACE systems.

    This class provides comprehensive analysis of which strategies contribute
    most to performance improvements, how strategies interact with each other,
    and what patterns emerge in successful vs unsuccessful cases.

    Example:
        >>> analyzer = AttributionAnalyzer()
        >>>
        >>> # During ACE adaptation, record usage and outcomes
        >>> for step_result in adaptation_results:
        ...     analyzer.record_bullet_usage(
        ...         step_result.generator_output.bullet_ids,
        ...         step_result.environment_result.metrics,
        ...         step_result.sample.sample_id,
        ...         epoch, step
        ...     )
        >>>
        >>> # Analyze attributions
        >>> attributions = analyzer.compute_attributions()
        >>> correlations = analyzer.analyze_strategy_correlations()
        >>> top_contributors = analyzer.get_top_contributors(n=10)
    """

    def __init__(self):
        self.bullet_usage_history: List[Dict] = []
        self.performance_history: List[Dict] = []
        self.bullet_attributions: Dict[str, BulletAttribution] = {}
        self.strategy_correlations: Dict[Tuple[str, str], StrategyCorrelation] = {}
        self.performance_attributions: List[PerformanceAttribution] = []

        # Internal tracking
        self._bullet_metadata: Dict[str, Dict] = {}  # bullet_id -> {section, content}
        self._global_performance_baseline: Dict[str, List[float]] = defaultdict(list)

    def record_bullet_usage(
        self,
        bullet_ids: List[str],
        performance_metrics: Dict[str, float],
        sample_id: str,
        epoch: int,
        step: int,
        success: Optional[bool] = None,
        bullet_metadata: Optional[Dict[str, Dict]] = None
    ) -> None:
        """Record bullet usage and associated performance."""
        timestamp = datetime.now().isoformat()

        # Update metadata if provided
        if bullet_metadata:
            self._bullet_metadata.update(bullet_metadata)

        # Record usage event
        usage_event = {
            'timestamp': timestamp,
            'epoch': epoch,
            'step': step,
            'sample_id': sample_id,
            'bullet_ids': bullet_ids,
            'performance_metrics': performance_metrics.copy(),
            'success': success
        }
        self.bullet_usage_history.append(usage_event)

        # Update global performance baseline
        for metric, value in performance_metrics.items():
            self._global_performance_baseline[metric].append(value)

        # Initialize bullet attributions if needed
        for bullet_id in bullet_ids:
            if bullet_id not in self.bullet_attributions:
                metadata = self._bullet_metadata.get(bullet_id, {})
                self.bullet_attributions[bullet_id] = BulletAttribution(
                    bullet_id=bullet_id,
                    section=metadata.get('section', 'unknown'),
                    content=metadata.get('content', '')
                )

        # Update bullet usage statistics
        used_bullets = set(bullet_ids)
        all_known_bullets = set(self.bullet_attributions.keys())

        for bullet_id in all_known_bullets:
            attribution = self.bullet_attributions[bullet_id]

            if bullet_id in used_bullets:
                attribution.usage_count += 1
                attribution.usage_by_epoch[epoch] = attribution.usage_by_epoch.get(epoch, 0) + 1

                # Record performance when used
                for metric, value in performance_metrics.items():
                    if metric in ['f1', 'accuracy', 'precision', 'recall']:  # Common success metrics
                        attribution.performance_when_used.append(value)

                # Record success/failure
                if success is not None:
                    if success:
                        attribution.success_count += 1
                    else:
                        attribution.failure_count += 1
                elif 'f1' in performance_metrics and performance_metrics['f1'] > 0.5:
                    # Heuristic: F1 > 0.5 is success
                    attribution.success_count += 1
                else:
                    attribution.failure_count += 1

            else:
                # Record performance when not used
                for metric, value in performance_metrics.items():
                    if metric in ['f1', 'accuracy', 'precision', 'recall']:
                        attribution.performance_when_not_used.append(value)

        # Update co-occurrence statistics
        self._update_cooccurrence_stats(bullet_ids, performance_metrics)

    def _update_cooccurrence_stats(
        self,
        bullet_ids: List[str],
        performance_metrics: Dict[str, float]
    ) -> None:
        """Update co-occurrence statistics between bullets."""
        for i, bullet_a in enumerate(bullet_ids):
            for bullet_b in bullet_ids[i+1:]:
                # Update co-occurrence in attributions
                if bullet_a in self.bullet_attributions:
                    self.bullet_attributions[bullet_a].frequently_used_with[bullet_b] = \
                        self.bullet_attributions[bullet_a].frequently_used_with.get(bullet_b, 0) + 1

                if bullet_b in self.bullet_attributions:
                    self.bullet_attributions[bullet_b].frequently_used_with[bullet_a] = \
                        self.bullet_attributions[bullet_b].frequently_used_with.get(bullet_a, 0) + 1

                # Update strategy correlations
                pair = tuple(sorted([bullet_a, bullet_b]))
                if pair not in self.strategy_correlations:
                    self.strategy_correlations[pair] = StrategyCorrelation(
                        bullet_a=pair[0],
                        bullet_b=pair[1],
                        co_occurrence_count=0,
                        total_a_usage=0,
                        total_b_usage=0,
                        joint_success_rate=0.0,
                        individual_a_success_rate=0.0,
                        individual_b_success_rate=0.0
                    )

                correlation = self.strategy_correlations[pair]
                correlation.co_occurrence_count += 1

                # Calculate success for this co-occurrence
                success = any(v > 0.5 for k, v in performance_metrics.items()
                             if k in ['f1', 'accuracy', 'precision', 'recall'])
                if success:
                    correlation.joint_success_rate = \
                        (correlation.joint_success_rate * (correlation.co_occurrence_count - 1) + 1.0) / correlation.co_occurrence_count

    def compute_attributions(self) -> Dict[str, BulletAttribution]:
        """Compute comprehensive attribution analysis for all bullets."""
        # Update attribution statistics
        for bullet_id, attribution in self.bullet_attributions.items():
            # Calculate effectiveness trend
            attribution.effectiveness_trend = []
            for event in self.bullet_usage_history:
                if bullet_id in event['bullet_ids']:
                    effectiveness = event['performance_metrics'].get('f1', 0.0)
                    attribution.effectiveness_trend.append((event['timestamp'], effectiveness))

        # Update strategy correlation statistics
        for pair, correlation in self.strategy_correlations.items():
            bullet_a, bullet_b = pair

            if bullet_a in self.bullet_attributions:
                correlation.total_a_usage = self.bullet_attributions[bullet_a].usage_count
                correlation.individual_a_success_rate = self.bullet_attributions[bullet_a].success_rate

            if bullet_b in self.bullet_attributions:
                correlation.total_b_usage = self.bullet_attributions[bullet_b].usage_count
                correlation.individual_b_success_rate = self.bullet_attributions[bullet_b].success_rate

        return self.bullet_attributions

    def analyze_strategy_correlations(self) -> Dict[Tuple[str, str], StrategyCorrelation]:
        """Analyze correlations between different strategies."""
        self.compute_attributions()  # Ensure attributions are up to date
        return self.strategy_correlations

    def get_top_contributors(self, n: int = 10, metric: str = 'attribution_score') -> List[BulletAttribution]:
        """Get top N contributing bullets by specified metric."""
        self.compute_attributions()

        if metric == 'attribution_score':
            key_func = lambda x: x.attribution_score
        elif metric == 'performance_impact':
            key_func = lambda x: x.performance_impact
        elif metric == 'success_rate':
            key_func = lambda x: x.success_rate
        elif metric == 'usage_count':
            key_func = lambda x: x.usage_count
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted(self.bullet_attributions.values(), key=key_func, reverse=True)[:n]

    def get_strategy_synergies(self, min_co_occurrence: int = 3) -> List[StrategyCorrelation]:
        """Get strategy pairs with positive synergy effects."""
        self.analyze_strategy_correlations()

        synergistic_pairs = []
        for correlation in self.strategy_correlations.values():
            if (correlation.co_occurrence_count >= min_co_occurrence and
                correlation.synergy_score > 0.1):  # 10% synergy threshold
                synergistic_pairs.append(correlation)

        return sorted(synergistic_pairs, key=lambda x: x.synergy_score, reverse=True)

    def identify_performance_drivers(self) -> Dict[str, List[str]]:
        """Identify which bullets drive performance in different metrics."""
        drivers = defaultdict(list)

        for bullet_id, attribution in self.bullet_attributions.items():
            if attribution.performance_impact > 0.05:  # 5% improvement threshold
                # Determine which metrics this bullet helps with
                for event in self.bullet_usage_history:
                    if bullet_id in event['bullet_ids']:
                        for metric, value in event['performance_metrics'].items():
                            if value > 0.7:  # High performance threshold
                                if bullet_id not in drivers[metric]:
                                    drivers[metric].append(bullet_id)

        return dict(drivers)

    def generate_attribution_report(self) -> Dict:
        """Generate comprehensive attribution analysis report."""
        self.compute_attributions()

        top_contributors = self.get_top_contributors(10)
        strategy_synergies = self.get_strategy_synergies()
        performance_drivers = self.identify_performance_drivers()

        # Calculate overall statistics
        total_bullets = len(self.bullet_attributions)
        active_bullets = sum(1 for attr in self.bullet_attributions.values() if attr.usage_count > 0)
        avg_attribution_score = mean([attr.attribution_score for attr in self.bullet_attributions.values()])

        # Section-wise analysis
        section_performance = defaultdict(list)
        for attribution in self.bullet_attributions.values():
            section_performance[attribution.section].append(attribution.attribution_score)

        section_stats = {}
        for section, scores in section_performance.items():
            section_stats[section] = {
                'avg_score': mean(scores) if scores else 0,
                'bullet_count': len(scores),
                'top_bullet': max(scores) if scores else 0
            }

        return {
            'summary': {
                'total_bullets_analyzed': total_bullets,
                'active_bullets': active_bullets,
                'avg_attribution_score': avg_attribution_score,
                'total_usage_events': len(self.bullet_usage_history)
            },
            'top_contributors': [
                {
                    'bullet_id': attr.bullet_id,
                    'section': attr.section,
                    'content': attr.content[:100] + '...' if len(attr.content) > 100 else attr.content,
                    'attribution_score': attr.attribution_score,
                    'usage_count': attr.usage_count,
                    'success_rate': attr.success_rate,
                    'performance_impact': attr.performance_impact
                }
                for attr in top_contributors
            ],
            'strategy_synergies': [
                {
                    'bullet_a': corr.bullet_a,
                    'bullet_b': corr.bullet_b,
                    'synergy_score': corr.synergy_score,
                    'co_occurrence_count': corr.co_occurrence_count,
                    'joint_success_rate': corr.joint_success_rate
                }
                for corr in strategy_synergies
            ],
            'performance_drivers': performance_drivers,
            'section_analysis': section_stats,
            'generated_at': datetime.now().isoformat()
        }

    def export_analysis(self, file_path: Union[str, Path]) -> None:
        """Export complete attribution analysis to JSON file."""
        report = self.generate_attribution_report()

        # Add detailed data
        detailed_data = {
            'report': report,
            'detailed_attributions': {
                bullet_id: asdict(attribution)
                for bullet_id, attribution in self.bullet_attributions.items()
            },
            'detailed_correlations': {
                f"{pair[0]}_{pair[1]}": asdict(correlation)
                for pair, correlation in self.strategy_correlations.items()
            },
            'usage_history': self.bullet_usage_history,
            'export_timestamp': datetime.now().isoformat()
        }

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_analysis(cls, file_path: Union[str, Path]) -> AttributionAnalyzer:
        """Load attribution analysis from JSON file."""
        file_path = Path(file_path)

        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        analyzer = cls()

        # Restore usage history
        analyzer.bullet_usage_history = data.get('usage_history', [])

        # Restore attributions
        for bullet_id, attr_data in data.get('detailed_attributions', {}).items():
            analyzer.bullet_attributions[bullet_id] = BulletAttribution(**attr_data)

        # Restore correlations
        for pair_key, corr_data in data.get('detailed_correlations', {}).items():
            bullet_a, bullet_b = pair_key.split('_', 1)
            pair = (bullet_a, bullet_b)
            analyzer.strategy_correlations[pair] = StrategyCorrelation(**corr_data)

        return analyzer