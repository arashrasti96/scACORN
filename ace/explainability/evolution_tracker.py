"""
Playbook Evolution Tracker for ACE explainability.

Tracks how playbooks evolve over time, including bullet lifecycles,
strategy emergence patterns, and learning convergence behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from ..playbook import Bullet, Playbook
from ..delta import DeltaBatch, DeltaOperation


@dataclass
class PlaybookSnapshot:
    """Snapshot of playbook state at a specific point in time."""

    timestamp: str
    epoch: int
    step: int
    total_bullets: int
    sections: Dict[str, int]  # section -> bullet count
    bullet_stats: Dict[str, int]  # helpful/harmful/neutral totals
    bullets: Dict[str, Dict[str, Union[str, int]]]  # bullet_id -> bullet data
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    context: str = ""  # context about what triggered this snapshot

    @classmethod
    def from_playbook(
        cls,
        playbook: Playbook,
        epoch: int = 0,
        step: int = 0,
        performance_metrics: Optional[Dict[str, float]] = None,
        context: str = ""
    ) -> PlaybookSnapshot:
        """Create a snapshot from a playbook instance."""
        sections = {}
        bullets = {}

        for bullet in playbook.bullets():
            # Count bullets per section
            sections[bullet.section] = sections.get(bullet.section, 0) + 1

            # Store bullet data
            bullets[bullet.id] = {
                'section': bullet.section,
                'content': bullet.content,
                'helpful': bullet.helpful,
                'harmful': bullet.harmful,
                'neutral': bullet.neutral,
                'created_at': bullet.created_at,
                'updated_at': bullet.updated_at
            }

        stats = playbook.stats()
        bullet_stats = stats.get('tags', {})

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            epoch=epoch,
            step=step,
            total_bullets=len(bullets),
            sections=sections,
            bullet_stats=bullet_stats,
            bullets=bullets,
            performance_metrics=performance_metrics or {},
            context=context
        )


@dataclass
class BulletChange:
    """Represents a change to a specific bullet."""

    bullet_id: str
    operation: str  # ADD, UPDATE, TAG, REMOVE
    timestamp: str
    epoch: int
    step: int
    old_values: Dict[str, Union[str, int]] = field(default_factory=dict)
    new_values: Dict[str, Union[str, int]] = field(default_factory=dict)
    trigger_context: str = ""  # what caused this change


@dataclass
class StrategyEvolution:
    """Tracks the evolution of a specific strategy (bullet) over time."""

    bullet_id: str
    section: str
    content: str
    birth_epoch: int
    birth_step: int
    death_epoch: Optional[int] = None
    death_step: Optional[int] = None

    # Effectiveness tracking
    helpful_progression: List[Tuple[str, int]] = field(default_factory=list)  # (timestamp, count)
    harmful_progression: List[Tuple[str, int]] = field(default_factory=list)
    neutral_progression: List[Tuple[str, int]] = field(default_factory=list)

    # Usage tracking
    usage_contexts: List[str] = field(default_factory=list)
    performance_impact: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)

    @property
    def lifespan_steps(self) -> int:
        """Calculate how many steps this strategy survived."""
        if self.death_step is None:
            return -1  # Still alive
        return self.death_step - self.birth_step

    @property
    def final_effectiveness_score(self) -> float:
        """Calculate final effectiveness score (helpful - harmful) / total."""
        if not self.helpful_progression and not self.harmful_progression:
            return 0.0

        helpful = self.helpful_progression[-1][1] if self.helpful_progression else 0
        harmful = self.harmful_progression[-1][1] if self.harmful_progression else 0
        neutral = self.neutral_progression[-1][1] if self.neutral_progression else 0

        total = helpful + harmful + neutral
        if total == 0:
            return 0.0

        return (helpful - harmful) / total


class EvolutionTracker:
    """
    Tracks the evolution of ACE playbooks over time.

    This class monitors how playbooks change during adaptation,
    tracking bullet lifecycles, strategy emergence patterns,
    and overall learning dynamics.

    Example:
        >>> tracker = EvolutionTracker()
        >>>
        >>> # During ACE adaptation loop
        >>> for epoch in range(epochs):
        ...     for step, sample in enumerate(samples):
        ...         # ... ACE processing ...
        ...
        ...         # Track changes
        ...         tracker.record_delta(delta_batch, epoch, step, context="after_curator")
        ...         tracker.take_snapshot(playbook, epoch, step, metrics, "post_adaptation")
        >>>
        >>> # Analyze evolution
        >>> summary = tracker.get_evolution_summary()
        >>> strategy_lifespans = tracker.analyze_strategy_lifespans()
        >>> learning_patterns = tracker.identify_learning_patterns()
    """

    def __init__(self):
        self.snapshots: List[PlaybookSnapshot] = []
        self.bullet_changes: List[BulletChange] = []
        self.strategy_evolutions: Dict[str, StrategyEvolution] = {}
        self.active_bullets: Set[str] = set()

    def take_snapshot(
        self,
        playbook: Playbook,
        epoch: int,
        step: int,
        performance_metrics: Optional[Dict[str, float]] = None,
        context: str = ""
    ) -> PlaybookSnapshot:
        """Take a snapshot of the current playbook state."""
        snapshot = PlaybookSnapshot.from_playbook(
            playbook, epoch, step, performance_metrics, context
        )
        self.snapshots.append(snapshot)

        # Update strategy tracking
        self._update_strategy_tracking(snapshot)

        return snapshot

    def record_delta(
        self,
        delta: DeltaBatch,
        epoch: int,
        step: int,
        context: str = ""
    ) -> None:
        """Record a delta batch and track individual bullet changes."""
        timestamp = datetime.now(timezone.utc).isoformat()

        for operation in delta.operations:
            change = BulletChange(
                bullet_id=operation.bullet_id or f"unknown_{len(self.bullet_changes)}",
                operation=operation.type.upper(),
                timestamp=timestamp,
                epoch=epoch,
                step=step,
                trigger_context=context
            )

            # Track specific changes based on operation type
            if operation.type.upper() == "ADD":
                change.new_values = {
                    'section': operation.section,
                    'content': operation.content or "",
                    **(operation.metadata or {})
                }
                self.active_bullets.add(change.bullet_id)

                # Start tracking this strategy
                if change.bullet_id not in self.strategy_evolutions:
                    self.strategy_evolutions[change.bullet_id] = StrategyEvolution(
                        bullet_id=change.bullet_id,
                        section=operation.section,
                        content=operation.content or "",
                        birth_epoch=epoch,
                        birth_step=step
                    )

            elif operation.type.upper() == "UPDATE":
                change.new_values = {}
                if operation.content:
                    change.new_values['content'] = operation.content
                if operation.metadata:
                    change.new_values.update(operation.metadata)

            elif operation.type.upper() == "TAG":
                change.new_values = operation.metadata or {}

            elif operation.type.upper() == "REMOVE":
                self.active_bullets.discard(change.bullet_id)
                # Mark strategy as dead
                if change.bullet_id in self.strategy_evolutions:
                    evolution = self.strategy_evolutions[change.bullet_id]
                    evolution.death_epoch = epoch
                    evolution.death_step = step

            self.bullet_changes.append(change)

    def _update_strategy_tracking(self, snapshot: PlaybookSnapshot) -> None:
        """Update strategy evolution tracking with latest snapshot data."""
        timestamp = snapshot.timestamp

        for bullet_id, bullet_data in snapshot.bullets.items():
            if bullet_id in self.strategy_evolutions:
                evolution = self.strategy_evolutions[bullet_id]

                # Update effectiveness progressions
                helpful = bullet_data.get('helpful', 0)
                harmful = bullet_data.get('harmful', 0)
                neutral = bullet_data.get('neutral', 0)

                if evolution.helpful_progression and evolution.helpful_progression[-1][1] != helpful:
                    evolution.helpful_progression.append((timestamp, helpful))
                elif not evolution.helpful_progression:
                    evolution.helpful_progression.append((timestamp, helpful))

                if evolution.harmful_progression and evolution.harmful_progression[-1][1] != harmful:
                    evolution.harmful_progression.append((timestamp, harmful))
                elif not evolution.harmful_progression:
                    evolution.harmful_progression.append((timestamp, harmful))

                if evolution.neutral_progression and evolution.neutral_progression[-1][1] != neutral:
                    evolution.neutral_progression.append((timestamp, neutral))
                elif not evolution.neutral_progression:
                    evolution.neutral_progression.append((timestamp, neutral))

                # Track performance impact if metrics available
                if snapshot.performance_metrics:
                    evolution.performance_impact.append((timestamp, snapshot.performance_metrics.copy()))

    def get_evolution_summary(self) -> Dict[str, Union[int, float, Dict]]:
        """Get a high-level summary of playbook evolution."""
        if not self.snapshots:
            return {}

        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]

        # Calculate growth metrics
        bullet_growth = last_snapshot.total_bullets - first_snapshot.total_bullets
        section_growth = len(last_snapshot.sections) - len(first_snapshot.sections)

        # Calculate strategy metrics
        total_strategies = len(self.strategy_evolutions)
        dead_strategies = sum(1 for s in self.strategy_evolutions.values() if s.death_epoch is not None)
        alive_strategies = total_strategies - dead_strategies

        # Calculate effectiveness distribution
        effectiveness_scores = [s.final_effectiveness_score for s in self.strategy_evolutions.values()]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0

        # Performance trend
        performance_trend = {}
        if self.snapshots and self.snapshots[0].performance_metrics:
            for metric in self.snapshots[0].performance_metrics:
                first_val = first_snapshot.performance_metrics.get(metric, 0)
                last_val = last_snapshot.performance_metrics.get(metric, 0)
                performance_trend[metric] = {
                    'start': first_val,
                    'end': last_val,
                    'change': last_val - first_val,
                    'relative_change': (last_val - first_val) / first_val if first_val != 0 else 0
                }

        return {
            'total_snapshots': len(self.snapshots),
            'total_changes': len(self.bullet_changes),
            'bullet_growth': bullet_growth,
            'section_growth': section_growth,
            'total_strategies': total_strategies,
            'alive_strategies': alive_strategies,
            'dead_strategies': dead_strategies,
            'survival_rate': alive_strategies / total_strategies if total_strategies > 0 else 0,
            'avg_effectiveness': avg_effectiveness,
            'performance_trends': performance_trend,
            'change_operations': {
                op: sum(1 for c in self.bullet_changes if c.operation == op)
                for op in ['ADD', 'UPDATE', 'TAG', 'REMOVE']
            }
        }

    def analyze_strategy_lifespans(self) -> Dict[str, Union[List, Dict]]:
        """Analyze strategy lifespans and survival patterns."""
        lifespans = []
        effectiveness_by_lifespan = {}

        for evolution in self.strategy_evolutions.values():
            lifespan = evolution.lifespan_steps
            effectiveness = evolution.final_effectiveness_score

            if lifespan >= 0:  # Only include dead strategies
                lifespans.append(lifespan)
                if lifespan not in effectiveness_by_lifespan:
                    effectiveness_by_lifespan[lifespan] = []
                effectiveness_by_lifespan[lifespan].append(effectiveness)

        # Calculate statistics
        avg_lifespan = sum(lifespans) / len(lifespans) if lifespans else 0
        min_lifespan = min(lifespans) if lifespans else 0
        max_lifespan = max(lifespans) if lifespans else 0

        # Effectiveness by lifespan
        lifespan_effectiveness = {}
        for lifespan, scores in effectiveness_by_lifespan.items():
            lifespan_effectiveness[lifespan] = {
                'avg_effectiveness': sum(scores) / len(scores),
                'count': len(scores)
            }

        return {
            'lifespans': lifespans,
            'avg_lifespan': avg_lifespan,
            'min_lifespan': min_lifespan,
            'max_lifespan': max_lifespan,
            'effectiveness_by_lifespan': lifespan_effectiveness,
            'long_lived_strategies': [
                evolution.bullet_id for evolution in self.strategy_evolutions.values()
                if evolution.lifespan_steps > avg_lifespan or evolution.death_epoch is None
            ]
        }

    def identify_learning_patterns(self) -> Dict[str, Union[List, Dict]]:
        """Identify patterns in learning behavior."""
        patterns = {
            'rapid_additions': [],  # Epochs with many ADD operations
            'pruning_phases': [],   # Epochs with many REMOVE operations
            'refinement_phases': [], # Epochs with many UPDATE operations
            'convergence_points': [], # Where changes slow down
            'performance_jumps': []   # Significant performance improvements
        }

        # Group changes by epoch
        changes_by_epoch = {}
        for change in self.bullet_changes:
            epoch = change.epoch
            if epoch not in changes_by_epoch:
                changes_by_epoch[epoch] = {'ADD': 0, 'UPDATE': 0, 'TAG': 0, 'REMOVE': 0}
            changes_by_epoch[epoch][change.operation] += 1

        # Identify pattern epochs
        for epoch, counts in changes_by_epoch.items():
            total_changes = sum(counts.values())

            if counts['ADD'] > total_changes * 0.6:  # 60% additions
                patterns['rapid_additions'].append(epoch)

            if counts['REMOVE'] > total_changes * 0.4:  # 40% removals
                patterns['pruning_phases'].append(epoch)

            if counts['UPDATE'] > total_changes * 0.5:  # 50% updates
                patterns['refinement_phases'].append(epoch)

            if total_changes < 2:  # Very few changes
                patterns['convergence_points'].append(epoch)

        # Identify performance jumps
        if len(self.snapshots) > 1:
            for i in range(1, len(self.snapshots)):
                prev_snap = self.snapshots[i-1]
                curr_snap = self.snapshots[i]

                for metric in prev_snap.performance_metrics:
                    if metric in curr_snap.performance_metrics:
                        prev_val = prev_snap.performance_metrics[metric]
                        curr_val = curr_snap.performance_metrics[metric]

                        if prev_val > 0 and (curr_val - prev_val) / prev_val > 0.1:  # 10% improvement
                            patterns['performance_jumps'].append({
                                'epoch': curr_snap.epoch,
                                'step': curr_snap.step,
                                'metric': metric,
                                'improvement': curr_val - prev_val,
                                'relative_improvement': (curr_val - prev_val) / prev_val
                            })

        return patterns

    def get_timeline_data(self) -> Dict[str, Any]:
        """Get complete evolution timeline data."""
        return {
            'snapshots': [asdict(snapshot) for snapshot in self.snapshots],
            'bullet_changes': [asdict(change) for change in self.bullet_changes],
            'strategy_evolutions': {
                bullet_id: asdict(evolution)
                for bullet_id, evolution in self.strategy_evolutions.items()
            },
            'summary': self.get_evolution_summary(),
            'lifespan_analysis': self.analyze_strategy_lifespans(),
            'learning_patterns': self.identify_learning_patterns(),
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }

    def export_timeline(self, file_path: Union[str, Path]) -> None:
        """Export complete evolution timeline to JSON file."""
        timeline_data = self.get_timeline_data()
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_timeline(cls, file_path: Union[str, Path]) -> EvolutionTracker:
        """Load evolution timeline from JSON file."""
        file_path = Path(file_path)

        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        tracker = cls()

        # Reconstruct snapshots
        for snapshot_data in data.get('snapshots', []):
            tracker.snapshots.append(PlaybookSnapshot(**snapshot_data))

        # Reconstruct bullet changes
        for change_data in data.get('bullet_changes', []):
            tracker.bullet_changes.append(BulletChange(**change_data))

        # Reconstruct strategy evolutions
        for bullet_id, evolution_data in data.get('strategy_evolutions', {}).items():
            tracker.strategy_evolutions[bullet_id] = StrategyEvolution(**evolution_data)

        # Reconstruct active bullets
        tracker.active_bullets = set(
            bullet_id for bullet_id, evolution in tracker.strategy_evolutions.items()
            if evolution.death_epoch is None
        )

        return tracker