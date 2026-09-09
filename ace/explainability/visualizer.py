"""
Visual Explainability Dashboard for ACE.

Creates interactive visualizations and dashboards to explain ACE behavior,
including playbook evolution, strategy effectiveness, and learning patterns.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .evolution_tracker import EvolutionTracker, PlaybookSnapshot, StrategyEvolution
from .attribution_analyzer import AttributionAnalyzer, BulletAttribution
from .interaction_tracer import InteractionTracer, RoleInteraction


class ExplainabilityVisualizer:
    """
    Creates visual explanations of ACE behavior and learning patterns.

    This class generates various types of visualizations to help understand
    how ACE systems evolve, which strategies are most effective, and how
    different components interact over time.

    Example:
        >>> visualizer = ExplainabilityVisualizer()
        >>>
        >>> # Create evolution timeline
        >>> visualizer.plot_playbook_evolution(evolution_tracker)
        >>>
        >>> # Create attribution analysis
        >>> visualizer.plot_bullet_attribution(attribution_analyzer)
        >>>
        >>> # Create interaction dashboard
        >>> visualizer.create_interaction_dashboard(interaction_tracer)
        >>>
        >>> # Generate comprehensive report
        >>> visualizer.generate_html_report(
        ...     evolution_tracker, attribution_analyzer, interaction_tracer,
        ...     output_path="ace_explanation.html"
        ... )
    """

    def __init__(self, style: str = 'seaborn', figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size for plots
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#7B8FA1',
            'background': '#F5F5F5'
        }

        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style if style in plt.style.available else 'default')

    def plot_playbook_evolution(
        self,
        evolution_tracker: EvolutionTracker,
        save_path: Optional[Union[str, Path]] = None,
        show_performance: bool = True
    ) -> Optional[str]:
        """
        Plot the evolution of the playbook over time.

        Args:
            evolution_tracker: EvolutionTracker instance with recorded data
            save_path: Path to save the plot
            show_performance: Whether to include performance metrics

        Returns:
            Path to saved plot or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._generate_text_plot(evolution_tracker, 'evolution')

        snapshots = evolution_tracker.snapshots
        if not snapshots:
            return None

        fig, axes = plt.subplots(2 if show_performance else 1, 1, figsize=self.figsize, sharex=True)
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.tolist() if axes.ndim > 0 else [axes]

        # Extract data
        epochs = [s.epoch for s in snapshots]
        steps = [s.step for s in snapshots]
        bullet_counts = [s.total_bullets for s in snapshots]
        section_counts = [len(s.sections) for s in snapshots]

        # Plot bullet evolution
        ax1 = axes[0]
        ax1.plot(range(len(snapshots)), bullet_counts,
                marker='o', color=self.colors['primary'], linewidth=2, label='Total Bullets')
        ax1.plot(range(len(snapshots)), section_counts,
                marker='s', color=self.colors['secondary'], linewidth=2, label='Sections')

        ax1.set_ylabel('Count')
        ax1.set_title('Playbook Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot performance evolution if requested
        if show_performance and len(axes) > 1:
            ax2 = axes[1]
            performance_metrics = {}

            for snapshot in snapshots:
                for metric, value in snapshot.performance_metrics.items():
                    if metric not in performance_metrics:
                        performance_metrics[metric] = []
                    performance_metrics[metric].append(value)

            for metric, values in performance_metrics.items():
                if len(values) == len(snapshots):
                    ax2.plot(range(len(snapshots)), values,
                            marker='o', label=metric.capitalize(), linewidth=2)

            ax2.set_ylabel('Performance')
            ax2.set_xlabel('Adaptation Steps')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return str(save_path)

        plt.show()
        return None

    def plot_bullet_attribution(
        self,
        attribution_analyzer: AttributionAnalyzer,
        top_n: int = 15,
        save_path: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        Plot bullet attribution analysis.

        Args:
            attribution_analyzer: AttributionAnalyzer instance
            top_n: Number of top bullets to show
            save_path: Path to save the plot

        Returns:
            Path to saved plot or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._generate_text_attribution(attribution_analyzer, top_n)

        top_bullets = attribution_analyzer.get_top_contributors(top_n)
        if not top_bullets:
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Attribution scores
        bullets = [f"{b.bullet_id[:8]}" for b in top_bullets]
        scores = [b.attribution_score for b in top_bullets]

        ax1.barh(bullets, scores, color=self.colors['primary'], alpha=0.7)
        ax1.set_xlabel('Attribution Score')
        ax1.set_title('Top Bullet Attribution Scores')
        ax1.grid(True, alpha=0.3)

        # 2. Performance impact vs usage
        usage_counts = [b.usage_count for b in top_bullets]
        impacts = [b.performance_impact for b in top_bullets]

        scatter = ax2.scatter(usage_counts, impacts,
                            c=scores, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Usage Count')
        ax2.set_ylabel('Performance Impact')
        ax2.set_title('Usage vs Performance Impact')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Attribution Score')

        # 3. Success rates
        success_rates = [b.success_rate for b in top_bullets]
        ax3.barh(bullets, success_rates, color=self.colors['accent'], alpha=0.7)
        ax3.set_xlabel('Success Rate')
        ax3.set_title('Bullet Success Rates')
        ax3.grid(True, alpha=0.3)

        # 4. Section distribution
        section_counts = {}
        for bullet in top_bullets:
            section = bullet.section
            section_counts[section] = section_counts.get(section, 0) + 1

        if section_counts:
            ax4.pie(section_counts.values(), labels=section_counts.keys(),
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('Top Bullets by Section')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return str(save_path)

        plt.show()
        return None

    def plot_strategy_lifespans(
        self,
        evolution_tracker: EvolutionTracker,
        save_path: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        Plot strategy lifespans and effectiveness over time.

        Args:
            evolution_tracker: EvolutionTracker instance
            save_path: Path to save the plot

        Returns:
            Path to saved plot or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._generate_text_lifespans(evolution_tracker)

        strategies = evolution_tracker.strategy_evolutions
        if not strategies:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Strategy timeline
        y_pos = 0
        strategy_positions = {}

        for bullet_id, evolution in strategies.items():
            start_step = evolution.birth_step
            end_step = evolution.death_step or max(s.step for s in evolution_tracker.snapshots)

            # Color based on effectiveness
            effectiveness = evolution.final_effectiveness_score
            if effectiveness > 0.5:
                color = self.colors['success']
            elif effectiveness > 0:
                color = self.colors['accent']
            else:
                color = self.colors['neutral']

            # Draw lifespan bar
            ax1.barh(y_pos, end_step - start_step, left=start_step,
                    height=0.8, color=color, alpha=0.7,
                    label=f"{bullet_id[:8]}" if y_pos < 10 else "")

            strategy_positions[bullet_id] = y_pos
            y_pos += 1

        ax1.set_ylabel('Strategies')
        ax1.set_title('Strategy Lifespans')
        ax1.grid(True, alpha=0.3)

        # Effectiveness over time
        effectiveness_data = {}
        for bullet_id, evolution in strategies.items():
            if evolution.helpful_progression:
                for timestamp, helpful in evolution.helpful_progression:
                    step = evolution.birth_step  # Simplified mapping
                    if step not in effectiveness_data:
                        effectiveness_data[step] = []
                    effectiveness_data[step].append(helpful)

        if effectiveness_data:
            steps = sorted(effectiveness_data.keys())
            avg_effectiveness = [np.mean(effectiveness_data[step]) for step in steps]

            ax2.plot(steps, avg_effectiveness, marker='o',
                    color=self.colors['primary'], linewidth=2)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Average Effectiveness')
            ax2.set_title('Strategy Effectiveness Over Time')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return str(save_path)

        plt.show()
        return None

    def create_interaction_heatmap(
        self,
        interaction_tracer: InteractionTracer,
        save_path: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        Create a heatmap of role interactions and decision patterns.

        Args:
            interaction_tracer: InteractionTracer instance
            save_path: Path to save the plot

        Returns:
            Path to saved plot or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._generate_text_interactions(interaction_tracer)

        interactions = interaction_tracer.interactions
        if not interactions:
            return None

        # Create interaction matrix
        role_activities = {
            'Generator': [],
            'Reflector': [],
            'Curator': []
        }

        for interaction in interactions:
            # Generator activity (number of bullets used)
            bullet_count = len(interaction.generator_output.get('bullet_ids', []))
            role_activities['Generator'].append(bullet_count)

            # Reflector activity (number of tags)
            tag_count = len(interaction.reflector_output.get('bullet_tags', []))
            role_activities['Reflector'].append(tag_count)

            # Curator activity (number of operations)
            op_count = len(interaction.curator_output.get('operations', []))
            role_activities['Curator'].append(op_count)

        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)

        data_matrix = np.array([
            role_activities['Generator'],
            role_activities['Reflector'],
            role_activities['Curator']
        ])

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks and labels
        ax.set_yticks(range(len(role_activities)))
        ax.set_yticklabels(role_activities.keys())
        ax.set_xlabel('Interaction Steps')
        ax.set_title('Role Activity Heatmap')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Activity Level')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return str(save_path)

        plt.show()
        return None

    def generate_html_report(
        self,
        evolution_tracker: Optional[EvolutionTracker] = None,
        attribution_analyzer: Optional[AttributionAnalyzer] = None,
        interaction_tracer: Optional[InteractionTracer] = None,
        output_path: Union[str, Path] = "ace_explainability_report.html",
        include_plots: bool = True
    ) -> str:
        """
        Generate a comprehensive HTML explainability report.

        Args:
            evolution_tracker: EvolutionTracker instance
            attribution_analyzer: AttributionAnalyzer instance
            interaction_tracer: InteractionTracer instance
            output_path: Path for the HTML report
            include_plots: Whether to include generated plots

        Returns:
            Path to the generated HTML report
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html_template()

        # Generate sections
        sections = []

        if evolution_tracker:
            evolution_section = self._generate_evolution_section(evolution_tracker, output_dir, include_plots)
            sections.append(evolution_section)

        if attribution_analyzer:
            attribution_section = self._generate_attribution_section(attribution_analyzer, output_dir, include_plots)
            sections.append(attribution_section)

        if interaction_tracer:
            interaction_section = self._generate_interaction_section(interaction_tracer, output_dir, include_plots)
            sections.append(interaction_section)

        # Combine sections
        content = "\n".join(sections)
        html_content = html_content.replace("{CONTENT}", content)
        html_content = html_content.replace("{TIMESTAMP}", datetime.now().isoformat())

        # Write HTML file
        with output_path.open('w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path)

    def _generate_html_template(self) -> str:
        """Generate the base HTML template."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ACE Explainability Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .section {
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #2E86AB;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2E86AB;
        }
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        .plot-container {
            text-align: center;
            margin: 2rem 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .insight-box {
            background: #e8f4fd;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .insight-box h4 {
            margin-top: 0;
            color: #0c5460;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 2rem;
            padding: 1rem;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 ACE Explainability Report</h1>
        <p>Understanding Agentic Context Engineering Behavior</p>
        <p><small>Generated on {TIMESTAMP}</small></p>
    </div>

    {CONTENT}

    <div class="footer">
        <p>Report generated by ACE Explainability Suite</p>
    </div>
</body>
</html>
        '''

    def _generate_evolution_section(self, tracker: EvolutionTracker, output_dir: Path, include_plots: bool) -> str:
        """Generate HTML section for evolution analysis."""
        summary = tracker.get_evolution_summary()
        lifespan_analysis = tracker.analyze_strategy_lifespans()

        section = f'''
        <div class="section">
            <h2>📈 Playbook Evolution Analysis</h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_strategies', 0)}</div>
                    <div class="metric-label">Total Strategies</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('survival_rate', 0):.1%}</div>
                    <div class="metric-label">Survival Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('bullet_growth', 0):+d}</div>
                    <div class="metric-label">Bullet Growth</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{lifespan_analysis.get('avg_lifespan', 0):.1f}</div>
                    <div class="metric-label">Avg Strategy Lifespan</div>
                </div>
            </div>
        '''

        if include_plots and MATPLOTLIB_AVAILABLE:
            plot_path = output_dir / "evolution_plot.png"
            self.plot_playbook_evolution(tracker, plot_path)
            section += f'''
            <div class="plot-container">
                <img src="{plot_path.name}" alt="Playbook Evolution Plot">
            </div>
            '''

        # Add insights
        patterns = tracker.identify_learning_patterns()
        if patterns:
            section += '''
            <div class="insight-box">
                <h4>🔍 Key Insights</h4>
            '''
            if patterns.get('rapid_additions'):
                section += f"<p><strong>Rapid Learning:</strong> Heavy strategy addition in epochs {patterns['rapid_additions']}</p>"
            if patterns.get('pruning_phases'):
                section += f"<p><strong>Strategy Pruning:</strong> Cleanup phases in epochs {patterns['pruning_phases']}</p>"
            if patterns.get('performance_jumps'):
                jumps = patterns['performance_jumps']
                if jumps:
                    section += f"<p><strong>Performance Breakthroughs:</strong> {len(jumps)} significant improvements detected</p>"
            section += '</div>'

        section += '</div>'
        return section

    def _generate_attribution_section(self, analyzer: AttributionAnalyzer, output_dir: Path, include_plots: bool) -> str:
        """Generate HTML section for attribution analysis."""
        report = analyzer.generate_attribution_report()
        top_contributors = report['top_contributors'][:10]

        section = f'''
        <div class="section">
            <h2>🎯 Strategy Attribution Analysis</h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['total_bullets_analyzed']}</div>
                    <div class="metric-label">Bullets Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['active_bullets']}</div>
                    <div class="metric-label">Active Bullets</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['avg_attribution_score']:.3f}</div>
                    <div class="metric-label">Avg Attribution Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(report['strategy_synergies'])}</div>
                    <div class="metric-label">Strategy Synergies</div>
                </div>
            </div>
        '''

        if include_plots and MATPLOTLIB_AVAILABLE:
            plot_path = output_dir / "attribution_plot.png"
            self.plot_bullet_attribution(analyzer, save_path=plot_path)
            section += f'''
            <div class="plot-container">
                <img src="{plot_path.name}" alt="Attribution Analysis Plot">
            </div>
            '''

        # Top contributors table
        if top_contributors:
            section += '''
            <h3>🏆 Top Contributing Strategies</h3>
            <table>
                <thead>
                    <tr>
                        <th>Bullet ID</th>
                        <th>Section</th>
                        <th>Attribution Score</th>
                        <th>Usage Count</th>
                        <th>Success Rate</th>
                        <th>Content Preview</th>
                    </tr>
                </thead>
                <tbody>
            '''
            for contributor in top_contributors:
                section += f'''
                <tr>
                    <td>{contributor['bullet_id'][:12]}</td>
                    <td>{contributor['section']}</td>
                    <td>{contributor['attribution_score']:.3f}</td>
                    <td>{contributor['usage_count']}</td>
                    <td>{contributor['success_rate']:.1%}</td>
                    <td>{contributor['content'][:50]}...</td>
                </tr>
                '''
            section += '</tbody></table>'

        section += '</div>'
        return section

    def _generate_interaction_section(self, tracer: InteractionTracer, output_dir: Path, include_plots: bool) -> str:
        """Generate HTML section for interaction analysis."""
        report = tracer.generate_interaction_report()

        section = f'''
        <div class="section">
            <h2>🔄 Role Interaction Analysis</h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['total_interactions']}</div>
                    <div class="metric-label">Total Interactions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['decision_chains_identified']}</div>
                    <div class="metric-label">Decision Chains</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['avg_chain_length']:.1f}</div>
                    <div class="metric-label">Avg Chain Length</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['feedback_loops_total']}</div>
                    <div class="metric-label">Feedback Loops</div>
                </div>
            </div>
        '''

        if include_plots and MATPLOTLIB_AVAILABLE:
            plot_path = output_dir / "interaction_heatmap.png"
            self.create_interaction_heatmap(tracer, plot_path)
            section += f'''
            <div class="plot-container">
                <img src="{plot_path.name}" alt="Role Interaction Heatmap">
            </div>
            '''

        section += '</div>'
        return section

    def _generate_text_plot(self, tracker: EvolutionTracker, plot_type: str) -> str:
        """Generate text-based visualization when matplotlib is not available."""
        summary = tracker.get_evolution_summary()

        text_viz = f"""
ACE Evolution Summary ({plot_type}):
{'='*40}
Total Strategies: {summary.get('total_strategies', 0)}
Alive Strategies: {summary.get('alive_strategies', 0)}
Dead Strategies: {summary.get('dead_strategies', 0)}
Survival Rate: {summary.get('survival_rate', 0):.1%}
Bullet Growth: {summary.get('bullet_growth', 0):+d}
        """
        return text_viz

    def _generate_text_attribution(self, analyzer: AttributionAnalyzer, top_n: int) -> str:
        """Generate text-based attribution analysis."""
        top_bullets = analyzer.get_top_contributors(top_n)

        text_viz = "Top Contributing Bullets:\n" + "="*30 + "\n"
        for i, bullet in enumerate(top_bullets[:10], 1):
            text_viz += f"{i:2d}. {bullet.bullet_id[:12]} | Score: {bullet.attribution_score:.3f} | Usage: {bullet.usage_count}\n"

        return text_viz

    def _generate_text_lifespans(self, tracker: EvolutionTracker) -> str:
        """Generate text-based lifespan analysis."""
        analysis = tracker.analyze_strategy_lifespans()

        text_viz = f"""
Strategy Lifespan Analysis:
{'='*30}
Average Lifespan: {analysis.get('avg_lifespan', 0):.1f} steps
Min Lifespan: {analysis.get('min_lifespan', 0)} steps
Max Lifespan: {analysis.get('max_lifespan', 0)} steps
Long-lived Strategies: {len(analysis.get('long_lived_strategies', []))}
        """
        return text_viz

    def _generate_text_interactions(self, tracer: InteractionTracer) -> str:
        """Generate text-based interaction analysis."""
        report = tracer.generate_interaction_report()

        text_viz = f"""
Role Interaction Summary:
{'='*30}
Total Interactions: {report['summary']['total_interactions']}
Decision Chains: {report['summary']['decision_chains_identified']}
Avg Chain Length: {report['summary']['avg_chain_length']:.1f}
Feedback Loops: {report['summary']['feedback_loops_total']}
        """
        return text_viz