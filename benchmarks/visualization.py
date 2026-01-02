"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - RESEARCH VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Phase 4 Enhancement: Research-quality visualization generation.

Generates:
1. Baseline Comparison Bar Charts
2. Ablation Study Waterfall Charts
3. Quality vs Latency Tradeoff Scatter Plots
4. Component Impact Radar Charts
5. Performance Timeline Plots
6. Statistical Distribution Histograms

All visualizations are generated in ASCII for terminal display and
can export data for matplotlib/seaborn rendering.

Version: 1.0.0
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import statistics


@dataclass
class ChartData:
    """Data structure for chart export"""
    chart_type: str
    title: str
    data: Dict[str, Any]
    labels: List[str]
    values: List[float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "data": self.data,
            "labels": self.labels,
            "values": self.values,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ASCIIChart:
    """Generate ASCII charts for terminal display"""
    
    @staticmethod
    def horizontal_bar(
        labels: List[str],
        values: List[float],
        title: str = "",
        max_width: int = 50,
        value_format: str = "{:.2f}"
    ) -> str:
        """
        Generate horizontal bar chart.
        
        Example:
        ┌─────────────────────────────────────────┐
        │ Baseline Comparison                     │
        ├─────────────────────────────────────────┤
        │ Full System    ████████████████████ 0.92│
        │ No Review      ████████████████     0.78│
        │ No RAG         ███████████████      0.75│
        │ Minimal        ██████████           0.55│
        └─────────────────────────────────────────┘
        """
        if not labels or not values:
            return "No data to display"
        
        max_label_len = max(len(l) for l in labels)
        max_val = max(values) if values else 1
        
        lines = []
        
        # Header
        total_width = max_label_len + max_width + 10
        lines.append("┌" + "─" * total_width + "┐")
        if title:
            title_padded = f" {title}".ljust(total_width)
            lines.append("│" + title_padded + "│")
            lines.append("├" + "─" * total_width + "┤")
        
        # Bars
        for label, value in zip(labels, values):
            bar_length = int((value / max_val) * max_width) if max_val > 0 else 0
            bar = "█" * bar_length
            label_padded = label.ljust(max_label_len)
            value_str = value_format.format(value)
            line = f"│ {label_padded} {bar.ljust(max_width)} {value_str} │"
            lines.append(line)
        
        lines.append("└" + "─" * total_width + "┘")
        
        return "\n".join(lines)
    
    @staticmethod
    def comparison_chart(
        labels: List[str],
        values_a: List[float],
        values_b: List[float],
        title: str = "",
        legend: Tuple[str, str] = ("System A", "System B")
    ) -> str:
        """
        Generate side-by-side comparison chart.
        
        Example:
        ┌────────────────────────────────────────────────┐
        │ Quality Comparison                             │
        │ █ Full System  ░ Baseline                      │
        ├────────────────────────────────────────────────┤
        │ Simple   █████████████████████ 0.95            │
        │          ░░░░░░░░░░░░░░░░░    0.82             │
        │ Medium   ████████████████████ 0.90             │
        │          ░░░░░░░░░░░░░        0.68             │
        └────────────────────────────────────────────────┘
        """
        if not labels:
            return "No data to display"
        
        max_label_len = max(len(l) for l in labels)
        max_val = max(max(values_a), max(values_b)) if values_a and values_b else 1
        bar_width = 30
        
        lines = []
        total_width = max_label_len + bar_width + 20
        
        # Header
        lines.append("┌" + "─" * total_width + "┐")
        if title:
            lines.append("│ " + title.ljust(total_width - 1) + "│")
            legend_line = f"│ █ {legend[0]}  ░ {legend[1]}".ljust(total_width + 1) + "│"
            lines.append(legend_line)
            lines.append("├" + "─" * total_width + "┤")
        
        # Data rows
        for i, label in enumerate(labels):
            val_a = values_a[i] if i < len(values_a) else 0
            val_b = values_b[i] if i < len(values_b) else 0
            
            bar_a = int((val_a / max_val) * bar_width) if max_val > 0 else 0
            bar_b = int((val_b / max_val) * bar_width) if max_val > 0 else 0
            
            lines.append(f"│ {label.ljust(max_label_len)} {'█' * bar_a}".ljust(total_width + 1) + f" {val_a:.2f} │")
            lines.append(f"│ {''.ljust(max_label_len)} {'░' * bar_b}".ljust(total_width + 1) + f" {val_b:.2f} │")
        
        lines.append("└" + "─" * total_width + "┘")
        
        return "\n".join(lines)
    
    @staticmethod
    def waterfall_chart(
        components: List[str],
        impacts: List[float],
        base_value: float,
        title: str = "Impact Analysis"
    ) -> str:
        """
        Generate waterfall chart showing cumulative impact.
        
        Useful for ablation study visualization.
        """
        lines = []
        width = 60
        
        lines.append("┌" + "─" * width + "┐")
        lines.append("│ " + title.ljust(width - 1) + "│")
        lines.append("├" + "─" * width + "┤")
        
        current = base_value
        max_val = base_value + sum(abs(i) for i in impacts)
        
        for component, impact in zip(components, impacts):
            bar_pos = int((current / max_val) * 40)
            impact_bar = int((abs(impact) / max_val) * 40)
            
            if impact >= 0:
                bar = " " * bar_pos + "▓" * impact_bar
                symbol = "+"
            else:
                bar = " " * max(0, bar_pos - impact_bar) + "░" * impact_bar
                symbol = "-"
            
            line = f"│ {component[:15].ljust(15)} {bar[:40].ljust(40)} {symbol}{abs(impact):.2f} │"
            lines.append(line)
            current += impact
        
        lines.append("├" + "─" * width + "┤")
        final_bar = int((current / max_val) * 40)
        lines.append(f"│ {'TOTAL'.ljust(15)} {'█' * final_bar}".ljust(width + 1) + f" {current:.2f} │")
        lines.append("└" + "─" * width + "┘")
        
        return "\n".join(lines)


class ResearchVisualizer:
    """
    Generate research-quality visualizations from benchmark results.
    """
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts: List[ChartData] = []
    
    def generate_baseline_comparison(
        self,
        comparison_results: List[Dict]
    ) -> ChartData:
        """Generate baseline comparison visualization"""
        labels = [r["baseline_name"] for r in comparison_results]
        full_scores = [r["full_system_score"] for r in comparison_results]
        baseline_scores = [r["baseline_score"] for r in comparison_results]
        improvements = [r["improvement_percent"] for r in comparison_results]
        
        chart = ChartData(
            chart_type="grouped_bar",
            title="Full System vs Baselines",
            data={
                "full_system": full_scores,
                "baselines": baseline_scores,
                "improvements": improvements
            },
            labels=labels,
            values=full_scores,
            metadata={
                "y_axis": "Quality Score",
                "subtitle": "Quality comparison across configurations",
                "generated_at": datetime.now().isoformat()
            }
        )
        
        self.charts.append(chart)
        return chart
    
    def generate_ablation_waterfall(
        self,
        ablation_results: Dict
    ) -> ChartData:
        """Generate ablation study waterfall chart"""
        base_quality = ablation_results.get("full_system", {}).get("quality", 1.0)
        
        components = []
        impacts = []
        
        for key, data in ablation_results.items():
            if key.startswith("without_"):
                component = key.replace("without_", "").replace("_", " ").title()
                impact = data.get("quality_impact", 0)
                components.append(component)
                impacts.append(-impact)  # Negative because removal hurts quality
        
        chart = ChartData(
            chart_type="waterfall",
            title="Component Ablation Impact",
            data={
                "base_value": base_quality,
                "components": components,
                "impacts": impacts
            },
            labels=components,
            values=impacts,
            metadata={
                "interpretation": "Negative values show quality loss when component removed",
                "generated_at": datetime.now().isoformat()
            }
        )
        
        self.charts.append(chart)
        return chart
    
    def generate_quality_latency_tradeoff(
        self,
        results: List[Dict]
    ) -> ChartData:
        """Generate quality vs latency scatter plot data"""
        points = []
        
        for r in results:
            points.append({
                "name": r.get("baseline_name", "Unknown"),
                "quality": r.get("baseline_score", 0),
                "latency": r.get("baseline_latency", 0)
            })
            
            # Add full system point
            if "full_system_score" in r:
                points.append({
                    "name": "Full System",
                    "quality": r["full_system_score"],
                    "latency": r.get("full_system_latency", 0)
                })
                break  # Only add once
        
        chart = ChartData(
            chart_type="scatter",
            title="Quality vs Latency Tradeoff",
            data={"points": points},
            labels=[p["name"] for p in points],
            values=[p["quality"] for p in points],
            metadata={
                "x_axis": "Latency (seconds)",
                "y_axis": "Quality Score",
                "ideal_position": "upper-left",
                "generated_at": datetime.now().isoformat()
            }
        )
        
        self.charts.append(chart)
        return chart
    
    def generate_component_radar(
        self,
        component_ranking: List[Tuple[str, float]]
    ) -> ChartData:
        """Generate radar chart data for component importance"""
        components = [c[0] for c in component_ranking]
        importance = [c[1] for c in component_ranking]
        
        # Normalize to 0-1 scale
        max_imp = max(importance) if importance else 1
        normalized = [i / max_imp for i in importance]
        
        chart = ChartData(
            chart_type="radar",
            title="Component Importance",
            data={
                "raw_importance": importance,
                "normalized": normalized
            },
            labels=components,
            values=normalized,
            metadata={
                "interpretation": "Higher values = more important component",
                "generated_at": datetime.now().isoformat()
            }
        )
        
        self.charts.append(chart)
        return chart
    
    def generate_statistical_summary(
        self,
        scores: List[float],
        name: str = "System Scores"
    ) -> ChartData:
        """Generate statistical summary visualization data"""
        if not scores:
            scores = [0]
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        summary = {
            "n": n,
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if n > 1 else 0,
            "min": min(scores),
            "max": max(scores),
            "median": statistics.median(scores),
            "q1": sorted_scores[int(0.25 * n)] if n > 1 else sorted_scores[0],
            "q3": sorted_scores[int(0.75 * n)] if n > 1 else sorted_scores[0],
            "iqr": None
        }
        summary["iqr"] = summary["q3"] - summary["q1"]
        
        # Create histogram buckets
        bucket_count = 10
        bucket_size = (summary["max"] - summary["min"]) / bucket_count if summary["max"] > summary["min"] else 1
        buckets = [0] * bucket_count
        
        for score in scores:
            bucket_idx = min(int((score - summary["min"]) / bucket_size), bucket_count - 1)
            buckets[bucket_idx] += 1
        
        bucket_labels = [
            f"{summary['min'] + i * bucket_size:.2f}-{summary['min'] + (i+1) * bucket_size:.2f}"
            for i in range(bucket_count)
        ]
        
        chart = ChartData(
            chart_type="histogram",
            title=f"Distribution: {name}",
            data={
                "summary": summary,
                "histogram": {"labels": bucket_labels, "counts": buckets}
            },
            labels=bucket_labels,
            values=buckets,
            metadata={
                "generated_at": datetime.now().isoformat()
            }
        )
        
        self.charts.append(chart)
        return chart
    
    def print_ascii_report(
        self,
        comparison_results: List[Dict],
        ablation_results: Dict
    ) -> str:
        """Generate full ASCII report"""
        lines = []
        
        # Header
        lines.append("═" * 70)
        lines.append("  NEXUS LLM ANALYTICS - BENCHMARK VISUALIZATION REPORT  ")
        lines.append("═" * 70)
        lines.append("")
        
        # Baseline comparison chart
        if comparison_results:
            labels = [r["baseline_name"][:20] for r in comparison_results]
            full_scores = [r["full_system_score"] for r in comparison_results]
            baseline_scores = [r["baseline_score"] for r in comparison_results]
            
            lines.append("1. BASELINE COMPARISON (Quality Scores)")
            lines.append("-" * 70)
            lines.append(ASCIIChart.horizontal_bar(
                labels, 
                full_scores,
                title="Full System Scores",
                value_format="{:.3f}"
            ))
            lines.append("")
            lines.append(ASCIIChart.horizontal_bar(
                labels,
                baseline_scores,
                title="Baseline Scores",
                value_format="{:.3f}"
            ))
            lines.append("")
        
        # Improvement percentages
        if comparison_results:
            improvements = [(r["baseline_name"][:15], r["improvement_percent"]) for r in comparison_results]
            improvements.sort(key=lambda x: x[1], reverse=True)
            
            lines.append("2. IMPROVEMENT OVER BASELINES (%)")
            lines.append("-" * 70)
            lines.append(ASCIIChart.horizontal_bar(
                [i[0] for i in improvements],
                [i[1] for i in improvements],
                title="Improvement Percentage",
                value_format="{:.1f}%"
            ))
            lines.append("")
        
        # Ablation impact
        if ablation_results:
            base_quality = ablation_results.get("full_system", {}).get("quality", 1.0)
            components = []
            impacts = []
            
            for key, data in ablation_results.items():
                if key.startswith("without_"):
                    component = key.replace("without_", "").title()
                    impact = data.get("quality_impact", 0)
                    components.append(component)
                    impacts.append(impact)
            
            lines.append("3. COMPONENT ABLATION IMPACT")
            lines.append("-" * 70)
            lines.append(ASCIIChart.waterfall_chart(
                components,
                [-i for i in impacts],  # Negative because removal hurts
                base_quality,
                title="Quality Impact When Component Removed"
            ))
            lines.append("")
        
        # Summary statistics
        if comparison_results:
            lines.append("4. SUMMARY STATISTICS")
            lines.append("-" * 70)
            avg_improvement = statistics.mean([r["improvement_percent"] for r in comparison_results])
            avg_latency = statistics.mean([r["latency_overhead_percent"] for r in comparison_results])
            
            lines.append(f"  Average Improvement:     {avg_improvement:>8.2f}%")
            lines.append(f"  Average Latency Overhead:{avg_latency:>8.2f}%")
            lines.append(f"  Improvement/Latency:     {avg_improvement/max(1,avg_latency):>8.2f}x")
            lines.append("")
        
        lines.append("═" * 70)
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("═" * 70)
        
        return "\n".join(lines)
    
    def export_for_matplotlib(self, filepath: str = None) -> str:
        """Export all chart data for matplotlib rendering"""
        export_data = {
            "charts": [c.to_dict() for c in self.charts],
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "chart_count": len(self.charts)
            }
        }
        
        if filepath is None:
            filepath = self.output_dir / "visualization_data.json"
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)
    
    def generate_matplotlib_script(self) -> str:
        """Generate Python script for matplotlib visualization"""
        script = '''"""
Auto-generated matplotlib visualization script.
Run this to generate publication-quality figures.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('benchmarks/results/visualization_data.json') as f:
    data = json.load(f)

charts = data['charts']

fig_num = 1
for chart in charts:
    plt.figure(fig_num, figsize=(10, 6))
    
    if chart['chart_type'] == 'grouped_bar':
        # Grouped bar chart
        labels = chart['labels']
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width/2, chart['data']['full_system'], width, label='Full System')
        bars2 = ax.bar(x + width/2, chart['data']['baselines'], width, label='Baseline')
        
        ax.set_ylabel('Quality Score')
        ax.set_title(chart['title'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
    elif chart['chart_type'] == 'scatter':
        # Scatter plot
        points = chart['data']['points']
        x = [p['latency'] for p in points]
        y = [p['quality'] for p in points]
        labels = [p['name'] for p in points]
        
        plt.scatter(x, y, s=100)
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]))
        
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Quality Score')
        plt.title(chart['title'])
        
    elif chart['chart_type'] == 'radar':
        # Radar chart
        labels = chart['labels']
        values = chart['values']
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(chart['title'])
    
    elif chart['chart_type'] == 'histogram':
        # Histogram
        plt.bar(range(len(chart['values'])), chart['values'])
        plt.xticks(range(len(chart['labels'])), chart['labels'], rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title(chart['title'])
    
    plt.tight_layout()
    plt.savefig(f'benchmarks/results/figure_{fig_num}.png', dpi=300)
    fig_num += 1

plt.show()
print(f"Generated {fig_num - 1} figures")
'''
        
        script_path = self.output_dir / "generate_figures.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        return str(script_path)


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'ResearchVisualizer',
    'ChartData',
    'ASCIIChart'
]
