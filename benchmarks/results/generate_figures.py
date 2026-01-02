"""
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
