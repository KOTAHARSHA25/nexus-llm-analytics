#!/usr/bin/env python3
"""Check what the optimizer generates for time series data"""

from src.backend.utils.data_optimizer import DataOptimizer

opt = DataOptimizer()
result = opt.optimize_for_llm('data/samples/sales_timeseries.json')

print('='*80)
print('OPTIMIZER OUTPUT FOR TIME SERIES DATA')
print('='*80)
print(f"Total rows in dataset: {result['total_rows']}")
print(f"Sample rows sent to LLM: {len(result.get('sample', []))}")
print('\n' + '='*80)
print('PREVIEW TEXT:')
print('='*80)
print(result['preview'])
