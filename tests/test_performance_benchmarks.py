"""
Performance & Resource Usage Benchmarks
========================================
Measures response times, memory usage, and identifies bottlenecks.
"""

import asyncio
import time
import sys
import psutil
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class PerformanceBenchmark:
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    async def benchmark_query(self, service, query, filename, expected_max_time=120):
        """Benchmark a single query"""
        mem_before = self.get_memory_usage()
        start = time.time()
        
        result = await service.analyze(query=query, context={'filename': filename})
        
        elapsed = time.time() - start
        mem_after = self.get_memory_usage()
        mem_delta = mem_after - mem_before
        
        return {
            'query': query,
            'filename': filename,
            'elapsed': elapsed,
            'memory_delta_mb': mem_delta,
            'within_target': elapsed < expected_max_time,
            'success': result.get('success', False)
        }

async def run_performance_tests():
    """Run performance benchmarks"""
    from backend.services.analysis_service import get_analysis_service
    
    print("=" * 80)
    print("PERFORMANCE & RESOURCE BENCHMARKS")
    print("=" * 80)
    print()
    
    service = get_analysis_service()
    bench = PerformanceBenchmark()
    
    # Test cases with expected max times
    test_cases = [
        {
            'name': 'Simple JSON Lookup',
            'query': 'what is the name',
            'filename': '1.json',
            'target_time': 60,  # Should complete in 60s
            'complexity': 'simple'
        },
        {
            'name': 'CSV Row Count',
            'query': 'how many rows are there',
            'filename': 'test_student_grades.csv',
            'target_time': 90,
            'complexity': 'medium'
        },
        {
            'name': 'CSV Average Calculation',
            'query': 'what is the average grade',
            'filename': 'test_student_grades.csv',
            'target_time': 120,
            'complexity': 'complex'
        },
        {
            'name': 'Large File Handling',
            'query': 'show me the first 5 rows',
            'filename': 'comprehensive_ecommerce.csv',
            'target_time': 90,
            'complexity': 'medium'
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {test['name']}")
        print(f"  Query: {test['query']}")
        print(f"  Target: <{test['target_time']}s")
        
        result = await bench.benchmark_query(
            service,
            test['query'],
            test['filename'],
            test['target_time']
        )
        
        results.append({**test, **result})
        
        status = '✅' if result['within_target'] else '⚠️'
        print(f"  {status} Time: {result['elapsed']:.2f}s | Memory: {result['memory_delta_mb']:.1f}MB")
        print()
        
        # Recovery break
        await asyncio.sleep(5)
    
    # Analysis
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    total_within_target = sum(1 for r in results if r['within_target'])
    print(f"Tests within target: {total_within_target}/{len(results)}")
    print()
    
    # Identify bottlenecks
    slow_tests = [r for r in results if not r['within_target']]
    if slow_tests:
        print("⚠️ Performance Bottlenecks:")
        for t in slow_tests:
            print(f"  - {t['name']}: {t['elapsed']:.2f}s (target: {t['target_time']}s)")
            overhead = t['elapsed'] - t['target_time']
            print(f"    Overhead: +{overhead:.2f}s (+{overhead/t['target_time']*100:.0f}%)")
    else:
        print("✅ All tests within performance targets")
    
    print()
    
    # Memory analysis
    avg_memory = sum(r['memory_delta_mb'] for r in results) / len(results)
    max_memory = max(r['memory_delta_mb'] for r in results)
    print(f"Memory Usage:")
    print(f"  Average per query: {avg_memory:.1f}MB")
    print(f"  Maximum per query: {max_memory:.1f}MB")
    
    # Save detailed results
    output_file = project_root / "test_performance_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': len(results),
                'within_target': total_within_target,
                'avg_time': sum(r['elapsed'] for r in results) / len(results),
                'avg_memory_mb': avg_memory,
                'max_memory_mb': max_memory
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results: {output_file}")
    
    return total_within_target == len(results)

if __name__ == "__main__":
    success = asyncio.run(run_performance_tests())
    sys.exit(0 if success else 1)
