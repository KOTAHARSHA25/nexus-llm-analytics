"""
GPU Performance Testing Suite
Tests project improvements with cloud GPU resources
"""

import asyncio
import time
import json
from pathlib import Path
from statistics import mean

# Configuration for different test environments
ENVIRONMENTS = {
    "local_cpu": {"provider": "Local PC", "hardware": "CPU 15.7GB RAM"},
    "colab_t4": {"provider": "Google Colab", "hardware": "Tesla T4 GPU"},
    "colab_a100": {"provider": "Google Colab Pro", "hardware": "A100 GPU"},
    "kaggle_p100": {"provider": "Kaggle", "hardware": "Tesla P100 GPU"},
    "paperspace_m4000": {"provider": "Paperspace", "hardware": "M4000 GPU"}
}

# Models to test
MODELS_TO_TEST = [
    "tinyllama:latest",
    "phi3:mini",
    "llama3.1:8b",
    "mistral:7b"
]

# Quick test queries (representative sample)
TEST_QUERIES = [
    ("1.json", "what is the name", "simple"),
    ("sales_data.csv", "show products price over 100", "moderate"),
    ("test_sales_monthly.csv", "calculate average sales", "complex")
]

async def benchmark_environment(env_name):
    """Benchmark current environment performance"""
    from backend.services.analysis_service import AnalysisService
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {ENVIRONMENTS[env_name]['provider']}")
    print(f"Hardware: {ENVIRONMENTS[env_name]['hardware']}")
    print(f"{'='*60}\n")
    
    service = AnalysisService()
    results = []
    
    for filename, query, complexity in TEST_QUERIES:
        print(f"[{complexity:8}] {query[:40]:40}", end=" ", flush=True)
        
        start = time.time()
        try:
            full_path = str(Path("data/uploads") / filename)
            context = {'filename': filename, 'filepath': full_path}
            result = await service.analyze(query, context)
            elapsed = time.time() - start
            
            success = result.get("success", False)
            status = "✓" if success else "✗"
            print(f"[{status}] {elapsed:.1f}s")
            
            results.append({
                "query": query,
                "complexity": complexity,
                "success": success,
                "time": elapsed
            })
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"[✗] {elapsed:.1f}s - {str(e)[:30]}")
            results.append({
                "query": query,
                "complexity": complexity,
                "success": False,
                "time": elapsed,
                "error": str(e)
            })
    
    # Calculate metrics
    successful = [r for r in results if r["success"]]
    accuracy = (len(successful) / len(results) * 100) if results else 0
    avg_time = mean([r["time"] for r in successful]) if successful else 0
    
    print(f"\n{'='*60}")
    print(f"Results: {len(successful)}/{len(results)} passed ({accuracy:.1f}%)")
    print(f"Average Time: {avg_time:.1f}s")
    print(f"{'='*60}\n")
    
    return {
        "environment": env_name,
        "provider": ENVIRONMENTS[env_name]["provider"],
        "hardware": ENVIRONMENTS[env_name]["hardware"],
        "accuracy": accuracy,
        "avg_time": avg_time,
        "results": results
    }

async def compare_models(models_list):
    """Compare different LLM models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON TESTING")
    print("="*60 + "\n")
    
    comparison = []
    
    for model in models_list:
        print(f"\nTesting model: {model}")
        print("-" * 60)
        
        # Here you would switch the model in config
        # For now, just runs with current model
        benchmark = await benchmark_environment("local_cpu")
        benchmark["model"] = model
        comparison.append(benchmark)
    
    return comparison

def generate_improvement_report(benchmarks):
    """Generate report comparing different environments/models"""
    
    print("\n" + "="*80)
    print("PERFORMANCE IMPROVEMENT REPORT")
    print("="*80 + "\n")
    
    print("Environment Performance Comparison:")
    print("-" * 80)
    print(f"{'Environment':<25} | {'Accuracy':>10} | {'Avg Time':>12} | {'Speedup':>10}")
    print("-" * 80)
    
    baseline = benchmarks[0]["avg_time"] if benchmarks else 0
    
    for bench in benchmarks:
        speedup = baseline / bench["avg_time"] if bench["avg_time"] > 0 else 0
        print(f"{bench['provider']:<25} | {bench['accuracy']:>9.1f}% | "
              f"{bench['avg_time']:>10.1f}s | {speedup:>9.1f}x")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT:")
    print("="*80)
    
    # Find best environment
    best_time = min(benchmarks, key=lambda x: x["avg_time"])
    best_accuracy = max(benchmarks, key=lambda x: x["accuracy"])
    
    print(f"\n✓ Fastest Environment: {best_time['provider']}")
    print(f"  Average Response Time: {best_time['avg_time']:.1f}s")
    print(f"  Speedup vs baseline: {baseline/best_time['avg_time']:.1f}x")
    
    print(f"\n✓ Most Accurate: {best_accuracy['provider']}")
    print(f"  Accuracy: {best_accuracy['accuracy']:.1f}%")
    
    print("\n" + "="*80)
    
    # Save results
    output = {
        "benchmarks": benchmarks,
        "recommendations": {
            "fastest": best_time['provider'],
            "most_accurate": best_accuracy['provider'],
            "speedup_potential": f"{baseline/best_time['avg_time']:.1f}x"
        }
    }
    
    with open("gpu_performance_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to: gpu_performance_comparison.json\n")

async def main():
    """Main testing harness"""
    
    print("="*80)
    print("GPU PERFORMANCE TESTING & IMPROVEMENT SUITE")
    print("="*80)
    print("\nThis script helps you:")
    print("1. Benchmark your project on different GPU platforms")
    print("2. Compare LLM model performance")
    print("3. Identify optimization opportunities")
    print("\n" + "="*80)
    
    # Detect current environment
    import platform
    import subprocess
    
    try:
        gpu_check = subprocess.run(['nvidia-smi'], capture_output=True)
        has_gpu = gpu_check.returncode == 0
    except:
        has_gpu = False
    
    if has_gpu:
        print("\n✓ GPU detected! Running GPU benchmarks...")
        env = "colab_t4"  # Auto-detect if possible
    else:
        print("\n⚠ No GPU detected. Running CPU baseline...")
        env = "local_cpu"
    
    # Run benchmark
    results = await benchmark_environment(env)
    
    # Generate report
    generate_improvement_report([results])
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Upload this script to Google Colab with GPU enabled")
    print("2. Upload your project files as a dataset")
    print("3. Run the benchmarks and compare results")
    print("4. Test different models (phi3, llama3, mistral)")
    print("5. Identify which model + platform gives best results")
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())
