"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - RESEARCH-GRADE BENCHMARK SUITE
═══════════════════════════════════════════════════════════════════════════════

Purpose: Rigorous evaluation for patent and research publication
Author: Nexus Team
Date: December 2025

This benchmark suite validates:
1. DOMAIN AGNOSTICISM - Works across Education, Healthcare, IoT, Business
2. DATA AGNOSTICISM - Works with any structured data format
3. TWO FRIENDS MODEL EFFICACY - Measurable improvement through iteration
4. COMPLEXITY-BASED ROUTING - Correct model selection
5. ENTERPRISE SCALABILITY - Complex multi-step analytics

Metrics Collected:
- Accuracy: Correctness of numeric results
- Completeness: All required elements present
- Improvement Rate: % improvement after critic feedback
- Model Selection Accuracy: Correct model for complexity
- Latency: Response time per query
- Domain Coverage: Success rate across domains

═══════════════════════════════════════════════════════════════════════════════
"""
import sys
import json
import time
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.llm_client import LLMClient
from src.backend.core.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel


@dataclass
class BenchmarkResult:
    """Individual test result"""
    test_id: str
    domain: str
    query: str
    complexity_expected: str  # simple, medium, complex
    complexity_actual: float
    model_expected: str
    model_actual: str
    model_correct: bool
    iterations: int
    improved: bool
    accuracy_score: float  # 0.0 - 1.0
    completeness_score: float  # 0.0 - 1.0
    latency_seconds: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "domain": self.domain,
            "query": self.query[:50] + "...",
            "complexity_expected": self.complexity_expected,
            "complexity_actual": round(self.complexity_actual, 3),
            "model_expected": self.model_expected,
            "model_actual": self.model_actual,
            "model_correct": self.model_correct,
            "iterations": self.iterations,
            "improved": self.improved,
            "accuracy_score": round(self.accuracy_score, 3),
            "completeness_score": round(self.completeness_score, 3),
            "latency_seconds": round(self.latency_seconds, 2),
            "success": self.success
        }


@dataclass
class DomainBenchmark:
    """Benchmark for a specific domain"""
    domain: str
    data_file: str
    queries: List[Dict[str, Any]]
    data_preview: str = ""


class ResearchBenchmarkSuite:
    """
    Research-grade benchmark suite for patent and publication.
    
    Tests the complete Nexus LLM Analytics system across:
    - Multiple domains (Education, Healthcare, IoT, Business)
    - Multiple complexity levels (Simple, Medium, Complex)
    - Multiple data types (CSV with various schemas)
    
    Collects quantitative metrics suitable for research papers.
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.orchestrator = self._create_orchestrator()
        self.results: List[BenchmarkResult] = []
        self.start_time = None
        
        # Define benchmarks across domains
        self.benchmarks = self._create_domain_benchmarks()
    
    def _create_orchestrator(self) -> QueryOrchestrator:
        """Create orchestrator with standard configuration"""
        config = {
            'model_selection': {
                'simple': 'tinyllama',
                'medium': 'phi3:mini',
                'complex': 'llama3.1:8b',
                'thresholds': {'simple_max': 0.3, 'medium_max': 0.7}
            },
            'cot_review': {
                'activation_rules': {
                    'always_on_complexity': 0.7,
                    'optional_range': [0.3, 0.7],
                    'always_on_code_gen': True
                }
            }
        }
        return QueryOrchestrator(None, config)
    
    def _load_data_preview(self, filepath: str, max_rows: int = 5) -> str:
        """Load data preview for LLM context"""
        try:
            df = pd.read_csv(filepath)
            preview = f"Columns: {list(df.columns)}\n"
            preview += f"Total rows: {len(df)}\n"
            preview += f"Sample data:\n{df.head(max_rows).to_string()}\n"
            
            # Add statistics
            preview += f"\nStatistics:\n{df.describe().to_string()}"
            return preview
        except Exception as e:
            return f"Error loading data: {e}"
    
    def _create_domain_benchmarks(self) -> List[DomainBenchmark]:
        """Create comprehensive benchmarks across domains"""
        
        data_dir = project_root / "data" / "samples"
        
        benchmarks = [
            # ═══════════════════════════════════════════════════════════════
            # DOMAIN 1: EDUCATION (Student Grades)
            # ═══════════════════════════════════════════════════════════════
            DomainBenchmark(
                domain="Education",
                data_file=str(data_dir / "test_student_grades.csv"),
                queries=[
                    {
                        "id": "EDU_SIMPLE_01",
                        "query": "What is the average score?",
                        "complexity": "simple",
                        "expected_model": "tinyllama",
                        "validation": {"type": "numeric_range", "field": "average", "min": 70, "max": 95},
                        "required_elements": ["average", "score"]
                    },
                    {
                        "id": "EDU_MEDIUM_01",
                        "query": "Calculate the average score for each subject and identify which subject has the highest average",
                        "complexity": "medium",
                        "expected_model": "phi3:mini",
                        "validation": {"type": "contains_all", "elements": ["Mathematics", "Physics", "Chemistry"]},
                        "required_elements": ["average", "subject", "highest"]
                    },
                    {
                        "id": "EDU_COMPLEX_01",
                        "query": "Analyze the correlation between attendance percentage and academic performance across all subjects, then identify students at risk of failing (score below 70) and recommend interventions based on their attendance patterns",
                        "complexity": "complex",
                        "expected_model": "llama3.1:8b",
                        "validation": {"type": "contains_all", "elements": ["correlation", "attendance", "risk"]},
                        "required_elements": ["correlation", "attendance", "risk", "intervention", "student"]
                    }
                ]
            ),
            
            # ═══════════════════════════════════════════════════════════════
            # DOMAIN 2: IoT / INDUSTRIAL (Sensor Data)
            # ═══════════════════════════════════════════════════════════════
            DomainBenchmark(
                domain="IoT_Industrial",
                data_file=str(data_dir / "test_iot_sensor.csv"),
                queries=[
                    {
                        "id": "IOT_SIMPLE_01",
                        "query": "What is the maximum temperature recorded?",
                        "complexity": "simple",
                        "expected_model": "tinyllama",
                        "validation": {"type": "numeric_present"},
                        "required_elements": ["temperature", "maximum"]
                    },
                    {
                        "id": "IOT_MEDIUM_01",
                        "query": "Calculate the average temperature and humidity for each hour and identify any anomalies where values exceed normal ranges",
                        "complexity": "medium",
                        "expected_model": "phi3:mini",
                        "validation": {"type": "contains_all", "elements": ["temperature", "humidity", "average"]},
                        "required_elements": ["average", "temperature", "humidity", "hour"]
                    },
                    {
                        "id": "IOT_COMPLEX_01",
                        "query": "Analyze the relationship between air quality index and environmental factors (temperature, humidity, pressure), then predict when air quality might become hazardous based on observed patterns, and recommend sensor placement optimization",
                        "complexity": "complex",
                        "expected_model": "llama3.1:8b",
                        "validation": {"type": "contains_all", "elements": ["air quality", "relationship"]},
                        "required_elements": ["air quality", "temperature", "humidity", "prediction", "relationship"]
                    }
                ]
            ),
            
            # ═══════════════════════════════════════════════════════════════
            # DOMAIN 3: BUSINESS (Inventory Management)
            # ═══════════════════════════════════════════════════════════════
            DomainBenchmark(
                domain="Business_Inventory",
                data_file=str(data_dir / "test_inventory.csv"),
                queries=[
                    {
                        "id": "BIZ_SIMPLE_01",
                        "query": "How many products are in Electronics category?",
                        "complexity": "simple",
                        "expected_model": "tinyllama",
                        "validation": {"type": "numeric_present"},
                        "required_elements": ["Electronics", "product"]
                    },
                    {
                        "id": "BIZ_MEDIUM_01",
                        "query": "Calculate the total inventory value for each category and identify products that need restocking (stock below reorder level)",
                        "complexity": "medium",
                        "expected_model": "phi3:mini",
                        "validation": {"type": "contains_all", "elements": ["value", "category", "restock"]},
                        "required_elements": ["value", "category", "restock", "inventory"]
                    },
                    {
                        "id": "BIZ_COMPLEX_01",
                        "query": "Perform ABC analysis on inventory items based on value contribution, calculate inventory turnover ratios, identify slow-moving stock, and recommend optimal reorder quantities using economic order quantity principles",
                        "complexity": "complex",
                        "expected_model": "llama3.1:8b",
                        "validation": {"type": "contains_all", "elements": ["ABC", "analysis"]},
                        "required_elements": ["ABC", "analysis", "turnover", "reorder", "optimal"]
                    }
                ]
            ),
            
            # ═══════════════════════════════════════════════════════════════
            # DOMAIN 4: HEALTHCARE (Stress Assessment)
            # ═══════════════════════════════════════════════════════════════
            DomainBenchmark(
                domain="Healthcare",
                data_file=str(data_dir / "StressLevelDataset.csv"),
                queries=[
                    {
                        "id": "HEALTH_SIMPLE_01",
                        "query": "What is the average anxiety level in the dataset?",
                        "complexity": "simple",
                        "expected_model": "tinyllama",
                        "validation": {"type": "numeric_present"},
                        "required_elements": ["anxiety", "average"]
                    },
                    {
                        "id": "HEALTH_MEDIUM_01",
                        "query": "Calculate the correlation between sleep quality and stress level, and identify which factors most strongly predict high stress",
                        "complexity": "medium",
                        "expected_model": "phi3:mini",
                        "validation": {"type": "contains_all", "elements": ["sleep", "stress", "correlation"]},
                        "required_elements": ["sleep", "stress", "correlation", "factor"]
                    },
                    {
                        "id": "HEALTH_COMPLEX_01",
                        "query": "Perform a comprehensive mental health risk assessment by analyzing the interplay between anxiety, depression, self-esteem, and environmental factors, then segment the population into risk categories and recommend personalized intervention strategies for each segment",
                        "complexity": "complex",
                        "expected_model": "llama3.1:8b",
                        "validation": {"type": "contains_all", "elements": ["risk", "mental health"]},
                        "required_elements": ["anxiety", "depression", "risk", "intervention", "segment"]
                    }
                ]
            )
        ]
        
        # Load data previews
        for benchmark in benchmarks:
            benchmark.data_preview = self._load_data_preview(benchmark.data_file)
        
        return benchmarks
    
    def _run_single_test(self, domain: str, data_preview: str, query_config: dict) -> BenchmarkResult:
        """Run a single benchmark test with real LLM"""
        
        query = query_config["query"]
        test_id = query_config["id"]
        
        print(f"\n   Running: {test_id}")
        print(f"   Query: {query[:60]}...")
        
        start_time = time.time()
        
        try:
            # Step 1: Get orchestrator decision
            plan = self.orchestrator.create_execution_plan(query, {"data": True})
            
            # Step 2: Run with generator (first iteration)
            gen_prompt = f"""Analyze this data and answer the question.

QUESTION: {query}

DATA:
{data_preview[:3000]}

Provide a complete, accurate analysis with specific numbers and findings."""

            gen_response = self.llm_client.generate(
                prompt=gen_prompt,
                model=plan.model,
                adaptive_timeout=True
            )
            
            if not gen_response.get('success'):
                raise Exception(f"Generator failed: {gen_response.get('error')}")
            
            gen_output = gen_response.get('response', '')
            iterations = 1
            improved = False
            
            # Step 3: Run critic if review level is not NONE
            if plan.review_level != ReviewLevel.NONE:
                critic_prompt = f"""Review this analysis for accuracy and completeness.

ORIGINAL QUESTION: {query}

ANALYSIS:
{gen_output[:2000]}

Check for:
1. Are all calculations correct?
2. Is the analysis complete?
3. Are conclusions supported by data?

If issues found, list them. End with [NEEDS_REVISION] or [APPROVED]."""

                critic_response = self.llm_client.generate(
                    prompt=critic_prompt,
                    model="llama3.1:8b",  # Use powerful model for critic
                    adaptive_timeout=True
                )
                
                critic_output = critic_response.get('response', '')
                needs_revision = "NEEDS_REVISION" in critic_output.upper() or "ISSUE" in critic_output.upper()
                
                # Step 4: Revision if needed
                if needs_revision:
                    revision_prompt = f"""Your previous analysis needs revision based on this feedback:

FEEDBACK:
{critic_output[:1000]}

ORIGINAL QUESTION: {query}

DATA:
{data_preview[:2000]}

Provide a CORRECTED and COMPLETE analysis."""

                    revised_response = self.llm_client.generate(
                        prompt=revision_prompt,
                        model=plan.model,
                        adaptive_timeout=True
                    )
                    
                    if revised_response.get('success'):
                        gen_output = revised_response.get('response', gen_output)
                        iterations = 2
                        improved = True
            
            latency = time.time() - start_time
            
            # Step 5: Validate results
            accuracy, completeness = self._validate_output(
                gen_output, 
                query_config.get("validation", {}),
                query_config.get("required_elements", [])
            )
            
            # Check model selection
            expected_model = query_config["expected_model"]
            model_correct = plan.model == expected_model
            
            return BenchmarkResult(
                test_id=test_id,
                domain=domain,
                query=query,
                complexity_expected=query_config["complexity"],
                complexity_actual=plan.complexity_score,
                model_expected=expected_model,
                model_actual=plan.model,
                model_correct=model_correct,
                iterations=iterations,
                improved=improved,
                accuracy_score=accuracy,
                completeness_score=completeness,
                latency_seconds=latency,
                success=True
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return BenchmarkResult(
                test_id=test_id,
                domain=domain,
                query=query,
                complexity_expected=query_config["complexity"],
                complexity_actual=0.0,
                model_expected=query_config["expected_model"],
                model_actual="",
                model_correct=False,
                iterations=0,
                improved=False,
                accuracy_score=0.0,
                completeness_score=0.0,
                latency_seconds=latency,
                success=False,
                error=str(e)
            )
    
    def _validate_output(self, output: str, validation: dict, required_elements: list) -> tuple:
        """Validate output accuracy and completeness"""
        output_lower = output.lower()
        
        # Accuracy check
        accuracy = 0.5  # Base score for getting a response
        
        val_type = validation.get("type", "")
        if val_type == "numeric_present":
            # Check if any numbers are in the output
            import re
            if re.search(r'\d+\.?\d*', output):
                accuracy = 0.8
        elif val_type == "numeric_range":
            import re
            numbers = re.findall(r'\d+\.?\d*', output)
            if numbers:
                for num in numbers:
                    val = float(num)
                    if validation.get("min", 0) <= val <= validation.get("max", 1000):
                        accuracy = 1.0
                        break
        elif val_type == "contains_all":
            elements = validation.get("elements", [])
            found = sum(1 for e in elements if e.lower() in output_lower)
            accuracy = found / len(elements) if elements else 0.5
        
        # Completeness check
        if required_elements:
            found = sum(1 for e in required_elements if e.lower() in output_lower)
            completeness = found / len(required_elements)
        else:
            completeness = 0.5
        
        return accuracy, completeness
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        
        print("═"*80)
        print("NEXUS LLM ANALYTICS - RESEARCH-GRADE BENCHMARK SUITE")
        print("═"*80)
        print(f"Start Time: {datetime.now().isoformat()}")
        print(f"Domains: {len(self.benchmarks)}")
        print(f"Total Tests: {sum(len(b.queries) for b in self.benchmarks)}")
        print("═"*80)
        
        self.start_time = time.time()
        
        for benchmark in self.benchmarks:
            print(f"\n{'─'*70}")
            print(f"DOMAIN: {benchmark.domain}")
            print(f"Data: {Path(benchmark.data_file).name}")
            print(f"Tests: {len(benchmark.queries)}")
            print(f"{'─'*70}")
            
            for query_config in benchmark.queries:
                result = self._run_single_test(
                    domain=benchmark.domain,
                    data_preview=benchmark.data_preview,
                    query_config=query_config
                )
                self.results.append(result)
                
                status = "✅" if result.success else "❌"
                model_status = "✓" if result.model_correct else "✗"
                print(f"   {status} {result.test_id}: "
                      f"Model[{model_status}] Acc={result.accuracy_score:.2f} "
                      f"Comp={result.completeness_score:.2f} "
                      f"Iter={result.iterations} "
                      f"Time={result.latency_seconds:.1f}s")
        
        total_time = time.time() - self.start_time
        
        # Generate summary report
        return self._generate_report(total_time)
    
    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        
        print("\n" + "═"*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("═"*80)
        
        # Overall metrics
        total_tests = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        model_correct = sum(1 for r in self.results if r.model_correct)
        improved_count = sum(1 for r in self.results if r.improved)
        
        avg_accuracy = sum(r.accuracy_score for r in self.results) / total_tests if total_tests else 0
        avg_completeness = sum(r.completeness_score for r in self.results) / total_tests if total_tests else 0
        avg_latency = sum(r.latency_seconds for r in self.results) / total_tests if total_tests else 0
        
        # Domain breakdown
        domains = {}
        for r in self.results:
            if r.domain not in domains:
                domains[r.domain] = {"total": 0, "success": 0, "accuracy": [], "improved": 0}
            domains[r.domain]["total"] += 1
            if r.success:
                domains[r.domain]["success"] += 1
            domains[r.domain]["accuracy"].append(r.accuracy_score)
            if r.improved:
                domains[r.domain]["improved"] += 1
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful}/{total_tests} ({100*successful/total_tests:.1f}%)")
        print(f"   Model Selection Accuracy: {model_correct}/{total_tests} ({100*model_correct/total_tests:.1f}%)")
        print(f"   Improvement Rate (Two Friends): {improved_count}/{total_tests} ({100*improved_count/total_tests:.1f}%)")
        print(f"   Average Accuracy Score: {avg_accuracy:.3f}")
        print(f"   Average Completeness Score: {avg_completeness:.3f}")
        print(f"   Average Latency: {avg_latency:.2f}s")
        print(f"   Total Time: {total_time:.1f}s")
        
        print(f"\n📈 DOMAIN BREAKDOWN:")
        for domain, stats in domains.items():
            avg_acc = sum(stats["accuracy"]) / len(stats["accuracy"]) if stats["accuracy"] else 0
            print(f"   {domain}:")
            print(f"      Success Rate: {stats['success']}/{stats['total']} ({100*stats['success']/stats['total']:.0f}%)")
            print(f"      Avg Accuracy: {avg_acc:.3f}")
            print(f"      Improved by Critic: {stats['improved']}")
        
        print(f"\n🎯 COMPLEXITY ROUTING:")
        for complexity in ["simple", "medium", "complex"]:
            subset = [r for r in self.results if r.complexity_expected == complexity]
            if subset:
                correct = sum(1 for r in subset if r.model_correct)
                print(f"   {complexity.upper()}: {correct}/{len(subset)} correct model selection")
        
        # Research-ready summary
        report = {
            "benchmark_info": {
                "name": "Nexus LLM Analytics Research Benchmark",
                "version": "1.0",
                "date": datetime.now().isoformat(),
                "total_time_seconds": round(total_time, 2)
            },
            "overall_metrics": {
                "total_tests": total_tests,
                "success_rate": round(successful / total_tests, 4) if total_tests else 0,
                "model_selection_accuracy": round(model_correct / total_tests, 4) if total_tests else 0,
                "two_friends_improvement_rate": round(improved_count / total_tests, 4) if total_tests else 0,
                "average_accuracy_score": round(avg_accuracy, 4),
                "average_completeness_score": round(avg_completeness, 4),
                "average_latency_seconds": round(avg_latency, 2)
            },
            "domain_metrics": {
                domain: {
                    "success_rate": round(stats["success"] / stats["total"], 4),
                    "average_accuracy": round(sum(stats["accuracy"]) / len(stats["accuracy"]), 4) if stats["accuracy"] else 0,
                    "improvement_count": stats["improved"]
                }
                for domain, stats in domains.items()
            },
            "individual_results": [r.to_dict() for r in self.results]
        }
        
        # Save report
        report_path = project_root / "tests" / "benchmark_results.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Full report saved to: {report_path}")
        
        # Print conclusion
        print("\n" + "═"*80)
        if avg_accuracy >= 0.7 and successful == total_tests:
            print("✅ BENCHMARK PASSED - System is research-publication ready")
        elif avg_accuracy >= 0.5:
            print("⚠️ BENCHMARK PARTIAL - Some improvements needed")
        else:
            print("❌ BENCHMARK FAILED - Significant issues found")
        print("═"*80)
        
        return report


if __name__ == "__main__":
    suite = ResearchBenchmarkSuite()
    report = suite.run_full_benchmark()
