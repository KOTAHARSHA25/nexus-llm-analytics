# Research Metrics & Testing Documentation
**Project:** Nexus LLM Analytics  
**Purpose:** Define metrics for research evaluation and testing  
**Date:** December 2024

---

## Table of Contents
1. [System Performance Metrics](#1-system-performance-metrics)
2. [Query Routing Metrics](#2-query-routing-metrics)
3. [Model Performance Metrics](#3-model-performance-metrics)
4. [Self-Learning Metrics](#4-self-learning-metrics)
5. [Code Generation Metrics](#5-code-generation-metrics)
6. [Resource Optimization Metrics](#6-resource-optimization-metrics)
7. [User Experience Metrics](#7-user-experience-metrics)
8. [Testing Framework](#8-testing-framework)

---

## 1. System Performance Metrics

### 1.1 Query Response Time
**Definition:** Total time from query submission to result delivery

**Measurement:**
```python
start_time = time.time()
result = await analysis_service.analyze(query, data)
response_time = time.time() - start_time
```

**Target Metrics:**
- Simple queries: < 5 seconds
- Medium complexity: < 15 seconds
- Complex queries: < 30 seconds

**Logging Location:** `analysis_service.py` - execution time logged per query

---

### 1.2 Component Breakdown
**Sub-Metrics:**
- **Routing Time:** Time to create execution plan
- **LLM Inference Time:** Model execution time
- **Code Execution Time:** Sandbox execution time
- **Review Time:** Self-correction cycle time

**Formula:**
```
Total Response Time = Routing + LLM + Execution + Review
```

---

### 1.3 Throughput
**Definition:** Queries processed per minute

**Measurement:**
```python
queries_completed = len(completed_queries)
time_elapsed = end_time - start_time
throughput = queries_completed / (time_elapsed / 60)  # QPM
```

**Target:** 5-10 queries/minute for concurrent users

---

## 2. Query Routing Metrics

### 2.1 Routing Accuracy
**Definition:** Percentage of queries routed to the correct agent

**Ground Truth Labels:**
- Financial queries → Financial Agent
- Statistical queries → Statistical Agent
- Visualization queries → Visualizer Agent
- General analysis → Data Analyst Agent

**Formula:**
```
Routing Accuracy = (Correct Routes / Total Queries) × 100%
```

**Target:** > 90% accuracy

**Test Dataset:** [benchmarks/benchmark_dataset.json](../benchmarks/benchmark_dataset.json)

---

### 2.2 Semantic vs Keyword Routing
**Definition:** Frequency of semantic routing vs keyword fallback

**Measurement:**
```python
semantic_routes = count_log_entries("Semantic routing successful")
keyword_fallbacks = count_log_entries("Using keyword-based heuristic")
semantic_ratio = semantic_routes / (semantic_routes + keyword_fallbacks)
```

**Target:** > 80% semantic routing (keyword fallback < 20%)

**Logging:** `query_orchestrator.py` lines 258, 276

---

### 2.3 Complexity Scoring Accuracy
**Definition:** Correlation between complexity score and actual query difficulty

**Measurement:**
```python
predicted_complexity = plan.complexity_score
actual_complexity = execution_time / baseline_time
accuracy = 1 - abs(predicted_complexity - actual_complexity)
```

**Target:** Correlation coefficient > 0.7

---

## 3. Model Performance Metrics

### 3.1 Model Selection Accuracy
**Definition:** Percentage of queries where selected model was optimal

**Optimal Model Criteria:**
- Sufficient capability to solve query
- Minimal resource usage
- Fastest response time

**Formula:**
```
Model Accuracy = (Optimal Selections / Total Queries) × 100%
```

**Target:** > 85%

---

### 3.2 Downgrade Frequency
**Definition:** How often optimizer downgrades model due to resource constraints

**Measurement:**
```python
downgrades = count_log_entries("[OPTIMIZER] Model downgraded")
total_queries = total_query_count
downgrade_rate = downgrades / total_queries
```

**Target:** < 10% under normal load, > 50% under high load

**Logging:** `query_orchestrator.py` line 294

---

### 3.3 Fallback Success Rate
**Definition:** Percentage of queries that succeed using fallback models

**Measurement:**
```python
fallback_attempts = count_fallback_invocations()
fallback_successes = count_fallback_successes()
fallback_success_rate = fallback_successes / fallback_attempts
```

**Target:** > 70% (indicates good fallback chain)

---

## 4. Self-Learning Metrics

### 4.1 Error Pattern Recognition
**Definition:** Percentage of errors that match stored patterns

**Measurement:**
```python
similar_errors = memory.query(error_embedding, top_k=3)
match_found = any(similarity > 0.8 for similarity in similar_errors)
recognition_rate = matches / total_errors
```

**Target:** > 50% recognition rate after 100 queries

**Implementation:** `self_correction_engine.py` line 558

---

### 4.2 Fix Success Rate
**Definition:** Percentage of errors successfully corrected using learned fixes

**Measurement:**
```python
corrections_attempted = count_correction_attempts()
corrections_successful = count_successful_corrections()
fix_success_rate = corrections_successful / corrections_attempted
```

**Target:** > 60% after learning phase

---

### 4.3 Memory Growth
**Definition:** Rate at which new error patterns are stored

**Measurement:**
```python
patterns_stored = memory.collection.count()
queries_processed = total_query_count
storage_rate = patterns_stored / queries_processed
```

**Target:** 10-20% (only novel errors should be stored)

**Implementation:** `self_correction_engine.py` line 527

---

## 5. Code Generation Metrics

### 5.1 Code Correctness
**Definition:** Percentage of generated code that executes without errors

**Measurement:**
```python
code_executions = count_code_executions()
successful_executions = count_successful_executions()
correctness_rate = successful_executions / code_executions
```

**Target:** > 80% first-attempt success

---

### 5.2 Security Violation Rate
**Definition:** Percentage of generated code blocked by security guards

**Measurement:**
```python
security_blocks = count_log_entries("Code validation failed")
total_generations = count_code_generations()
violation_rate = security_blocks / total_generations
```

**Target:** < 5% (low violation indicates good LLM prompt engineering)

**Logging:** `sandbox.py`, `security_guards.py`

---

### 5.3 Review Cycles
**Definition:** Average number of Generator-Critic cycles before success

**Measurement:**
```python
total_cycles = sum(query_cycles for query in queries)
avg_cycles = total_cycles / len(queries)
```

**Target:** < 2.0 cycles per query

**Implementation:** `self_correction_engine.py` - tracks iteration count

---

## 6. Resource Optimization Metrics

### 6.1 RAM Usage
**Definition:** System memory usage during query execution

**Measurement:**
```python
import psutil
ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
ram_percent = psutil.virtual_memory().percent
```

**Target:** 
- Average usage < 4GB
- Peak usage < 6GB
- Never exceed 85% of available RAM

**Implementation:** `optimizers.py` line 617

---

### 6.2 CPU Utilization
**Definition:** Processor usage during LLM inference

**Measurement:**
```python
cpu_percent = psutil.cpu_percent(interval=1)
```

**Target:** 
- Average < 70%
- Peaks acceptable during inference

---

### 6.3 Optimization Impact
**Definition:** Performance improvement from optimizer decisions

**Measurement:**
```python
time_without_optimization = baseline_execution_time
time_with_optimization = actual_execution_time
speedup = time_without_optimization / time_with_optimization
```

**Target:** 1.2x - 2.0x speedup for optimized queries

---

## 7. User Experience Metrics

### 7.1 Query Success Rate
**Definition:** Percentage of queries that produce valid results

**Measurement:**
```python
successful_queries = count_queries_with_results()
total_queries = total_query_count
success_rate = successful_queries / total_queries
```

**Target:** > 85%

---

### 7.2 Interpretation Quality
**Definition:** Human-readable interpretation provided with results

**Measurement (Manual Evaluation):**
- **Scale:** 1-5 (1=poor, 5=excellent)
- **Criteria:**
  - Clarity: Is the interpretation easy to understand?
  - Accuracy: Does it match the data?
  - Completeness: Are key insights included?

**Target:** Average rating > 4.0

---

### 7.3 Visualization Quality
**Definition:** Quality of auto-generated charts

**Criteria:**
- Correct chart type for data
- Properly labeled axes
- Appropriate colors/styling
- No execution errors

**Measurement:**
```python
viz_successful = count_successful_visualizations()
viz_attempted = count_visualization_requests()
viz_success_rate = viz_successful / viz_attempted
```

**Target:** > 75%

---

## 8. Testing Framework

### 8.1 Benchmark Suite
**Location:** [benchmarks/](../benchmarks/)

**Test Categories:**
1. **Routing Tests:** Verify correct agent selection
2. **Model Selection Tests:** Verify optimal model choice
3. **Code Generation Tests:** Verify correctness and security
4. **Optimization Tests:** Verify resource-aware downgrade
5. **Self-Learning Tests:** Verify error pattern storage and retrieval

**Run Command:**
```bash
python benchmarks/benchmark_runner.py
```

---

### 8.2 Unit Tests
**Location:** [tests/backend/](../tests/backend/)

**Coverage Areas:**
- Query Orchestrator
- Self-Correction Engine
- Security Guards
- Agent System
- Optimizers

**Run Command:**
```bash
pytest tests/backend/ -v --cov
```

---

### 8.3 Integration Tests
**Test Scenarios:**
1. **End-to-End Query Flow**
   - Submit query → Route → Execute → Review → Return result
2. **Fallback Chain**
   - Primary model fails → Try fallback → Success
3. **Resource Exhaustion**
   - Low RAM → Model downgrade → Continued operation
4. **Error Recovery**
   - Code fails → Critic feedback → Retry → Success

**Run Command:**
```bash
python scripts/automated_test_runner.py
```

---

### 8.4 Performance Benchmarks
**Location:** [benchmarks/run_research_benchmarks.py](../benchmarks/run_research_benchmarks.py)

**Metrics Collected:**
- Response time distribution (p50, p90, p99)
- Routing accuracy per agent
- Model performance comparison
- Resource usage trends

**Output:** JSON report with statistical analysis

---

## 9. Monitoring & Logging

### 9.1 Structured Logging
**Format:** JSON-structured logs for easy parsing

**Required Fields:**
```json
{
  "timestamp": "2024-12-21T10:30:00Z",
  "level": "INFO",
  "component": "query_orchestrator",
  "event": "routing_decision",
  "query": "Forecast sales for next 3 days",
  "model": "phi3:mini",
  "complexity": 0.75,
  "execution_method": "code_generation",
  "response_time": 12.3
}
```

---

### 9.2 Backend Visibility Logs
**New Additions (Audit Phase):**

**Agent Execution Logs:**
```
================================================================================
🤖 AGENT EXECUTION: Financial Forecasting Agent
================================================================================
📝 Query: Forecast the total daily sales for the next 3 days.
⏱️  Execution Time: 12.34s
✅ Status: Success
📊 Result: Forecast generated: Day 1=$1,234, Day 2=$1,456, Day 3=$1,389
🔍 Metadata: code_generated=true, visualization=false
================================================================================
```

**Routing Decision Logs:**
```
================================================================================
🎯 ROUTING DECISION
================================================================================
📝 Query: Forecast the total daily sales for the next 3 days.
🤖 Model Selected: phi3:mini
⚙️  Execution Method: code_generation
🔍 Review Level: standard
📊 Complexity Score: 0.75
🧠 Intent: forecasting
🔧 User Override: No
💡 Reasoning: Medium complexity query requiring statistical forecasting
================================================================================
```

**Location:** 
- Agent logs: `plugin_system.py` lines 60-87
- Routing logs: `query_orchestrator.py` lines 341-354

---

### 9.3 Metrics Collection
**Prometheus Integration:**
- Counter: `nexus_queries_total`
- Histogram: `nexus_response_time_seconds`
- Gauge: `nexus_ram_usage_gb`
- Counter: `nexus_routing_accuracy_total`

**Endpoint:** `/metrics` (if Prometheus client available)

---

## 10. Research Evaluation Protocol

### 10.1 Baseline Comparison
**Methodology:**
1. Establish baseline with keyword-only routing
2. Compare against semantic routing
3. Measure improvement in accuracy and speed

**Metrics:**
- Routing accuracy improvement
- Response time reduction
- User satisfaction increase

---

### 10.2 A/B Testing
**Variables to Test:**
- Semantic vs Keyword routing
- Optimizer enabled vs disabled
- Different model combinations
- Review levels (strict vs standard)

**Sample Size:** Minimum 100 queries per variant

---

### 10.3 Error Analysis
**Location:** [benchmarks/error_analysis.py](../benchmarks/error_analysis.py)

**Analysis:**
- Categorize error types
- Identify common failure patterns
- Track fix success rate over time
- Generate improvement recommendations

---

## 11. Key Performance Indicators (KPIs)

### 11.1 System Health
| KPI | Target | Critical Threshold |
|-----|--------|-------------------|
| Query Success Rate | > 85% | < 70% |
| Average Response Time | < 15s | > 30s |
| RAM Usage | < 4GB | > 6GB |
| CPU Usage | < 70% | > 90% |

### 11.2 Intelligence
| KPI | Target | Critical Threshold |
|-----|--------|-------------------|
| Routing Accuracy | > 90% | < 75% |
| Model Selection Accuracy | > 85% | < 70% |
| Code Correctness | > 80% | < 60% |
| Fix Success Rate | > 60% | < 40% |

### 11.3 User Experience
| KPI | Target | Critical Threshold |
|-----|--------|-------------------|
| Interpretation Quality | > 4.0/5 | < 3.0/5 |
| Visualization Success | > 75% | < 50% |
| Security Violations | < 5% | > 15% |

---

## 12. Data Collection Tools

### 12.1 Log Aggregation
**Tool:** Python logging with file handlers

**Configuration:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nexus.log'),
        logging.StreamHandler()
    ]
)
```

---

### 12.2 Query History
**Location:** [history/query_history.json](../history/query_history.json)

**Schema:**
```json
{
  "query_id": "uuid",
  "timestamp": "ISO8601",
  "query": "text",
  "agent": "agent_name",
  "model": "model_name",
  "complexity": 0.75,
  "execution_time": 12.34,
  "success": true,
  "error": null
}
```

---

### 12.3 Benchmark Results
**Location:** [reports/benchmarks/](../reports/benchmarks/)

**Files:**
- `routing_accuracy_YYYYMMDD.json`
- `performance_report_YYYYMMDD.json`
- `error_analysis_YYYYMMDD.json`

---

## 13. Continuous Monitoring

### 13.1 Real-Time Dashboards
**Metrics to Display:**
- Query rate (QPM)
- Success rate (%)
- Average response time (s)
- RAM usage (GB)
- Active queries

**Tool:** Grafana (if Prometheus enabled) or custom web dashboard

---

### 13.2 Alerting
**Alert Conditions:**
- Success rate < 70% for 10 consecutive queries
- Response time > 30s for 5 consecutive queries
- RAM usage > 85% for 1 minute
- Security violations > 10% in last 100 queries

**Notification:** Log warning or send email

---

## 14. Testing Checklist

### 14.1 Pre-Release Testing
- [ ] Run full benchmark suite
- [ ] Verify routing accuracy > 90%
- [ ] Verify code correctness > 80%
- [ ] Check security guard coverage
- [ ] Test fallback chains
- [ ] Verify resource optimization
- [ ] Test logging output format

### 14.2 Regression Testing
- [ ] Run after each major change
- [ ] Compare metrics to baseline
- [ ] Flag any degradation > 10%
- [ ] Re-run failed tests individually

### 14.3 Load Testing
- [ ] Test with 10 concurrent users
- [ ] Monitor RAM and CPU usage
- [ ] Verify optimizer triggers downgrade
- [ ] Check query queue handling

---

## 15. Research Questions

### 15.1 Routing Intelligence
**Question:** Does semantic routing outperform keyword-based routing?

**Hypothesis:** Semantic routing will achieve > 90% accuracy vs < 75% for keywords

**Metrics:** Routing accuracy, false positive rate, response time

---

### 15.2 Self-Learning
**Question:** Does vector memory improve error correction over time?

**Hypothesis:** Fix success rate will increase from 40% to > 60% after 100 queries

**Metrics:** Fix success rate, recognition rate, memory growth

---

### 15.3 Resource Optimization
**Question:** Does automatic model downgrade prevent system crashes without sacrificing accuracy?

**Hypothesis:** System uptime increases by > 50%, accuracy drops < 10%

**Metrics:** Crash rate, downgrade frequency, accuracy comparison

---

## 16. Conclusion

This metrics framework provides comprehensive evaluation for the Nexus LLM Analytics system. Key areas:

1. **Performance:** Response time, throughput, resource usage
2. **Intelligence:** Routing accuracy, model selection, self-learning
3. **Quality:** Code correctness, security, user experience
4. **Monitoring:** Real-time logging, alerting, dashboards

**Next Steps:**
1. Implement Prometheus metrics collection
2. Set up automated benchmark runs (weekly)
3. Create Grafana dashboard for real-time monitoring
4. Conduct A/B tests on routing methods

---

## Appendix: Metric Calculation Examples

### Example 1: Routing Accuracy
```python
def calculate_routing_accuracy(test_queries):
    correct = 0
    for query, expected_agent in test_queries:
        result = orchestrator.route_query(query)
        if result.agent == expected_agent:
            correct += 1
    return correct / len(test_queries) * 100
```

### Example 2: Fix Success Rate
```python
def calculate_fix_success_rate(error_log):
    corrections = [e for e in error_log if e['correction_attempted']]
    successes = [e for e in corrections if e['correction_successful']]
    return len(successes) / len(corrections) * 100 if corrections else 0
```

### Example 3: Resource Efficiency
```python
def calculate_resource_efficiency():
    total_queries = len(query_history)
    downgrade_events = count_downgrades(query_history)
    avg_ram = sum(q['ram_usage'] for q in query_history) / total_queries
    
    return {
        'downgrade_rate': downgrade_events / total_queries,
        'avg_ram_gb': avg_ram,
        'efficiency_score': 1 - (avg_ram / max_ram_capacity)
    }
```

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Status:** ✅ Ready for Research Evaluation
