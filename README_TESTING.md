# 🧪 Comprehensive Testing Suite - README

## Overview
This directory contains a complete testing framework for Nexus LLM Analytics, including:
- **85+ test questions** covering all agent capabilities
- **18 diverse datasets** spanning multiple domains
- **Automated test runner** with reporting
- **Deep codebase analysis** documentation

## 📁 Files Created

### Documentation
- **`COMPREHENSIVE_TEST_QUESTIONS.md`** - 85+ test questions with expected answers
- **`DEEP_ANALYSIS_SUMMARY.md`** - Complete codebase analysis and insights
- **`README_TESTING.md`** - This file

### Data Files (New)
#### Production-Ready Datasets:
1. `data/samples/comprehensive_ecommerce.csv` - E-commerce orders (50 records)
2. `data/samples/healthcare_patients.csv` - Patient medical records (50 records)
3. `data/samples/time_series_stock.csv` - Stock market data (100 records, 4 sectors)
4. `data/samples/hr_employee_data.csv` - Employee HR data (50 records)
5. `data/samples/university_academic_data.csv` - Student performance (50 records)
6. `data/samples/nested_manufacturing.json` - Manufacturing IoT data
7. `data/samples/iot_sensor_data.csv` - Sensor readings with anomalies
8. `data/samples/multi_country_sales.csv` - International sales data

#### Edge Case Datasets:
9. `data/samples/edge_cases/data_quality_issues.csv` - Nulls, invalid dates, extreme values
10. `data/samples/edge_cases/complex_structures.json` - Deep nesting, mixed types

### Test Automation
- **`automated_test_runner.py`** - Python script to execute test suites

---

## 🚀 Quick Start

### 1. Start the Backend
```bash
# From project root
python -m src.backend.main
```

Wait for: `✅ Backend ready to serve requests!`

### 2. Run Basic Tests
```bash
python automated_test_runner.py basic
```

### 3. Run All Tests
```bash
python automated_test_runner.py all
```

---

## 📊 Test Suites

### `basic` - Fundamental Functionality (4 tests)
- Simple statistical analysis
- Basic financial calculations
- Time series trends
- SQL queries
- **Expected Time:** ~2 minutes

### `advanced` - Complex Analysis (4 tests)
- Hypothesis testing (t-tests, ANOVA)
- Customer Lifetime Value
- ARIMA forecasting
- ML classification models
- **Expected Time:** ~5 minutes

### `cross-agent` - Multi-Agent Coordination (2 tests)
- Statistical + Financial workflows
- Time Series + ML predictions
- **Expected Time:** ~3 minutes

### `edge-cases` - Robustness Testing (3 tests)
- Null value handling
- Mixed data types
- File not found errors
- **Expected Time:** ~2 minutes

### `all` - Complete Suite (13 tests)
- All of the above
- **Expected Time:** ~12 minutes

---

## 📝 Test Question Categories

### By Agent Type:
| Agent | Questions | Complexity Range |
|-------|-----------|------------------|
| StatisticalAgent | 15 | Low to High |
| FinancialAgent | 15 | Low to High |
| TimeSeriesAgent | 12 | Low to High |
| MLInsightsAgent | 12 | Medium to High |
| SQLAgent | 10 | Low to High |
| Cross-Agent | 8 | High |
| Edge Cases | 8 | Medium |
| Domain-Specific | 12 | Medium to High |

### By Complexity:
- **Low (20%):** Basic queries, simple calculations
- **Medium (35%):** Aggregations, filtering, correlations
- **High (35%):** Statistical tests, ML models, forecasting
- **Very High (10%):** Multi-agent workflows, complex reports

---

## 🎯 Expected Results

### Agent Routing Accuracy
- **Target:** >85% correct agent selection
- **Metric:** `agent_match` in test results

### Performance
- **Simple queries:** <10 seconds
- **Medium queries:** <30 seconds
- **Complex queries:** <60 seconds
- **Cross-agent:** <120 seconds

### Success Rate
- **Basic Suite:** >95%
- **Advanced Suite:** >80%
- **Edge Cases:** >75% (graceful failures acceptable)

---

## 📈 Sample Test Execution

```bash
$ python automated_test_runner.py basic

================================================================================
🚀 Starting Test Suite: BASIC
================================================================================

🧪 Running Q1.1: Statistical - Low
   Query: What are the basic statistics for sales and revenue?...
   ✅ Success: success | Agent: StatisticalAgent | Time: 4.52s

🧪 Running Q2.1: Financial - Low
   Query: Calculate total revenue, average order value, and revenue by product...
   ✅ Success: success | Agent: FinancialAgent | Time: 3.18s

🧪 Running Q3.1: Time Series - Low
   Query: Show the price trend for TECH stock over the time period....
   ✅ Success: success | Agent: TimeSeriesAgent | Time: 2.87s

🧪 Running Q5.1: SQL - Low
   Query: Load this data into SQL and show me the top 10 orders by total_amou...
   ✅ Success: success | Agent: SQLAgent | Time: 5.22s

================================================================================
📊 TEST REPORT: BASIC
================================================================================

Summary:
  Total Tests: 4
  ✅ Successful: 4 (100.0%)
  ❌ Failed: 0 (0.0%)
  🎯 Agent Match: 4/4 (100.0%)
  ⏱️  Avg Response Time: 3.95s
  🕐 Total Duration: 19.2s

By Category:
  Statistical: 1/1 (100.0%)
  Financial: 1/1 (100.0%)
  Time Series: 1/1 (100.0%)
  SQL: 1/1 (100.0%)

📄 Detailed report saved: test_results/report_basic_20260104_143022.json
================================================================================
```

---

## 🔍 Interpreting Results

### Test Result Structure
```json
{
  "test_id": "Q1.1",
  "category": "Statistical",
  "query": "What are the basic statistics...",
  "expected_agent": "StatisticalAgent",
  "actual_agent": "StatisticalAgent",
  "agent_match": true,
  "success": true,
  "response_time": 4.52,
  "has_result": true,
  "has_interpretation": true
}
```

### Key Metrics:
- **`agent_match`**: Did the system route to the correct agent?
- **`success`**: Did the query execute without errors?
- **`response_time`**: How long did it take?
- **`has_interpretation`**: Is there a human-readable explanation?

---

## 🧩 Domain Coverage

### Healthcare
- Patient demographics analysis
- Treatment cost patterns
- Readmission prediction
- Comorbidity associations

### E-commerce
- Revenue and profitability
- Customer segmentation
- Return analysis
- Payment method optimization

### Financial/Stock Market
- Time series forecasting
- Volatility analysis
- Sector comparison
- Technical indicators

### Human Resources
- Salary analysis by department
- Performance drivers
- Remote vs office productivity
- Training ROI

### Education
- GPA prediction models
- Study habits correlation
- At-risk student identification
- Major difficulty comparison

### Manufacturing
- Production efficiency
- Quality metrics
- Sensor anomaly detection
- Supply chain optimization

---

## 🐛 Troubleshooting

### Backend Not Responding
```bash
# Check if running
curl http://localhost:8000/api/health

# View logs
tail -f logs/app.log
```

### Test Failures
1. **File not found**: Ensure data files exist in `data/samples/`
2. **Timeout**: Increase `TIMEOUT` in `automated_test_runner.py`
3. **Agent mismatch**: May indicate query ambiguity or routing issues
4. **500 errors**: Check backend logs for Python exceptions

### Model Issues
```bash
# Check available models
ollama list

# Pull required models if missing
ollama pull qwen2.5:1.5b
ollama pull deepseek-r1:7b
```

---

## 📚 Additional Resources

### Query Examples by Agent

**StatisticalAgent:**
- "Calculate correlation between X and Y"
- "Perform t-test comparing group A vs B"
- "Test for normality in the revenue distribution"

**FinancialAgent:**
- "What's the profit margin by product?"
- "Calculate ROI for each campaign"
- "Forecast next quarter revenue"

**TimeSeriesAgent:**
- "Show trend over time"
- "Forecast next 7 days using ARIMA"
- "Detect seasonality patterns"

**MLInsightsAgent:**
- "Cluster customers into segments"
- "Predict churn probability"
- "Find anomalies in transaction data"

**SQLAgent:**
- "Show top 10 customers by revenue"
- "Calculate average by category"
- "Filter records where status = 'active'"

---

## 🎓 Learning Objectives

This test suite teaches:

1. **Multi-Agent Systems** - How specialized agents collaborate
2. **Natural Language Processing** - Intent classification and routing
3. **Statistical Analysis** - Hypothesis testing, correlation, distributions
4. **Financial Analytics** - Profitability, CLV, forecasting
5. **Time Series** - ARIMA, decomposition, stationarity
6. **Machine Learning** - Classification, clustering, regression
7. **Data Quality** - Handling nulls, outliers, mixed types
8. **API Design** - RESTful patterns, error handling
9. **Performance Optimization** - Caching, model selection, RAM awareness

---

## 🏗️ Extending the Test Suite

### Adding New Tests

Edit `automated_test_runner.py`:

```python
NEW_TEST = {
    "id": "Q99.1",
    "category": "Custom",
    "complexity": "Medium",
    "file": "my_data.csv",
    "query": "Your custom query here",
    "expected_agent": "StatisticalAgent"
}

# Add to appropriate suite
BASIC_TESTS.append(NEW_TEST)
```

### Adding New Data

```bash
# Place file in data/samples/
cp my_dataset.csv data/samples/

# Test manually first
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze this data", "filename": "my_dataset.csv"}'
```

---

## 📊 Benchmarking

### Performance Baseline
| Query Type | Target Time | Acceptable Range |
|------------|-------------|------------------|
| Simple stats | 3-5s | 1-10s |
| Aggregation | 4-8s | 2-15s |
| Hypothesis test | 5-12s | 3-20s |
| ML clustering | 10-20s | 5-40s |
| ARIMA forecast | 15-30s | 10-60s |
| Cross-agent | 20-40s | 15-120s |

### System Requirements
- **RAM:** 8GB minimum, 16GB recommended
- **CPU:** 4 cores minimum
- **Disk:** 10GB free space
- **Network:** Local (no internet required)

---

## 🔐 Security Notes

- All tests run locally (no external API calls)
- No sensitive data in sample files
- SQL injection prevention tested
- Input sanitization validated

---

## 📞 Support

### Issues?
1. Check `DEEP_ANALYSIS_SUMMARY.md` for architecture details
2. Review `COMPREHENSIVE_TEST_QUESTIONS.md` for expected behaviors
3. Examine test results in `test_results/` directory
4. Check backend logs for errors

### Contributing
- Add new test cases for untested scenarios
- Create domain-specific datasets
- Improve test automation scripts
- Document edge cases discovered

---

## 🎉 Success Criteria

Your system is production-ready if:
- ✅ Basic suite: 100% pass
- ✅ Advanced suite: >80% pass
- ✅ Agent routing: >85% accuracy
- ✅ Avg response time: <10s (simple), <60s (complex)
- ✅ No crashes or unhandled exceptions
- ✅ Graceful error handling on invalid inputs
- ✅ Clear, actionable error messages

---

**Happy Testing! 🚀**

For detailed analysis methodology, see `DEEP_ANALYSIS_SUMMARY.md`  
For complete question list, see `COMPREHENSIVE_TEST_QUESTIONS.md`
