# 📋 Quick Test Reference Guide

## 🎯 One-Minute Setup

```bash
# 1. Start backend
python -m src.backend.main

# 2. Run tests (new terminal)
python automated_test_runner.py basic
```

---

## 📊 Available Test Suites

| Suite | Tests | Time | Purpose |
|-------|-------|------|---------|
| `basic` | 4 | ~2min | Smoke test all agents |
| `advanced` | 4 | ~5min | Complex analysis validation |
| `cross-agent` | 2 | ~3min | Multi-agent coordination |
| `edge-cases` | 3 | ~2min | Error handling |
| `all` | 13 | ~12min | Complete validation |

---

## 🗂️ Data Files Quick Reference

| File | Domain | Records | Use For |
|------|--------|---------|---------|
| `comprehensive_ecommerce.csv` | E-commerce | 50 | Financial, SQL, Statistical |
| `healthcare_patients.csv` | Healthcare | 50 | ML, Statistical, Domain-specific |
| `time_series_stock.csv` | Finance | 100 | Time Series, Forecasting |
| `hr_employee_data.csv` | HR | 50 | SQL, Statistical, ML |
| `university_academic_data.csv` | Education | 50 | ML prediction, Statistical |
| `sales_data.csv` | Sales | 100+ | Basic statistics |
| `StressLevelDataset.csv` | Psychology | 1100+ | ML classification |
| `nested_manufacturing.json` | Manufacturing | 3 plants | JSON parsing, IoT |
| `iot_sensor_data.csv` | IoT | 20 | Anomaly detection |
| `edge_cases/*.json` | Various | N/A | Error handling |

---

## 🤖 Agent Trigger Words

### StatisticalAgent
```
correlation, t-test, ANOVA, hypothesis, distribution, 
outlier, normality, variance, chi-square, significance
```

### FinancialAgent
```
revenue, profit, ROI, cost, CLV, margin, financial, 
cash flow, profitability, return rate, budget
```

### TimeSeriesAgent
```
trend, forecast, time series, ARIMA, seasonal, 
over time, predict, moving average, volatility
```

### MLInsightsAgent
```
cluster, segment, predict, classify, anomaly, 
pattern, feature importance, regression, PCA
```

### SQLAgent
```
SQL, query, database, table, join, group by, 
SELECT, top 10, filter where
```

---

## ✅ Quick Validation Checklist

After running tests, check:
- [ ] Agent routing >85% accurate
- [ ] No unhandled exceptions
- [ ] Response times reasonable (<60s)
- [ ] Results include interpretation
- [ ] Visualizations recommended where appropriate
- [ ] Error messages are helpful

---

## 🐛 Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| "Connection refused" | Start backend: `python -m src.backend.main` |
| "File not found" | Check file in `data/samples/` |
| Timeout | Increase TIMEOUT in test runner |
| Agent mismatch | Expected - query may be ambiguous |
| 500 error | Check `logs/app.log` for Python error |
| Model not found | Run `ollama pull <model-name>` |

---

## 📈 Expected Performance

| Metric | Target | Good | Acceptable |
|--------|--------|------|------------|
| Success Rate (basic) | 100% | >95% | >90% |
| Success Rate (advanced) | >80% | >75% | >70% |
| Agent Accuracy | >85% | >80% | >75% |
| Avg Response (simple) | 3-5s | <10s | <15s |
| Avg Response (complex) | 10-30s | <60s | <120s |

---

## 🧪 Manual Test Examples

### Test Statistical Analysis
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate correlation between sales and revenue", "filename": "sales_data.csv"}'
```

### Test Financial Analysis
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total revenue by category?", "filename": "comprehensive_ecommerce.csv"}'
```

### Test Time Series
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Show TECH stock trend", "filename": "time_series_stock.csv"}'
```

### Test ML
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Cluster customers into segments", "filename": "comprehensive_ecommerce.csv"}'
```

### Test SQL
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Show top 5 orders by amount using SQL", "filename": "comprehensive_ecommerce.csv"}'
```

---

## 📝 Test Result Interpretation

### ✅ Good Result
```json
{
  "success": true,
  "agent": "StatisticalAgent",
  "agent_match": true,
  "response_time": 4.2,
  "has_result": true,
  "has_interpretation": true
}
```

### ⚠️ Warning Result
```json
{
  "success": true,
  "agent": "DataAnalyst",  // Generic fallback
  "agent_match": false,     // Expected specific agent
  "response_time": 8.5,
  "has_result": true,
  "has_interpretation": true
}
```

### ❌ Error Result
```json
{
  "success": false,
  "error": "File not found: data.csv",
  "error_handled_correctly": true  // Graceful failure
}
```

---

## 🔄 Continuous Testing Workflow

```bash
# 1. Make code changes
vim src/backend/agents/statistical_agent.py

# 2. Quick smoke test
python automated_test_runner.py basic

# 3. Full validation before commit
python automated_test_runner.py all

# 4. Review results
cat test_results/report_all_*.json
```

---

## 🎓 Sample Questions by Complexity

### LOW (Should work 100%)
- "What are basic statistics?"
- "Show total revenue"
- "List top 10 customers"

### MEDIUM (Should work >90%)
- "Calculate correlation between X and Y"
- "What's the profit margin by category?"
- "Show 7-day moving average"

### HIGH (Should work >80%)
- "Perform t-test comparing regions"
- "Build ARIMA forecast"
- "Cluster customers using K-means"

### VERY HIGH (Should work >70%)
- "Analyze distribution and identify financial outliers"
- "Forecast stock price then predict up/down"
- "Generate comprehensive report with visualizations"

---

## 🚦 Status Indicators

When running tests, look for:
- 🧪 Test starting
- ✅ Test passed
- ❌ Test failed
- ⚠️ Warning (partial success)
- 📊 Report generated
- 💾 Results saved

---

## 📞 Quick Commands

```bash
# Run specific suite
python automated_test_runner.py basic

# Check backend health
curl http://localhost:8000/api/health

# View logs
tail -f logs/app.log

# Check models
ollama list

# Restart backend
# Ctrl+C then python -m src.backend.main
```

---

## 🎯 Success Indicators

Your system is working well if you see:
- ✅ "Backend ready to serve requests"
- ✅ Agent matches expected agent
- ✅ Response times under 60s
- ✅ Clear interpretation provided
- ✅ No Python exceptions in logs

---

**For detailed documentation:**
- Full questions: `COMPREHENSIVE_TEST_QUESTIONS.md`
- System analysis: `DEEP_ANALYSIS_SUMMARY.md`
- Usage guide: `README_TESTING.md`
