# Nexus LLM Analytics - Comprehensive Test Questions & Expected Answers
## Generated: January 4, 2026
## Based on Deep Code Analysis

---

## SECTION 1: BASIC STATISTICAL ANALYSIS (Statistical Agent)

### Q1.1 - Descriptive Statistics (Simple)
**File:** sales_data.csv  
**Question:** "What are the basic statistics for sales and revenue?"  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- Mean sales: ~5,500 units
- Mean revenue: ~$25,000
- Standard deviation calculations for both
- Min/max values
- Median values
**Complexity:** Low

### Q1.2 - Correlation Analysis (Medium)
**File:** sales_data.csv  
**Question:** "Is there a correlation between marketing_spend and revenue? Perform correlation analysis."  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- Pearson correlation coefficient
- P-value for significance
- Scatter plot visualization recommendation
- Interpretation: "Weak/moderate/strong positive/negative correlation"
**Complexity:** Medium

### Q1.3 - Hypothesis Testing (Advanced)
**File:** sales_data.csv  
**Question:** "Test if the average sales differs significantly between North and South regions using a t-test."  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- T-statistic value
- P-value
- Degrees of freedom
- Conclusion: "Reject/Fail to reject null hypothesis at α=0.05"
- Mean sales for each region
**Complexity:** High

### Q1.4 - ANOVA Test (Advanced)
**File:** sales_data.csv  
**Question:** "Perform ANOVA to test if revenue differs significantly across all regions."  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- F-statistic
- P-value
- Between-group variance
- Within-group variance
- Post-hoc test results if significant
**Complexity:** High

### Q1.5 - Distribution Analysis (Medium)
**File:** sales_data.csv  
**Question:** "What is the distribution of revenue? Test for normality."  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- Shapiro-Wilk test results
- Skewness and kurtosis values
- Q-Q plot recommendation
- Conclusion about normality
**Complexity:** Medium

---

## SECTION 2: FINANCIAL ANALYSIS (Financial Agent)

### Q2.1 - Revenue Analysis (Simple)
**File:** comprehensive_ecommerce.csv  
**Question:** "Calculate total revenue, average order value, and revenue by product category."  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- Total revenue: Sum of all total_amount values
- AOV: Total revenue / number of orders
- Revenue breakdown by category (Electronics, Home & Garden, Clothing, Books)
- Percentage contribution of each category
**Complexity:** Low

### Q2.2 - Profitability Analysis (Medium)
**File:** comprehensive_ecommerce.csv  
**Question:** "Calculate profit margins considering discount_percent and shipping_cost. Which categories are most profitable?"  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- Gross profit calculation: total_amount - (unit_price * quantity * discount_percent) - shipping_cost
- Profit margin percentages by category
- Ranking of categories by profitability
- Identification of loss-making orders
**Complexity:** Medium

### Q2.3 - Customer Lifetime Value (Advanced)
**File:** comprehensive_ecommerce.csv  
**Question:** "Calculate Customer Lifetime Value (CLV) for each customer segment. Which segment has highest CLV?"  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- CLV calculation by segment (Premium, Regular, Budget)
- Average order frequency per segment
- Average order value per segment
- Total revenue contribution by segment
- Retention rate indicators
**Complexity:** High

### Q2.4 - ROI Analysis (Medium)
**File:** comprehensive_ecommerce.csv  
**Question:** "What is the ROI for orders with different payment methods?"  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- Revenue by payment method
- Cost considerations (processing fees assumptions)
- ROI calculation and ranking
- Recommendations for payment method preferences
**Complexity:** Medium

### Q2.5 - Return Rate Impact (Advanced)
**File:** comprehensive_ecommerce.csv  
**Question:** "Analyze the financial impact of returns. Calculate return rate by category and estimate revenue loss."  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- Return rate percentage by category
- Total revenue lost from returns
- Average return value
- Return reasons analysis
- Financial recommendations to reduce returns
**Complexity:** High

### Q2.6 - Cash Flow Forecast (Advanced)
**File:** comprehensive_ecommerce.csv  
**Question:** "Project next month's revenue based on historical trends and order patterns."  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- Trend analysis from order_date
- Growth rate calculation
- Forecasted revenue range
- Confidence interval
- Seasonal considerations
**Complexity:** High

---

## SECTION 3: TIME SERIES ANALYSIS (Time Series Agent)

### Q3.1 - Trend Detection (Simple)
**File:** time_series_stock.csv  
**Question:** "Show the price trend for TECH stock over the time period."  
**Expected Agent:** TimeSeriesAgent  
**Expected Result:**
- Overall trend direction (upward/downward/stable)
- Trend line equation
- Percentage change over period
- Visualization recommendation (line chart)
**Complexity:** Low

### Q3.2 - Volatility Analysis (Medium)
**File:** time_series_stock.csv  
**Question:** "Which sector shows the highest volatility? Calculate daily volatility for each sector."  
**Expected Agent:** TimeSeriesAgent  
**Expected Result:**
- Volatility values already in dataset
- Average volatility by sector
- Standard deviation of volatility
- Ranking: Energy > Healthcare > Technology > Financial (expected order)
- Explanation of volatility patterns
**Complexity:** Medium

### Q3.3 - Seasonal Decomposition (Advanced)
**File:** time_series_stock.csv  
**Question:** "Perform seasonal decomposition on TECH stock prices. Identify trend, seasonal, and residual components."  
**Expected Agent:** TimeSeriesAgent  
**Expected Result:**
- Trend component extracted
- Seasonal patterns identified (if any within 1 month)
- Residual/noise component
- Visualization with 3 subplots
- Note: May indicate insufficient data for strong seasonality
**Complexity:** High

### Q3.4 - ARIMA Forecasting (Advanced)
**File:** time_series_stock.csv  
**Question:** "Build an ARIMA model to forecast TECH stock prices for the next 5 days."  
**Expected Agent:** TimeSeriesAgent  
**Expected Result:**
- ARIMA model parameters (p, d, q)
- Stationarity test results (ADF test)
- Forecasted prices for next 5 days
- Confidence intervals
- Model diagnostics (AIC, residual analysis)
**Complexity:** High

### Q3.5 - Moving Average (Medium)
**File:** time_series_stock.csv  
**Question:** "Calculate 7-day and 14-day moving averages for all stocks. Identify crossover points."  
**Expected Agent:** TimeSeriesAgent  
**Expected Result:**
- MA7 and MA14 values calculated
- Crossover points identified (bullish/bearish signals)
- Visualization with original and MA lines
- Trading signal interpretation
**Complexity:** Medium

### Q3.6 - Autocorrelation (Advanced)
**File:** time_series_stock.csv  
**Question:** "Perform autocorrelation analysis on FIN stock returns. What patterns exist?"  
**Expected Agent:** TimeSeriesAgent  
**Expected Result:**
- ACF and PACF plots
- Lag values showing significant correlation
- Interpretation of patterns
- Recommendations for time series modeling
**Complexity:** High

---

## SECTION 4: MACHINE LEARNING INSIGHTS (ML Insights Agent)

### Q4.1 - Clustering Analysis (Medium)
**File:** comprehensive_ecommerce.csv  
**Question:** "Segment customers into clusters based on their purchasing behavior using K-means."  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Optimal number of clusters (elbow method/silhouette score)
- Cluster characteristics (3-5 clusters expected)
- Features: total_amount, quantity, discount_percent, customer_segment
- Cluster profiles and interpretation
**Complexity:** Medium

### Q4.2 - Anomaly Detection (Medium)
**File:** healthcare_patients.csv  
**Question:** "Detect anomalies in patient treatment costs. Which patients have unusual costs?"  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Anomaly detection method (Isolation Forest or Z-score)
- List of anomalous patients with IDs
- Threshold used for detection
- Explanation: High-cost treatments (stroke, cancer, sepsis patients)
- Statistical justification
**Complexity:** Medium

### Q4.3 - Feature Importance (Advanced)
**File:** university_academic_data.csv  
**Question:** "What features most strongly predict student GPA? Perform feature importance analysis."  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Random Forest or similar model
- Feature importance scores ranked
- Expected top features: study_hours_week, attendance_rate, midterm_score, final_score
- Visualization (bar chart)
- Actionable insights for improving GPA
**Complexity:** High

### Q4.4 - Classification (Advanced)
**File:** StressLevelDataset.csv  
**Question:** "Build a classification model to predict stress_level based on other features. What's the accuracy?"  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Train/test split performed
- Model type (Random Forest, Decision Tree, Logistic Regression)
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Top predictive features (anxiety_level, depression, sleep_quality expected)
- Cross-validation results
**Complexity:** High

### Q4.5 - Dimensionality Reduction (Advanced)
**File:** healthcare_patients.csv  
**Question:** "Apply PCA to reduce dimensions of patient health metrics. How many components explain 90% variance?"  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- PCA performed on numeric health features
- Explained variance ratio
- Number of components for 90% variance (likely 5-8)
- Scree plot visualization
- Principal component loadings
**Complexity:** High

### Q4.6 - Regression Analysis (Advanced)
**File:** hr_employee_data.csv  
**Question:** "Predict employee salary based on experience, education, and performance. What factors matter most?"  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Regression model (Linear/Random Forest Regressor)
- R² score and RMSE
- Coefficient values or feature importance
- Expected key factors: position, hire_date (experience), education_level, performance_rating
- Residual analysis
**Complexity:** High

---

## SECTION 5: SQL & DATA QUERYING (SQL Agent)

### Q5.1 - Simple SQL Query (Simple)
**File:** comprehensive_ecommerce.csv  
**Question:** "Load this data into SQL and show me the top 10 orders by total_amount."  
**Expected Agent:** SQLAgent (EnhancedSQLAgent)  
**Expected Result:**
- CSV loaded into in-memory SQLite database
- SQL query: SELECT * FROM data ORDER BY total_amount DESC LIMIT 10
- Tabular result with top 10 rows
- Order IDs and amounts displayed
**Complexity:** Low

### Q5.2 - Aggregation Query (Medium)
**File:** comprehensive_ecommerce.csv  
**Question:** "Using SQL, calculate average total_amount by customer_segment and order by highest average."  
**Expected Agent:** SQLAgent  
**Expected Result:**
- SQL: SELECT customer_segment, AVG(total_amount) as avg_amount FROM data GROUP BY customer_segment ORDER BY avg_amount DESC
- Results showing Premium > Regular > Budget
- Proper formatting of currency values
**Complexity:** Medium

### Q5.3 - Join-Like Query (Advanced)
**File:** comprehensive_ecommerce.csv  
**Question:** "Find customers who made purchases in multiple categories. Use SQL."  
**Expected Agent:** SQLAgent  
**Expected Result:**
- SQL with GROUP BY customer_id, COUNT(DISTINCT product_category)
- Filter for count > 1
- List of multi-category customers
- Count of categories per customer
**Complexity:** High

### Q5.4 - Date Filtering (Medium)
**File:** comprehensive_ecommerce.csv  
**Question:** "Using SQL, show orders placed in January 2024 where payment was by Credit Card."  
**Expected Agent:** SQLAgent  
**Expected Result:**
- SQL with WHERE clause on order_date and payment_method
- Date parsing and filtering
- Count of matching orders
- Total revenue from these orders
**Complexity:** Medium

### Q5.5 - Complex Aggregation (Advanced)
**File:** hr_employee_data.csv  
**Question:** "Write SQL to find departments with average salary > $80,000 and more than 5 employees."  
**Expected Agent:** SQLAgent  
**Expected Result:**
- SQL with GROUP BY department, HAVING clauses
- AVG(salary) calculation
- COUNT(*) for employee count
- Results showing Engineering department primarily
**Complexity:** High

---

## SECTION 6: CROSS-AGENT COORDINATION (Multi-Agent)

### Q6.1 - Statistical + Financial (Cross-Agent)
**File:** comprehensive_ecommerce.csv  
**Question:** "Perform statistical analysis on revenue distribution and identify financial outliers. What's the business impact?"  
**Expected Agents:** StatisticalAgent → FinancialAgent  
**Expected Result:**
- Statistical: distribution shape, outliers via IQR/z-score
- Financial: revenue impact of outliers, profit analysis
- Combined interpretation
- Business recommendations
**Complexity:** High

### Q6.2 - Time Series + ML (Cross-Agent)
**File:** time_series_stock.csv  
**Question:** "Analyze TECH stock trend, then use ML to predict if price will increase or decrease tomorrow."  
**Expected Agents:** TimeSeriesAgent → MLInsightsAgent  
**Expected Result:**
- Time series: trend analysis, recent patterns
- ML: Binary classification (up/down prediction)
- Feature engineering from time series data
- Prediction with confidence score
**Complexity:** High

### Q6.3 - SQL + Statistical (Cross-Agent)
**File:** hr_employee_data.csv  
**Question:** "Query employees by department using SQL, then perform ANOVA to test if salaries differ significantly across departments."  
**Expected Agents:** SQLAgent → StatisticalAgent  
**Expected Result:**
- SQL: Data grouped by department
- Statistical: ANOVA test results
- F-statistic and p-value
- Post-hoc comparison between departments
**Complexity:** High

### Q6.4 - Financial + ML (Cross-Agent)
**File:** comprehensive_ecommerce.csv  
**Question:** "Calculate customer profitability, then cluster customers into segments for targeted marketing."  
**Expected Agents:** FinancialAgent → MLInsightsAgent  
**Expected Result:**
- Financial: CLV, profit per customer
- ML: K-means clustering on financial metrics
- Cluster profiles (high-value, medium-value, low-value)
- Marketing strategy recommendations
**Complexity:** High

---

## SECTION 7: COMPLEX NESTED DATA (JSON Handling)

### Q7.1 - Nested JSON Parsing (Medium)
**File:** nested_manufacturing.json  
**Question:** "Extract all sensor data and identify sensors with 'warning' status."  
**Expected Agent:** DataAnalyst (fallback)  
**Expected Result:**
- JSON parsing and flattening
- List of warning sensors: S003 (pressure), S006 (humidity)
- Plant locations where warnings occur
- Sensor types and values
**Complexity:** Medium

### Q7.2 - Multi-Level Aggregation (Advanced)
**File:** nested_manufacturing.json  
**Question:** "Calculate average efficiency_rate across all plants and identify the best performing plant."  
**Expected Agent:** FinancialAgent or DataAnalyst  
**Expected Result:**
- Efficiency rates: P001=0.87, P002=0.92, P003=0.95
- Average: 0.913
- Best: P003 (Berlin, Pharmaceuticals) at 0.95
- Analysis of why pharmaceutical has highest efficiency
**Complexity:** High

### Q7.3 - Supply Chain Analysis (Advanced)
**File:** nested_manufacturing.json  
**Question:** "Analyze supplier reliability and recommend which supplier to prioritize based on lead time and reliability score."  
**Expected Agent:** FinancialAgent  
**Expected Result:**
- Supplier comparison table
- Score calculation: reliability_score / lead_time
- Recommendation: SUP002 (balance) or SUP003 (highest reliability despite longer lead time)
- Risk assessment
**Complexity:** High

---

## SECTION 8: EDGE CASES & ERROR HANDLING

### Q8.1 - Missing Values (Edge Case)
**File:** edge_cases/null_values.json  
**Question:** "Analyze this data and handle null values appropriately."  
**Expected Agent:** DataAnalyst  
**Expected Result:**
- Identification of null/missing values
- Strategy: imputation, deletion, or flagging
- Impact assessment
- Cleaned data statistics
**Complexity:** Medium

### Q8.2 - Mixed Data Types (Edge Case)
**File:** edge_cases/mixed_types.json  
**Question:** "Calculate summary statistics despite mixed data types in value field."  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- Type detection and coercion
- Handling of "N/A" string values
- Numeric conversion where possible
- Statistics on valid numeric values only
- Warning about data quality
**Complexity:** Medium

### Q8.3 - Deep Nesting (Edge Case)
**File:** edge_cases/deep_nested.json  
**Question:** "Extract insights from this deeply nested structure."  
**Expected Agent:** DataAnalyst  
**Expected Result:**
- Successful parsing of nested levels
- Flattened data representation
- Key metrics extracted
- Structure visualization
**Complexity:** High

### Q8.4 - Unicode Data (Edge Case)
**File:** edge_cases/unicode_data.json  
**Question:** "Process and analyze this data with international characters."  
**Expected Agent:** DataAnalyst  
**Expected Result:**
- Proper UTF-8 handling
- No encoding errors
- Correct display of special characters
- Analysis regardless of character set
**Complexity:** Medium

---

## SECTION 9: HEALTHCARE DOMAIN SPECIFIC

### Q9.1 - Readmission Analysis (Domain-Specific)
**File:** healthcare_patients.csv  
**Question:** "What factors correlate with patient readmission? Identify high-risk profiles."  
**Expected Agent:** StatisticalAgent + MLInsightsAgent  
**Expected Result:**
- Correlation analysis with readmission flag
- Key factors: severity, comorbidities, hospital_stay_days
- Logistic regression or decision tree
- Risk scoring model
- Clinical recommendations
**Complexity:** High

### Q9.2 - Cost vs Outcome (Domain-Specific)
**File:** healthcare_patients.csv  
**Question:** "Is there a relationship between treatment cost and patient outcomes (severity, readmission)?"  
**Expected Agent:** StatisticalAgent + FinancialAgent  
**Expected Result:**
- Scatter plot: cost vs severity
- Correlation analysis
- Cost-effectiveness assessment
- Identification of high-cost low-outcome cases
- Healthcare policy implications
**Complexity:** High

### Q9.3 - Comorbidity Patterns (Advanced)
**File:** healthcare_patients.csv  
**Question:** "Identify common comorbidity combinations and their impact on treatment cost and hospital stay."  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Text parsing of comorbidities field
- Association rule mining
- Common patterns: Diabetes+Hypertension, Hypertension+Heart Disease
- Cost and stay duration by pattern
- Clinical insights
**Complexity:** High

---

## SECTION 10: EDUCATIONAL DOMAIN SPECIFIC

### Q10.1 - Academic Performance Drivers (Domain-Specific)
**File:** university_academic_data.csv  
**Question:** "What factors most influence final_score? Build a predictive model."  
**Expected Agent:** MLInsightsAgent  
**Expected Result:**
- Regression model predicting final_score
- Top features: midterm_score, study_hours_week, attendance_rate, gpa
- R² score (expected >0.80)
- Insights for educational intervention
**Complexity:** High

### Q10.2 - Grade Distribution by Major (Domain-Specific)
**File:** university_academic_data.csv  
**Question:** "Compare GPA distributions across majors. Are STEM majors harder?"  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- GPA statistics by course_category
- ANOVA test results
- Distribution visualizations
- Conclusion: STEM often shows bimodal distribution
- Grade inflation analysis
**Complexity:** Medium

### Q10.3 - At-Risk Student Identification (Advanced)
**File:** university_academic_data.csv  
**Question:** "Identify students at risk of poor performance. What intervention strategies would help?"  
**Expected Agent:** MLInsightsAgent + DataAnalyst  
**Expected Result:**
- Risk factors: low attendance_rate, high commute_time, part_time_job=Yes
- Classification model for risk prediction
- Student IDs flagged
- Personalized intervention recommendations
**Complexity:** High

---

## SECTION 11: VISUALIZATION REQUESTS

### Q11.1 - Simple Chart (Simple)
**File:** sales_data.csv  
**Question:** "Create a bar chart showing total sales by region."  
**Expected Agent:** Visualizer (if separate) or DataAnalyst  
**Expected Result:**
- Bar chart specification
- X-axis: region, Y-axis: sum of sales
- 4 bars (North, South, East, West)
- Color coding
- Chart title and labels
**Complexity:** Low

### Q11.2 - Multi-Series Line Chart (Medium)
**File:** time_series_stock.csv  
**Question:** "Plot closing prices for all sectors over time on the same chart."  
**Expected Agent:** Visualizer + TimeSeriesAgent  
**Expected Result:**
- Line chart with 4 series
- Date on X-axis, price on Y-axis
- Legend identifying each sector
- Color differentiation
- Trend comparison visible
**Complexity:** Medium

### Q11.3 - Heatmap (Advanced)
**File:** comprehensive_ecommerce.csv  
**Question:** "Create a heatmap showing correlation between numeric variables."  
**Expected Agent:** StatisticalAgent + Visualizer  
**Expected Result:**
- Correlation matrix calculated
- Heatmap with color gradient
- Values: quantity, unit_price, total_amount, discount_percent, shipping_cost
- Color scale (red-white-blue or similar)
- Annotations with correlation values
**Complexity:** High

---

## SECTION 12: NATURAL LANGUAGE COMPLEXITY

### Q12.1 - Ambiguous Query (Challenging)
**Question:** "Show me the best products."  
**File:** comprehensive_ecommerce.csv  
**Expected Agent:** DataAnalyst (with clarification)  
**Expected Result:**
- Interpretation needed: best by revenue? quantity? profit margin?
- System should either:
  a) Ask clarification, OR
  b) Provide multiple interpretations
- Expected: products ranked by total_amount (revenue)
**Complexity:** High

### Q12.2 - Multi-Part Query (Challenging)
**Question:** "For electronics in the ecommerce data, calculate average price, identify outliers, predict next month's sales, and visualize the trend."  
**File:** comprehensive_ecommerce.csv  
**Expected Agents:** Multiple (DataAnalyst → StatisticalAgent → TimeSeriesAgent → Visualizer)  
**Expected Result:**
- Filter: product_category='Electronics'
- Average unit_price
- Outlier detection
- Sales forecast (if dates sufficient)
- Trend visualization
- Comprehensive report combining all analyses
**Complexity:** Very High

### Q12.3 - Comparative Query (Challenging)
**Question:** "Compare the statistical properties of stress levels between students with part-time jobs versus those without."  
**File:** university_academic_data.csv  
**Expected Agent:** StatisticalAgent  
**Expected Result:**
- Group comparison (part_time_job: Yes vs No)
- T-test or Mann-Whitney U test
- Descriptive stats for both groups
- Effect size
- Interpretation and recommendations
**Complexity:** High

---

## SECTION 13: EXTREME EDGE CASES

### Q13.1 - Empty Dataset
**File:** edge_cases/empty_array.json  
**Question:** "Analyze this data."  
**Expected Result:**
- Graceful error handling
- Message: "Dataset is empty, no analysis possible"
- No crash or exception exposed to user
**Complexity:** Edge Case

### Q13.2 - Malformed Data
**File:** malformed.json  
**Question:** "Process this JSON file."  
**Expected Result:**
- JSON parsing error detected
- Repair attempt using repair_json function
- If repair successful: analysis proceeds
- If repair fails: clear error message
- No system crash
**Complexity:** Edge Case

### Q13.3 - Very Large Values
**Question:** "What if order amounts are in billions?"  
**File:** Custom data with extreme values  
**Expected Result:**
- Proper numeric handling (no overflow)
- Appropriate formatting (1,234,567,890.00)
- Statistical measures still valid
- Visualization scales appropriately
**Complexity:** Edge Case

---

## SECTION 14: PERFORMANCE & OPTIMIZATION

### Q14.1 - Large Dataset Handling
**File:** Large CSV (>10,000 rows)  
**Question:** "Calculate summary statistics."  
**Expected Result:**
- Efficient data loading (pandas chunks if needed)
- Memory optimization
- Response time <30 seconds
- Accurate results despite size
**Complexity:** Performance Test

### Q14.2 - Complex Calculation
**File:** hr_employee_data.csv  
**Question:** "For each employee, calculate: adjusted salary (with bonus), years of service, productivity score (projects/training hours), and rank within department."  
**Expected Agent:** DataAnalyst + FinancialAgent  
**Expected Result:**
- Multiple calculated fields
- Date parsing for years of service
- Complex ranking logic
- Efficient computation
- Result table with all new fields
**Complexity:** High

---

## SECTION 15: DOMAIN AGNOSTIC TESTS

### Q15.1 - Unknown Domain
**File:** Custom domain data (e.g., agricultural yields)  
**Question:** "Analyze crop yield patterns."  
**Expected Agent:** DynamicPlanner → DataAnalyst  
**Expected Result:**
- System recognizes unknown domain
- Dynamic plan generated
- Generic statistical analysis applied
- Insights provided without domain-specific hardcoding
- Proof of domain-agnostic architecture
**Complexity:** High

### Q15.2 - Mixed Domain
**File:** Multiple files from different domains  
**Question:** "Compare patterns across healthcare and ecommerce data."  
**Expected Result:**
- Multi-file processing
- Domain-specific analysis for each
- Cross-domain comparison where applicable
- Synthesis of insights
**Complexity:** Very High

---

## SECTION 16: MODEL ROUTING TESTS

### Q16.1 - Complexity-Based Routing
**Question:** "What's 2+2?" vs "Perform multivariate regression with cross-validation"  
**Expected Behavior:**
- Simple query → Fast, lightweight model (e.g., qwen2.5:1.5b)
- Complex query → Powerful model (e.g., deepseek-r1:32b)
- QueryOrchestrator makes intelligent decision
**Complexity:** Routing Test

### Q16.2 - Domain-Based Routing
**Question:** "Generate Python code to analyze this data" vs "Summarize this table"  
**Expected Behavior:**
- Code generation → Code-specialized model (deepseek-r1)
- Summarization → General model
- Model selection based on intent
**Complexity:** Routing Test

---

## SECTION 17: ERROR RECOVERY

### Q17.1 - Invalid File Reference
**Question:** "Analyze nonexistent_file.csv"  
**Expected Result:**
- Error: "File not found"
- Suggestion: List available files
- No system crash
**Complexity:** Error Handling

### Q17.2 - Invalid Column Reference
**File:** sales_data.csv  
**Question:** "Show me the average customer_age"  
**Expected Result:**
- Error: "Column 'customer_age' not found"
- Available columns listed
- Suggestion: Did you mean 'product' or 'region'?
**Complexity:** Error Handling

### Q17.3 - Computational Error
**Question:** "Divide sales by zero"  
**Expected Result:**
- Mathematical error caught
- Explanation of divide-by-zero issue
- Alternative computation suggested
**Complexity:** Error Handling

---

## SECTION 18: INTEGRATION TESTS

### Q18.1 - Upload + Analyze Flow
**Actions:**
1. Upload comprehensive_ecommerce.csv
2. Query: "What's the total revenue?"  
**Expected Result:**
- File successfully uploaded
- Analysis service retrieves file
- Correct calculation returned
**Complexity:** Integration

### Q18.2 - Session Continuity
**Actions:**
1. Query: "Analyze sales_data.csv"
2. Query: "Now show me just the North region"  
**Expected Result:**
- Second query understands context
- Filters applied to same dataset
- No need to re-specify file
**Complexity:** Integration

---

## SECTION 19: REPORTING

### Q19.1 - Comprehensive Report
**File:** healthcare_patients.csv  
**Question:** "Generate a comprehensive analysis report covering demographics, costs, outcomes, and recommendations."  
**Expected Agent:** Reporter (coordinator)  
**Expected Result:**
- Multi-section report
- Demographics: age, gender distribution
- Financial: cost analysis
- Clinical: diagnosis, severity, readmission
- Visualizations embedded
- Executive summary
- Detailed recommendations
**Complexity:** Very High

### Q19.2 - Executive Summary
**File:** comprehensive_ecommerce.csv  
**Question:** "Give me an executive summary of business performance."  
**Expected Agent:** Reporter + FinancialAgent  
**Expected Result:**
- High-level metrics only
- Total revenue, growth rate, top categories
- Key insights (3-5 bullets)
- Concise format (<500 words)
- Actionable recommendations
**Complexity:** High

---

## SECTION 20: STRESS TESTS

### Q20.1 - Concurrent Queries
**Actions:** Submit 10 queries simultaneously  
**Expected Result:**
- All queries processed (may be queued)
- No data corruption
- Accurate results for each
- Reasonable response times
**Complexity:** Stress Test

### Q20.2 - Memory Pressure
**Actions:** Process very large dataset under low memory  
**Expected Result:**
- System detects memory constraints
- May downgrade model or chunk data
- Completes successfully or fails gracefully
- No system crash
**Complexity:** Stress Test

---

## SUMMARY OF EXPECTED AGENT ROUTING

| Query Type | Primary Agent | Secondary Agent |
|------------|---------------|-----------------|
| Descriptive stats | StatisticalAgent | - |
| Hypothesis testing | StatisticalAgent | - |
| Revenue analysis | FinancialAgent | - |
| Profitability | FinancialAgent | - |
| Time trends | TimeSeriesAgent | - |
| Forecasting | TimeSeriesAgent | MLInsightsAgent |
| Clustering | MLInsightsAgent | - |
| Classification | MLInsightsAgent | - |
| SQL queries | SQLAgent | - |
| General analysis | DataAnalyst | - |
| Visualization | Visualizer | (agent providing data) |
| Reports | Reporter | (coordinates others) |

---

## TESTING METHODOLOGY

### Validation Criteria:
1. **Agent Selection**: Correct agent routed based on query intent
2. **Accuracy**: Results mathematically correct
3. **Completeness**: All requested metrics provided
4. **Interpretation**: Human-readable explanation included
5. **Error Handling**: Graceful failures with helpful messages
6. **Performance**: Response time <60s for complex queries, <10s for simple
7. **Visualization**: Appropriate chart types recommended/generated
8. **Domain Agnostic**: No hardcoded domain assumptions

### Test Execution:
- Run queries via API: POST /api/analyze
- Check response structure matches AnalyzeResponse model
- Validate agent field matches expected agent
- Verify result field contains expected data
- Check interpretation field for human-readable text

---

## NOTES ON EXPECTED BEHAVIOR

1. **Statistical Agent** should automatically check assumptions (normality, homogeneity of variance) before tests
2. **Financial Agent** should provide currency formatting and business context
3. **Time Series Agent** should test for stationarity before ARIMA
4. **ML Insights Agent** should perform train/test split and cross-validation
5. **SQL Agent** should handle CSV-to-SQL conversion transparently
6. **Query Orchestrator** should select appropriate models based on complexity
7. **All agents** should provide confidence scores or uncertainty measures where applicable

---

## EDGE CASE EXPECTATIONS

- **Null values**: Imputation or exclusion with explanation
- **Mixed types**: Type coercion with warnings
- **Outliers**: Identification and impact analysis
- **Small samples**: Warning about statistical power
- **Multicollinearity**: Detection in regression
- **Imbalanced classes**: Recognition in classification
- **Non-stationary time series**: Differencing applied
- **Missing files**: Clear error with file list

---

**End of Test Questions Document**  
**Total Questions: 85+**  
**Complexity Range: Low to Very High**  
**Coverage: All 5 plugin agents + core functionality + edge cases**
