# 🧪 Multi-Agent Demo Scenario: E-Commerce Analytics

This guide validates the multi-agent capabilities of Nexus LLM Analytics using a custom dataset.

**Dataset**: [`data/samples/MultiAgent_Demo_Data.csv`](../data/samples/MultiAgent_Demo_Data.csv)  
**Contains**: Sales data with Regions, Categories, Costs, and Ratings.

---

## 🤖 Scenario 1: The Statistical Agent (Hypothesis Testing)
**Goal:** Verify if the system routes to the Statistical Agent for significance testing.

*   **Question**: "Is there a statistically significant difference in Sales between the 'North' and 'South' regions?"
*   **Routing**: Query Orchestrator -> Statistical Agent (`t-test`)
*   **Correct Logic**:
    1.  Filter data for Region='North' and Region='South'.
    2.  Extract 'Sales' column for both.
    3.  Perform independent t-test.
*   **Expected Answer**:
    *   North Mean Sales: ~1295
    *   South Mean Sales: ~1117
    *   P-value: Likely > 0.05 (Not significant due to small sample size), or < 0.05 if variability is low.
    *   *Result*: "The difference is [Significant/Not Significant] with (p=0.xx)."

---

## 📈 Scenario 2: The Time Series Agent (Forecasting)
**Goal:** Verify ARIMA forecasting capabilities.

*   **Question**: "Forecast the total daily sales for the next 3 days."
*   **Routing**: Query Orchestrator -> Time Series Agent
*   **Correct Logic**:
    1.  Aggregate 'Sales' by 'Date'.
    2.  Fit ARIMA model to the daily time series.
    3.  Predict t+1, t+2, t+3.
*   **Expected Answer**:
    *   Should provide 3 date-value pairs.
    *   Look for a plot or a table in the output.

---

## 💰 Scenario 3: The Financial Agent (Profitability)
**Goal:** Verify business metric calculations.

*   **Question**: "Calculate the Profit Margin for each Product Category."
*   **Routing**: Query Orchestrator -> Financial Agent
*   **Correct Logic**:
    1.  `Profit = Sales - Cost`
    2.  `Margin = (Profit / Sales) * 100`
    3.  Group by 'Category'.
*   **Expected Answer**:
    *   **Electronics**:
        *   Total Sales: ~15,870
        *   Total Cost: ~12,380
        *   Profit: ~3,490
        *   **Margin**: ~22%
    *   **Clothing**:
        *   Total Sales: ~3,820
        *   Total Cost: ~1,500
        *   Profit: ~2,320
        *   **Margin**: ~60%

---

## 🧠 Scenario 4: The Dynamic Planner (Complex Logic)
**Goal:** Verify Chain-of-Thought planning for multi-step reasoning.

*   **Question**: "Identify the region with the lowest average Customer Rating, and then calculate what the total revenue would be if we increased that region's sales by 15%."
*   **Routing**: Query Orchestrator -> Dynamic Planner (CoT)
*   **Correct Logic**:
    1.  **Step 1**: Group by `Region`, avg `Customer_Rating`.
        *   North: ~4.5
        *   South: ~3.9
        *   East: ~4.8
        *   West: ~3.75 (**Lowest**)
    2.  **Step 2**: Filter `Region == West`.
    3.  **Step 3**: Sum `Sales` for West (~3,070).
    4.  **Step 4**: Apply increase: `3070 * 1.15`.
*   **Expected Answer**:
    *   Lowest Region: **West**
    *   Current Revenue: 3,070
    *   Projected Revenue: **~3,530.5**

---

## 📊 Scenario 5: Visualizer Agent
**Goal:** Verify plotting capabilities.

*   **Question**: "Plot the daily trend of Sales per Category."
*   **Routing**: Query Orchestrator -> Data Analyst -> Visualization Agent
*   **Expected Answer**:
    *   A line chart with two lines (Electronics, Clothing).
    *   X-axis: Date
    *   Y-axis: Sales

---

## 🚀 Speed Run
1.  Upload `MultiAgent_Demo_Data.csv`.
2.  Copy-paste the questions above.
3.  Verify the agents used in the "Thinking" logs.
