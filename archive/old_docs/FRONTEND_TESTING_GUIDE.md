# Frontend Verification Guide - Phase 4.1 Visualization

**Date:** October 27, 2025  
**Purpose:** Manual testing of Phase 4.1 visualization features through the frontend UI

---

## Sample Datasets Created

All files are in `data/samples/`:

1. **test_sales_monthly.csv** - Sales data with time series
   - 24 rows, 6 columns
   - Contains: dates, regions, categories, numeric metrics
   - Best for: Line charts (time trends), bar charts (comparisons)

2. **test_employee_data.csv** - HR/People analytics
   - 20 rows, 8 columns
   - Contains: employee info, departments, salaries, ratings
   - Best for: Scatter plots (correlation), bar charts (by department)

3. **test_iot_sensor.csv** - Time series sensor data
   - 20 rows, 6 columns
   - Contains: timestamps, temperature, humidity, pressure
   - Best for: Line charts (sensor trends over time)

4. **test_student_grades.csv** - Academic performance
   - 18 rows, 7 columns
   - Contains: student data, subjects, scores, grades
   - Best for: Bar charts (grade comparison), pie charts (grade distribution)

5. **test_inventory.csv** - Product inventory
   - 15 rows, 7 columns
   - Contains: products, categories, stock, prices
   - Best for: Bar charts (stock levels), pie charts (category distribution)

---

## Frontend Testing Steps

### 1. Start the Application

**Terminal 1 - Backend:**
```bash
cd c:\Users\mitta\OneDrive\Documents\nexus-llm-analytics-dist\nexus-llm-analytics-dist
python scripts/launch.py
```

**Terminal 2 - Frontend:**
```bash
cd c:\Users\mitta\OneDrive\Documents\nexus-llm-analytics-dist\nexus-llm-analytics-dist\src\frontend
npm run dev
```

**Access:** Open browser to `http://localhost:3000`

---

### 2. Test Case 1: Sales Data with Auto Chart Selection

**Dataset:** `test_sales_monthly.csv`

**Steps:**
1. Click "Upload File" button
2. Select `data/samples/test_sales_monthly.csv`
3. Wait for upload to complete
4. In query box, type: "Analyze sales trends"
5. Click "Analyze" button
6. Switch to "Charts" tab

**Expected Results:**
- ✅ "Smart Chart Suggestions" panel appears
- ✅ Shows 3 chart recommendations with priorities
- ✅ Line chart suggested (highest priority) for time series
- ✅ Chart displays below with revenue/units trend over months
- ✅ Chart type badge shows "LINE"
- ✅ Chart is interactive (hover shows values)

**Test Different Queries:**
- "Show revenue by region" → Should suggest BAR chart
- "Compare product categories" → Should suggest BAR/PIE chart
- "Revenue trends over time" → Should suggest LINE chart

---

### 3. Test Case 2: Employee Data with Correlations

**Dataset:** `test_employee_data.csv`

**Steps:**
1. Upload `data/samples/test_employee_data.csv`
2. Query: "Analyze employee salary patterns"
3. Go to "Charts" tab

**Expected Results:**
- ✅ Suggestions show SCATTER plot for correlation analysis
- ✅ Chart shows salary vs experience/age correlation
- ✅ Multiple chart options presented

**Alternative Queries:**
- "Compare salaries by department" → BAR chart
- "Show department distribution" → PIE chart
- "Salary vs performance relationship" → SCATTER plot

---

### 4. Test Case 3: Time Series Sensor Data

**Dataset:** `test_iot_sensor.csv`

**Steps:**
1. Upload `data/samples/test_iot_sensor.csv`
2. Query: "Show temperature changes over time"
3. Check "Charts" tab

**Expected Results:**
- ✅ LINE chart recommended (95/100 priority for time series)
- ✅ Temperature trend displayed clearly
- ✅ X-axis shows timestamps
- ✅ Y-axis shows temperature values
- ✅ Smooth line connecting data points

---

### 5. Test Case 4: Student Grades

**Dataset:** `test_student_grades.csv`

**Steps:**
1. Upload `data/samples/test_student_grades.csv`
2. Query: "Analyze student performance"
3. View "Charts" tab

**Expected Results:**
- ✅ BAR chart for grade comparisons
- ✅ Suggestions include pie chart for distribution
- ✅ Chart shows scores by student/subject

**Try:**
- "Show grade distribution" → PIE chart
- "Compare subjects" → BAR chart
- "Student performance by subject" → BAR/LINE chart

---

### 6. Test Case 5: Inventory Stock Levels

**Dataset:** `test_inventory.csv`

**Steps:**
1. Upload `data/samples/test_inventory.csv`
2. Query: "Show inventory status"
3. Check visualizations

**Expected Results:**
- ✅ BAR chart showing stock quantities
- ✅ PIE chart option for category distribution
- ✅ Clear product names on X-axis

---

## Verification Checklist

### Chart Suggestions Panel
- [ ] Panel appears for all datasets
- [ ] Shows exactly 3 suggestions
- [ ] Each suggestion has:
  - [ ] Chart type badge (BAR, LINE, etc.)
  - [ ] Priority score (0-100)
  - [ ] Clear reasoning text
  - [ ] Use case description
- [ ] Recommended chart is highlighted in blue
- [ ] Suggestions are ranked by priority

### Chart Display
- [ ] Chart renders correctly for all datasets
- [ ] Chart type badge matches suggestion
- [ ] Download button works (saves PNG)
- [ ] Fullscreen button works
- [ ] Chart is interactive:
  - [ ] Hover shows data values
  - [ ] Zoom in/out works
  - [ ] Pan works (if applicable)
- [ ] Chart resizes with window

### Auto-Selection Logic
- [ ] Time series data → LINE chart suggested
- [ ] Categorical data → BAR chart suggested
- [ ] Correlation data → SCATTER plot suggested
- [ ] Proportional data → PIE chart option available
- [ ] Distribution data → HISTOGRAM option available

### Determinism
- [ ] Same query produces same chart every time
- [ ] Regenerate button creates identical chart
- [ ] Suggestions order never changes
- [ ] No random variations in output

### Responsive Design
- [ ] Works on different screen sizes
- [ ] Mobile-friendly (if applicable)
- [ ] No UI breaks or overlaps
- [ ] Readable text at all sizes

---

## Common Issues & Solutions

### Issue: Chart Not Displaying
**Solution:**
- Check browser console for errors
- Verify Plotly.js loaded (look for "Loading chart library..." message)
- Ensure backend is running on port 8000

### Issue: Suggestions Not Appearing
**Solution:**
- Check that `/visualize/suggestions` endpoint is working
- Verify file was uploaded successfully
- Check Network tab for API call status

### Issue: Wrong Chart Type Selected
**Solution:**
- This is expected - system auto-selects based on data
- Use specific keywords in query to force chart type:
  - "bar chart" → BAR
  - "line chart" → LINE  
  - "pie chart" → PIE
  - "scatter plot" → SCATTER

---

## Performance Expectations

- **Chart Generation:** < 3 seconds
- **Suggestions API:** < 1 second
- **Chart Rendering:** < 1 second
- **Total Time (upload to display):** < 5 seconds

---

## Test Results Template

```
Dataset: _________________
Query: _________________
Date/Time: _________________

✅ / ❌  Suggestions panel displayed
✅ / ❌  Correct chart type recommended
✅ / ❌  Chart rendered successfully
✅ / ❌  Interactive features work
✅ / ❌  Regenerate produces identical result

Notes:
_________________________________
_________________________________
```

---

## Success Criteria

Phase 4.1 frontend is considered VERIFIED if:

1. ✅ All 5 sample datasets display charts correctly
2. ✅ Smart suggestions appear for all datasets
3. ✅ Auto-selection chooses appropriate chart types
4. ✅ Charts are interactive and downloadable
5. ✅ Deterministic behavior (same input → same output)
6. ✅ No errors in browser console
7. ✅ Performance meets expectations (< 5 sec total)

---

## Next Steps After Verification

Once frontend testing is complete:
1. Document any issues found
2. Fix critical bugs
3. Proceed to **Task 4.2: Report Generation**
4. Begin integrating charts into PDF/Excel reports
