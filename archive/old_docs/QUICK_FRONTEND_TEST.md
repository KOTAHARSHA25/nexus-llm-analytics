# 🎯 Quick Frontend Testing Reference

## Sample Files Location
```
data/samples/
├── test_sales_monthly.csv      (Time series - best for LINE charts)
├── test_employee_data.csv      (Correlations - best for SCATTER)
├── test_iot_sensor.csv         (Sensor data - best for LINE)
├── test_student_grades.csv     (Comparisons - best for BAR)
└── test_inventory.csv          (Stock levels - best for BAR/PIE)
```

## Quick Start Commands

**Terminal 1 (Backend):**
```bash
python scripts/launch.py
```

**Terminal 2 (Frontend):**
```bash
cd src\frontend
npm run dev
```

**Browser:** `http://localhost:3000`

## Test Queries to Try

### Sales Data (test_sales_monthly.csv)
- "Analyze sales trends" → LINE chart
- "Show revenue by region" → BAR chart
- "Compare product categories" → BAR/PIE chart

### Employee Data (test_employee_data.csv)
- "Analyze salary patterns" → SCATTER plot
- "Compare salaries by department" → BAR chart
- "Show department distribution" → PIE chart

### Sensor Data (test_iot_sensor.csv)
- "Show temperature changes" → LINE chart
- "Temperature trends over time" → LINE chart

### Student Grades (test_student_grades.csv)
- "Analyze student performance" → BAR chart
- "Show grade distribution" → PIE chart
- "Compare subjects" → BAR chart

### Inventory (test_inventory.csv)
- "Show inventory status" → BAR chart
- "Stock levels by category" → BAR/PIE chart

## What to Look For

✅ **Smart Chart Suggestions panel** (above chart)
✅ **3 recommendations** with priority scores
✅ **Recommended chart highlighted** in blue
✅ **Chart displays** correctly below
✅ **Chart type badge** matches suggestion
✅ **Interactive features** work (hover, zoom)
✅ **Download/Fullscreen buttons** work

## Force Specific Chart Types

Add keywords to your query:
- "**bar chart**" → Forces BAR
- "**line chart**" → Forces LINE
- "**pie chart**" → Forces PIE
- "**scatter plot**" → Forces SCATTER
- "**histogram**" → Forces HISTOGRAM
- "**box plot**" → Forces BOX

## Expected Timings

- Upload: < 1 sec
- Analysis: < 2 sec
- Chart generation: < 3 sec
- **Total: < 5 sec**

## Common Issues

**Chart not showing?**
→ Check browser console, verify backend running

**Suggestions missing?**
→ Check Network tab for `/visualize/suggestions` call

**Wrong chart type?**
→ Add chart type keyword to query

## Success = All Green ✅

- [ ] 5/5 datasets upload successfully
- [ ] Suggestions panel appears every time
- [ ] Charts render correctly
- [ ] Interactive features work
- [ ] Regenerate creates identical chart
- [ ] No console errors

**Once complete → Proceed to Task 4.2: Report Generation**
