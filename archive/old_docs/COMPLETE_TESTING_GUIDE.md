# 🧪 COMPLETE MECHANISM TESTING GUIDE
## Verify All Mechanisms Work Together

**Purpose:** Test Smart Model Selection + Intelligent Routing + CoT Self-Correction

**Date:** November 14, 2025

**⚠️ IMPORTANT:** The routing logs appear when the ANALYZE endpoint processes your query.
Make sure you're submitting queries through the MAIN text input, not just viewing visualizations!

---

## 🚀 QUICK START

### 1. Start Backend (Terminal 1)
```powershell
cd src\backend
python -m uvicorn main:app --reload
```
✅ Wait for: `Application startup complete.`

### 2. Start Frontend (Terminal 2)
```powershell
cd src\frontend
npm run dev
```
✅ Wait for: `Local: http://localhost:5173/`

### 3. Open Browser
Go to `http://localhost:5173`

### 4. Clear Cache (Optional but Recommended)
```powershell
# In root directory
python clear_cache.py
```
✅ This ensures routing logs will appear (cached results skip routing)

---

## 📊 TEST DATA GROUND TRUTH

File: `data/samples/sales_data.csv`

**Expected Answers:**
- Total rows: **100**
- Unique products: **5** (Widget A-E)
- Total revenue: **$2,563,044**
- Regions: **4** (North, South, East, West)
- Average price: **$61.17**
- Top region: **North**

---

## 🧪 SCENARIO 1: ALL ENABLED ⭐

### Configure
1. Click **Settings** tab
2. ✅ **Smart Model Selection:** ON
3. ✅ **Intelligent Routing:** ON
4. **Save**

### Upload
Select: `data\samples\sales_data.csv`

**📝 HOW TO SUBMIT QUERIES:**
1. **Type your query** in the main input text box (the large text area at the top)
2. **Click "Submit"** button (NOT "Analyze" or other buttons)
3. **Watch the backend terminal** for routing logs to appear
4. **Wait for the response** to appear in the Results section

⚠️ **Common Mistake:** Don't just click visualization buttons - type and submit the full query text!

---

### Query 1: COUNT (EASY) ⚡

**Type:** `How many rows are in this dataset?`

**Expected:**
- ✅ Answer: 100
- ⏱️ Time: 1-2s
- 🤖 Model: tinyllama

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.107
   ⚡ Tier: FAST
   🤖 Model: tinyllama:latest
   ⏱️  Expected: 1-3s
```

**Checklist:**
- [ ] Answer correct (100)
- [ ] Time < 3s
- [ ] Backend shows tinyllama
- [ ] Backend shows FAST tier

---

### Query 2: COUNT (EASY) ⚡

**Type:** `Count the unique products`

**Expected:**
- ✅ Answer: 5
- ⏱️ Time: 1-2s
- 🤖 Model: tinyllama

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.115
   ⚡ Tier: FAST
   🤖 Model: tinyllama:latest
   ⏱️  Expected: 1-3s
```

**Checklist:**
- [ ] Answer correct (5)
- [ ] Time < 3s
- [ ] Backend shows tinyllama
- [ ] Backend shows FAST tier

---

### Query 3: SUM (EASY) ⚡

**Type:** `What is the sum of revenue?`

**Expected:**
- ✅ Answer: ~$2.5M
- ⏱️ Time: 1-2s
- 🤖 Model: tinyllama

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.133
   ⚡ Tier: FAST
   🤖 Model: tinyllama:latest
   ⏱️  Expected: 1-3s
```

**Checklist:**
- [ ] Answer around $2,563,044
- [ ] Time < 3s
- [ ] Backend shows tinyllama
- [ ] Backend shows FAST tier

---

### Query 4: AGGREGATION (MEDIUM) ⚖️

**Type:** `Show average sales by region`

**Expected:**
- ✅ Answer: 4 regions with averages
- ⏱️ Time: 3-5s
- 🤖 Model: phi3:mini

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.420
   ⚖️  Tier: BALANCED
   � Model: phi3:mini
   ⏱️  Expected: 3-6s
```

**Checklist:**
- [ ] Shows all 4 regions
- [ ] Time 3-6s
- [ ] Backend shows phi3:mini
- [ ] Backend shows BALANCED tier

---

### Query 5: FILTER (MEDIUM) ⚖️

**Type:** `Which product has highest revenue?`

**Expected:**
- ✅ Answer: Product_A ($540,120)
- ⏱️ Time: 3-5s
- 🤖 Model: phi3:mini

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.444
   ⚖️  Tier: BALANCED
   🤖 Model: phi3:mini
   ⏱️  Expected: 3-6s
```

**Checklist:**
- [ ] Identifies Product_A
- [ ] Time 3-6s
- [ ] Backend shows phi3:mini
- [ ] Backend shows BALANCED tier

---

### Query 6: ANALYSIS (COMPLEX) 🚀🧠

**Type:** `Which region has best sales and why?`

**Expected:**
- ✅ Answer: North + reasoning
- ⏱️ Time: 10-15s
- 🤖 Model: llama3.1:8b
- 🧠 CoT: ACTIVATED

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.720
   🚀 Tier: FULL_POWER
   � Model: llama3.1:8b
   ⏱️  Expected: 8-15s

═══════════════════════════════════
🧠 CoT SELF-CORRECTION ACTIVATED
   📊 Complexity: 0.720 (threshold: 0.5)
   🤖 Generator: llama3.1:8b
   🔍 Critic: phi3:mini
   ⚙️  Max Iterations: 2
═══════════════════════════════════
```

**Checklist:**
- [ ] Answer says "North"
- [ ] Includes reasoning
- [ ] Time 8-15s
- [ ] Backend shows llama3.1:8b
- [ ] Backend shows FULL_POWER tier
- [ ] Backend shows CoT activation box

---

### Query 7: CORRELATION (COMPLEX) 🚀🧠

**Type:** `Find correlation between price and revenue`

**Expected:**
- ✅ Answer: Positive/negative mentioned
- ⏱️ Time: 10-15s
- 🤖 Model: llama3.1:8b
- 🧠 CoT: ACTIVATED

**Checklist:**
**Expected:**
- ✅ Answer: Positive/negative mentioned
- ⏱️ Time: 10-15s
- 🤖 Model: llama3.1:8b
- 🧠 CoT: ACTIVATED

**Backend Terminal Shows:**
```
🎯 [INTELLIGENT ROUTING] Complexity: 0.765
   🚀 Tier: FULL_POWER
   🤖 Model: llama3.1:8b
   ⏱️  Expected: 8-15s

═══════════════════════════════════
🧠 CoT SELF-CORRECTION ACTIVATED
   📊 Complexity: 0.765 (threshold: 0.5)
   🤖 Generator: llama3.1:8b
   🔍 Critic: phi3:mini
   ⚙️  Max Iterations: 2
═══════════════════════════════════
```

**Checklist:**
- [ ] Mentions correlation type
- [ ] Time 8-15s
- [ ] Backend shows llama3.1:8b
- [ ] Backend shows FULL_POWER tier
- [ ] Backend shows CoT activation box

---

## 📊 SCENARIO 1 RESULTS

| # | Query | Time | Model | Correct | CoT |
|---|-------|------|-------|---------|-----|
| 1 | Row count | ___s | _______ | ☐ | ☐ |
| 2 | Product count | ___s | _______ | ☐ | ☐ |
| 3 | Revenue sum | ___s | _______ | ☐ | ☐ |
| 4 | Avg by region | ___s | _______ | ☐ | ☐ |
| 5 | Highest revenue | ___s | _______ | ☐ | ☐ |
| 6 | Best region | ___s | _______ | ☐ | ☑ |
| 7 | Correlation | ___s | _______ | ☐ | ☑ |

**Total Time:** _____ s (Expected: 40-50s)

**CoT Triggered:** ___ times (Expected: 2)

---

## 🧪 SCENARIO 2: ROUTING OFF

### Configure
- ✅ Smart Selection: ON
- ❌ Routing: OFF

### Expected
- All use phi3:mini
- CoT still works for complex queries

**Run all 7 queries, record:**
- Total time: _____ s (Expected: 50-60s)
- CoT triggers: ___ times (Expected: 2)

---

## 🧪 SCENARIO 3: ALL OFF

### Configure
- ❌ Smart Selection: OFF
- ❌ Routing: OFF
- Manual: llama3.1:8b

### Expected
- All use llama3.1:8b
- Slower but consistent

**Run all 7 queries, record:**
- Total time: _____ s (Expected: ~70s)

---

## ✅ SUCCESS CRITERIA

**Scenario 1 (Optimal):**
- [ ] Queries 1-3: tinyllama, <3s each
- [ ] Queries 4-5: phi3, 3-6s each
- [ ] Queries 6-7: llama3.1, 8-15s each
- [ ] CoT triggers 2 times
- [ ] Total: 40-50s
- [ ] All accurate

**Performance Gain:**
- Scenario 1 vs 3: ____ % faster
- Expected: ~40% faster

---

## 🐛 TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| Wrong model | Check Settings saved, restart backend |
| CoT not showing | Check backend terminal logs |
| All queries slow | Routing might be OFF |
| Backend not logging | Check terminal window with uvicorn |

---

## 📋 FINAL CHECKLIST

- [ ] ✅ All 3 scenarios tested
- [ ] ✅ Backend logs visible in terminal
- [ ] ✅ Models switch correctly
- [ ] ✅ CoT triggers on complex queries
- [ ] ✅ Performance improvement verified
- [ ] ✅ Accuracy maintained

**🎉 TEST COMPLETE!**

---

## 📸 PROOF OF SUCCESS

Check backend terminal for:
1. Routing tier indicators (⚡/⚖️/🚀)
2. CoT activation boxes for complex queries
3. Model names matching expectations
4. Timing matching tier expectations

Document results in the tables above to verify the system works correctly!
