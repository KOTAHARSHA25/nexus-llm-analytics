# ADVANCED FEATURES TEST PLAN
## Comprehensive Testing for CoT, Routing, Plugins, Analysis, Visualization & Reports

---

## 📋 FEATURE ANALYSIS SUMMARY

### 1. **Chain-of-Thought (CoT) System** ✓ UNDERSTOOD
**Purpose**: Self-correction through multi-round reasoning

**Components**:
- `cot_parser.py` - Parses [REASONING] and [OUTPUT] sections
- `cot_critic_prompt.txt` - Critic validation template
- `cot_generator_prompt.txt` - Generator reasoning template
- `cot_review_config.json` - Configuration (enabled: true, max_iterations: 2)

**Flow**:
1. Generator model creates answer with [REASONING] and [OUTPUT]
2. CoT Parser extracts and validates sections
3. Critic model reviews for errors
4. If issues found, generator refines (up to 2 iterations)

**Test Requirements**:
- Parser correctly extracts reasoning
- Critic identifies errors
- Iterations improve quality
- Fallback when CoT unavailable

---

### 2. **Intelligent Routing System** ✓ UNDERSTOOD
**Purpose**: Match query complexity to appropriate model tier

**Components**:
- `intelligent_router.py` - Main routing engine
- `query_complexity_analyzer.py` - Analyzes semantic/data/operation complexity
- `ModelTier` enum: FAST (<0.25), BALANCED (0.25-0.45), FULL_POWER (>0.45)

**Available Models**:
- FAST: tinyllama:latest
- BALANCED: phi3:mini
- FULL_POWER: llama3.1:8b

**Test Requirements**:
- Simple queries → FAST tier
- Medium queries → BALANCED tier
- Complex queries → FULL_POWER tier
- Fallback chain works (FAST → BALANCED → FULL_POWER)
- User override respected
- Statistics tracked

---

### 3. **Plugin Agent System** ✓ UNDERSTOOD
**Purpose**: Specialized agents for different analysis types

**5 Active Agents**:
1. **StatisticalAgent** - Descriptive stats, hypothesis testing, correlation, regression
2. **FinancialAgent** - Revenue, profitability, ROI, financial metrics
3. **TimeSeriesAgent** - Trend detection, seasonality, ARIMA forecasting
4. **MLInsightsAgent** - Clustering, pattern recognition, predictions
5. **SQLAgent** - Database queries

**Discovery**: Automatic scanning of plugins/ directory

**Test Requirements**:
- All 5 agents initialize
- Each handles specific query types
- Scoring/ranking selects best agent
- Execution returns correct results

---

### 4. **Primary + Review Analysis** ✓ UNDERSTOOD
**Purpose**: Two-stage validation for quality assurance

**Workflow**:
1. Primary model (phi3:mini) executes analysis
2. Review model (tinyllama) validates results
3. If issues found, primary refines
4. Combined output with quality score

**Test Requirements**:
- Primary analysis completes
- Review provides feedback
- Refinement improves quality
- Can disable review

---

### 5. **Visualization System** ✓ UNDERSTOOD
**Components**:
- Chart suggestion engine
- Goal-based chart generation
- Multiple libraries (plotly, matplotlib, seaborn)
- Chart types: bar, line, pie, scatter, heatmap, etc.

**Test Requirements**:
- Chart suggestions for data
- Chart generation works
- Different chart types
- Proper JSON format

---

### 6. **Report Generation** ✓ UNDERSTOOD
**Components**:
- Markdown report creation
- PDF/HTML export
- Includes: summary, insights, charts, recommendations

**Test Requirements**:
- Reports generate
- Include all sections
- Export formats work

---

## 🧪 COMPREHENSIVE TEST EXECUTION PLAN

### Phase 1: CoT System Testing
**File**: `test_cot_advanced.py`
- ✅ Parser extracts valid CoT
- ✅ Parser handles missing sections
- ✅ Parser validates reasoning length (min 50 chars)
- ✅ Critic identifies errors
- ✅ Self-correction improves answers
- ✅ Max iterations respected
- ✅ Fallback when CoT unavailable

### Phase 2: Intelligent Routing Testing
**File**: `test_routing_advanced.py`
- ✅ Simple query routes to FAST
- ✅ Medium query routes to BALANCED
- ✅ Complex query routes to FULL_POWER
- ✅ Fallback chain works
- ✅ User override respected
- ✅ Statistics tracking
- ✅ Routing time <100ms

### Phase 3: Plugin Agent Testing
**File**: `test_plugins_advanced.py`
- ✅ All 5 agents initialize
- ✅ Statistical agent: descriptive stats, t-test, correlation
- ✅ Financial agent: revenue, profit margins
- ✅ Time series agent: trends, forecasting
- ✅ ML insights agent: clustering
- ✅ SQL agent: query execution
- ✅ Agent scoring/selection

### Phase 4: Primary+Review Analysis Testing
**File**: `test_analysis_advanced.py`
- ✅ Primary analysis executes
- ✅ Review provides feedback
- ✅ Refinement loop works
- ✅ Quality scores calculated
- ✅ Review can be disabled
- ✅ Error handling

### Phase 5: Visualization Testing
**File**: `test_visualization_advanced.py`
- ✅ Chart suggestions generated
- ✅ Bar charts created
- ✅ Line charts created
- ✅ Pie charts created
- ✅ Scatter plots created
- ✅ Heatmaps created
- ✅ JSON format correct

### Phase 6: Report Generation Testing
**File**: `test_reports_advanced.py`
- ✅ Report structure correct
- ✅ Includes summary
- ✅ Includes insights
- ✅ Includes visualizations
- ✅ Includes recommendations
- ✅ Markdown export
- ✅ PDF export (if available)

---

## 📊 SUCCESS CRITERIA

### Minimum Passing Requirements:
- **CoT System**: 90% accuracy improvement after self-correction
- **Routing**: 85% correct tier selection
- **Plugins**: All 5 agents operational with >80% query match
- **Analysis**: Review catches >70% of errors
- **Visualization**: All chart types generate without errors
- **Reports**: Complete reports with all sections

### Performance Requirements:
- Routing decision: <100ms
- CoT parsing: <50ms
- Plugin selection: <200ms
- Chart generation: <3s
- Report generation: <5s

---

## 🎯 TESTING APPROACH

1. **Unit Tests**: Individual components (parsers, routers, agents)
2. **Integration Tests**: Combined workflows (CoT + routing, plugin + analysis)
3. **End-to-End Tests**: Full user scenarios (upload → analyze → visualize → report)
4. **Accuracy Tests**: Verify correct results (not just execution)
5. **Performance Tests**: Measure timing and resource usage

---

## 📝 NEXT STEPS

1. Execute Phase 1: CoT System Testing
2. Execute Phase 2: Intelligent Routing Testing
3. Execute Phase 3: Plugin Agent Testing
4. Execute Phase 4: Primary+Review Analysis Testing
5. Execute Phase 5: Visualization Testing
6. Execute Phase 6: Report Generation Testing
7. Generate comprehensive results report
8. Identify improvements needed

---

**Status**: READY TO EXECUTE
**Backend**: Running on port 8000
**Models Available**: phi3:mini, tinyllama, llama3.1:8b
**Data Files**: CSV, JSON, PDF, TXT samples ready
