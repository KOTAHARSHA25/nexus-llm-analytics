# 🎯 UNIFIED ARCHITECTURE - Three Tracks, One System

**Date:** December 27, 2025  
**Status:** Design Document  
**Purpose:** Integrate all three innovation tracks into one coherent system

---

## The Three Tracks

Your project has **three parallel innovations** that need to work together:

### Track 1: Complexity-Based Model Selection
**Idea:** Simple queries use lightweight models (low RAM), complex queries use powerful models

**Current State:** 
- ✅ Three models available (llama3.1:8b, phi3:mini, tinyllama)
- ✅ Query complexity analyzer exists
- ❌ No logic connecting complexity score → model choice

### Track 2: Code Generation vs Direct LLM
**Idea:** Computational queries generate Python code (accurate), conversational queries use direct LLM

**Current State:**
- ✅ Code generation pipeline designed (Phase 2 of roadmap)
- ❌ Not implemented yet
- ❌ No decision logic for when to use code gen

### Track 3: Two Friends Model (Collaborative Intelligence)
**Idea:** Generator LLM + Critic LLM = Higher quality outputs

**Current State:**
- ✅ Infrastructure exists (CoT parser, self-correction engine)
- ❌ BROKEN - Critic approves everything (0% improvement rate)
- ❌ No rules for when to activate (every query? only complex ones?)

---

## The Unified Decision Flow

Here's how all three tracks integrate:

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY ORCHESTRATOR                         │
│                  (NEW - Master Decision Maker)                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  1. ANALYZE COMPLEXITY │
                    │  (Query Complexity     │
                    │   Analyzer)            │
                    └────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Complexity Score       │
                    │  0.0 - 1.0              │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
   [0.0 - 0.3]             [0.3 - 0.7]              [0.7 - 1.0]
    SIMPLE                   MEDIUM                  COMPLEX
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ 2a. SELECT    │      │ 2b. SELECT    │      │ 2c. SELECT    │
│     MODEL     │      │     MODEL     │      │     MODEL     │
├───────────────┤      ├───────────────┤      ├───────────────┤
│ tinyllama     │      │ phi3:mini     │      │ llama3.1:8b   │
│ (637 MB)      │      │ (2.2 GB)      │      │ (4.9 GB)      │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ 3a. EXECUTION │      │ 3b. EXECUTION │      │ 3c. EXECUTION │
│     METHOD    │      │     METHOD    │      │     METHOD    │
├───────────────┤      ├───────────────┤      ├───────────────┤
│ Direct LLM    │      │ Code Gen OR   │      │ Code Gen      │
│               │      │ Direct LLM    │      │ (Mandatory)   │
│               │      │ (Auto-decide) │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ 4a. REVIEW    │      │ 4b. REVIEW    │      │ 4c. REVIEW    │
│     DECISION  │      │     DECISION  │      │     DECISION  │
├───────────────┤      ├───────────────┤      ├───────────────┤
│ Skip Review   │      │ Two Friends   │      │ Two Friends   │
│ (Fast path)   │      │ (If enabled)  │      │ (Mandatory)   │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  FINAL RESULT  │
                        └────────────────┘
```

---

## Decision Rules (The Integration Logic)

### Rule Set 1: Model Selection (Track 1)

| Complexity Score | Model         | RAM Usage | Use Case                    |
|------------------|---------------|-----------|----------------------------|
| 0.0 - 0.3       | tinyllama     | 637 MB    | Simple lookups, definitions |
| 0.3 - 0.7       | phi3:mini     | 2.2 GB    | Medium analysis, filtering  |
| 0.7 - 1.0       | llama3.1:8b   | 4.9 GB    | Complex reasoning, math     |

**Implementation:** Enhance `model_selector.py` to use complexity score

### Rule Set 2: Execution Method (Track 2)

| Query Type               | Complexity | Method      | Reason                        |
|--------------------------|------------|-------------|-------------------------------|
| Calculation queries      | Any        | Code Gen    | Python math > LLM approximation |
| Aggregation (sum, avg)   | Any        | Code Gen    | Exact results required        |
| Filtering/sorting        | Medium+    | Code Gen    | Complex logic                 |
| Time series analysis     | High       | Code Gen    | Multi-step computation        |
| Conversational           | Low        | Direct LLM  | No computation needed         |
| Explanations             | Any        | Direct LLM  | Natural language task         |

**Detection Keywords:**
- **Code Gen:** calculate, compute, sum, average, total, count, filter, group by, correlation, percentage
- **Direct LLM:** explain, what is, why, how, describe, summarize

### Rule Set 3: Two Friends Model Activation (Track 3)

| Scenario                  | Apply Two Friends? | Reason                           |
|---------------------------|--------------------|----------------------------------|
| Simple query (< 0.3)      | ❌ No              | Overhead not justified           |
| Medium query (0.3 - 0.7)  | ⚠️ Optional        | Enable if accuracy critical      |
| Complex query (> 0.7)     | ✅ Yes             | High stakes, worth the overhead  |
| Code generation used      | ✅ Yes             | Validate generated code          |
| First attempt failed      | ✅ Yes             | Error recovery                   |

**Configuration:** Add to `cot_review_config.json`:
```json
{
  "enabled": true,
  "activation_rules": {
    "always_on_complexity": 0.7,
    "optional_range": [0.3, 0.7],
    "always_on_code_gen": true,
    "on_error_recovery": true
  }
}
```

---

## Implementation Plan

### Phase 0.7: Fix Two Friends Model (NEW - URGENT)
**Effort:** 1 day  
**Priority:** CRITICAL

**Step 1:** Fix Critic Prompt
```python
# File: src/backend/prompts/cot_critic_prompt.txt

# BEFORE (too lenient):
# "Be helpful but fair. Only report real problems, not small style differences."

# AFTER (strict):
"""
You are a STRICT quality reviewer. Your job is to find errors and improvements.

REVIEW CHECKLIST:
1. Logic Errors: Is the reasoning sound?
2. Calculation Errors: Are numbers computed correctly?
3. Missing Steps: Are there gaps in reasoning?
4. Incorrect Assumptions: Are assumptions valid?
5. Completeness: Does it fully answer the question?

CRITICAL: If you find ANY of these issues, you MUST return [INVALID] with specific feedback.
Only return [VALID] if the reasoning is FLAWLESS.

Do not be lenient. Catch mistakes.
"""
```

**Step 2:** Test with Intentional Errors
```python
# Test queries that SHOULD be caught by critic
test_cases = [
    {
        "query": "What is 2 + 2?",
        "bad_reasoning": "[REASONING]2+2=5[/REASONING][OUTPUT]5[/OUTPUT]",
        "should_catch": True
    },
    {
        "query": "Filter customers where age > 30",
        "bad_reasoning": "[REASONING]Filter age < 30[/REASONING][OUTPUT]Wrong filter[/OUTPUT]",
        "should_catch": True
    }
]
```

### Phase 0.8: Create Query Orchestrator (NEW)
**Effort:** 3 days  
**Priority:** HIGH

**New File:** `src/backend/core/query_orchestrator.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class ExecutionMethod(Enum):
    DIRECT_LLM = "direct_llm"
    CODE_GENERATION = "code_generation"

class ReviewLevel(Enum):
    NONE = "none"
    OPTIONAL = "optional"
    MANDATORY = "mandatory"

@dataclass
class ExecutionPlan:
    """Unified execution plan combining all three tracks"""
    model: str                          # Track 1: Which model to use
    execution_method: ExecutionMethod   # Track 2: Code gen or direct
    review_level: ReviewLevel           # Track 3: Two Friends activation
    complexity_score: float
    reasoning: str

class QueryOrchestrator:
    """Master decision maker - integrates all three innovation tracks"""
    
    def __init__(self, complexity_analyzer, model_selector, config):
        self.complexity_analyzer = complexity_analyzer
        self.model_selector = model_selector
        self.config = config
    
    def create_execution_plan(self, query: str, data: Any = None) -> ExecutionPlan:
        """
        Unified decision logic:
        1. Analyze complexity → determines model AND review level
        2. Detect query type → determines execution method
        3. Combine into coherent plan
        """
        
        # TRACK 1: Complexity → Model Selection
        complexity = self.complexity_analyzer.analyze(query, data)
        model = self._select_model(complexity)
        
        # TRACK 2: Query Type → Execution Method
        execution_method = self._select_execution_method(query, complexity, data)
        
        # TRACK 3: Complexity + Method → Review Level
        review_level = self._select_review_level(complexity, execution_method)
        
        return ExecutionPlan(
            model=model,
            execution_method=execution_method,
            review_level=review_level,
            complexity_score=complexity,
            reasoning=self._explain_plan(model, execution_method, review_level, complexity)
        )
    
    def _select_model(self, complexity: float) -> str:
        """Track 1: Complexity → Model"""
        if complexity < 0.3:
            return "tinyllama"
        elif complexity < 0.7:
            return "phi3:mini"
        else:
            return "llama3.1:8b"
    
    def _select_execution_method(self, query: str, complexity: float, data: Any) -> ExecutionMethod:
        """Track 2: Query Type → Execution Method"""
        code_gen_keywords = [
            'calculate', 'compute', 'sum', 'average', 'mean', 'total',
            'count', 'group by', 'aggregate', 'filter', 'sort', 'rank',
            'correlation', 'percentage', 'ratio'
        ]
        
        query_lower = query.lower()
        needs_computation = any(kw in query_lower for kw in code_gen_keywords)
        has_data = data is not None
        
        # Code generation for computational tasks with data
        if needs_computation and has_data and complexity >= 0.3:
            return ExecutionMethod.CODE_GENERATION
        
        return ExecutionMethod.DIRECT_LLM
    
    def _select_review_level(self, complexity: float, method: ExecutionMethod) -> ReviewLevel:
        """Track 3: When to apply Two Friends Model"""
        config = self.config.get('cot_review', {})
        
        # Mandatory for complex queries
        if complexity >= config.get('always_on_complexity', 0.7):
            return ReviewLevel.MANDATORY
        
        # Mandatory for code generation (validate generated code)
        if method == ExecutionMethod.CODE_GENERATION and config.get('always_on_code_gen', True):
            return ReviewLevel.MANDATORY
        
        # Optional for medium complexity
        if complexity >= 0.3:
            return ReviewLevel.OPTIONAL
        
        # Skip for simple queries (fast path)
        return ReviewLevel.NONE
    
    def _explain_plan(self, model: str, method: ExecutionMethod, review: ReviewLevel, complexity: float) -> str:
        """Human-readable explanation of the plan"""
        return f"""
Execution Plan:
- Complexity: {complexity:.2f}
- Model: {model} (RAM-efficient for this complexity)
- Method: {method.value} ({'accurate computation' if method == ExecutionMethod.CODE_GENERATION else 'natural language'})
- Review: {review.value} ({'will validate with critic' if review == ReviewLevel.MANDATORY else 'fast path'})
        """.strip()
```

### Phase 0.9: Integration with Existing System
**Effort:** 2 days

**Update `src/backend/plugins/data_analyst_agent.py`:**

```python
from ..core.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel

class DataAnalystAgent:
    def __init__(self, ...):
        # ... existing init ...
        self.orchestrator = QueryOrchestrator(
            self.complexity_analyzer,
            self.model_selector,
            self.config
        )
    
    async def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Unified execution using QueryOrchestrator"""
        
        # 1. CREATE EXECUTION PLAN (integrates all three tracks)
        plan = self.orchestrator.create_execution_plan(query, data)
        
        logger.info(f"Execution Plan: {plan.reasoning}")
        
        # 2. EXECUTE BASED ON PLAN
        if plan.execution_method == ExecutionMethod.CODE_GENERATION:
            result = await self._execute_with_code_generation(
                query, data, model=plan.model, review=plan.review_level
            )
        else:
            result = await self._execute_direct(
                query, data, model=plan.model, review=plan.review_level
            )
        
        result['execution_plan'] = {
            'model': plan.model,
            'method': plan.execution_method.value,
            'review': plan.review_level.value,
            'complexity': plan.complexity_score
        }
        
        return result
```

---

## Updated Roadmap Integration

Add these to **Phase 0** (before Task 1.1):

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| **0.7** | CRITICAL | 1 day | Fix Two Friends Model critic prompt |
| **0.8** | HIGH | 3 days | Create QueryOrchestrator (unify 3 tracks) |
| **0.9** | HIGH | 2 days | Integrate orchestrator with agents |

These must be done BEFORE Phase 1, as they are foundational.

---

## Success Metrics

After implementation, the system should:

✅ **Track 1 (Model Selection):**
- Simple queries use tinyllama (<1 GB RAM)
- Complex queries use llama3.1 (requires 5+ GB RAM)
- Medium queries use phi3 (2-3 GB RAM)

✅ **Track 2 (Execution Method):**
- Calculation queries generate Python code (100% accuracy)
- Conversational queries use direct LLM (natural language)

✅ **Track 3 (Two Friends Model):**
- Critic catches ≥80% of intentional errors
- Complex queries show ≥15% improvement with review
- Simple queries skip review (faster response)

✅ **Unified System:**
- All three tracks work together seamlessly
- Decision logic is transparent (explain plan to user)
- Performance improves across all query types

---

## Configuration File

**Update `config/cot_review_config.json`:**

```json
{
  "enabled": true,
  "max_iterations": 2,
  "complexity_threshold": 0.5,
  "activation_rules": {
    "always_on_complexity": 0.7,
    "optional_range": [0.3, 0.7],
    "always_on_code_gen": true,
    "on_error_recovery": true
  },
  "model_selection": {
    "simple": "tinyllama",
    "medium": "phi3:mini",
    "complex": "llama3.1:8b",
    "thresholds": {
      "simple_max": 0.3,
      "medium_max": 0.7
    }
  }
}
```

---

**Next Steps:** Fix critic prompt first, then implement QueryOrchestrator.
