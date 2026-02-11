# Specialist Agent Code Generation - Quick Reference

## Overview

**StatisticalAgent** and **FinancialAgent** now support **LLM-based code generation** for complex, unstructured, or novel queries. This enables them to handle sophisticated analyses beyond their deterministic methods.

## Architecture

### Hybrid Execution Model

```
Query → QueryOrchestrator → Decision:
                               ├─ CODE_GENERATION → _execute_with_code_gen()
                               └─ DIRECT_LLM → Standard deterministic methods
```

### Decision Logic

The `QueryOrchestrator` analyzes queries and decides:
- **Code Generation**: Complex multi-step analyses, novel statistical methods, custom calculations
- **Standard Methods**: Simple queries matching keyword patterns (t-test, correlation, profit margin, etc.)

## Implementation Details

### 1. Imports Added

Both agents now import:
```python
from backend.core.engine.query_orchestrator import QueryOrchestrator, ExecutionMethod
from backend.agents.model_manager import get_model_manager
```

### 2. Initialize Method

```python
def initialize(self, **kwargs) -> bool:
    # ... existing config ...
    
    # New: Setup for code generation
    self._orchestrator = None  # Lazy loaded
    self.initializer = get_model_manager()
```

### 3. Execute Method (Refactored)

```python
def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
    # 1. Load data (existing logic preserved)
    # ...
    
    # 2. Use QueryOrchestrator to decide execution method
    if self._orchestrator is None:
        self._orchestrator = QueryOrchestrator()
    
    plan = self._orchestrator.create_execution_plan(
        query=query,
        data=data_sample,
        context={'agent': self.get_metadata().name, 'columns': available_columns},
        llm_client=self.initializer.llm_client
    )
    
    # 3. Route to code generation if needed
    if plan.execution_method == ExecutionMethod.CODE_GENERATION:
        result, metadata = self._execute_with_code_gen(query, data, plan.model, filepath)
        return {"success": True, "result": result, "metadata": metadata}
    
    # 4. Fallback to existing deterministic logic
    intent = self._parse_intent(query)
    # ... existing dispatch to specialized methods ...
```

### 4. New Method: _execute_with_code_gen

Ported from `DataAnalystAgent`:

```python
def _execute_with_code_gen(self, query: str, data: Any, model: str, filepath: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Execute analysis using LLM code generation.
    
    Args:
        query: User's analysis query
        data: DataFrame or data object
        model: LLM model to use
        filepath: Path to data file (optional)
        
    Returns:
        Tuple of (result_text, metadata_dict)
    """
    from backend.io.code_generator import get_code_generator
    
    # Ensure DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(filepath)  # Load based on file type
    
    # Generate and execute code
    code_gen = get_code_generator()
    result = code_gen.generate_and_execute(
        query=query,
        df=data,
        model=model,
        max_retries=2,
        save_history=True,
        analysis_context={'agent': 'StatisticalAgent', 'specialization': 'statistical_analysis'}
    )
    
    # Format and return result
    if result.success:
        display_text = format_result(result.result)
        metadata = {
            "agent": "StatisticalAgent",
            "execution_method": "code_generation",
            "generated_code": result.generated_code,
            "execution_time_ms": result.execution_time_ms
        }
        return display_text, metadata
    else:
        raise Exception(f"Code generation failed: {result.error}")
```

## Example Queries

### StatisticalAgent

**Complex (Code Generation):**
- "Perform multivariate outlier detection using Mahalanobis distance"
- "Calculate rolling Z-scores with 30-day windows and detect anomalies"
- "Run principal component analysis and show explained variance ratios"

**Simple (Deterministic):**
- "Run a t-test comparing group A and group B"
- "Calculate correlation between revenue and marketing spend"
- "Show descriptive statistics for all columns"

### FinancialAgent

**Complex (Code Generation):**
- "Calculate customer lifetime value with cohort retention curves"
- "Perform seasonal decomposition of revenue with trend analysis"
- "Generate cash flow waterfall with cumulative net change"

**Simple (Deterministic):**
- "Calculate profitability metrics"
- "Show ROI analysis"
- "Calculate growth rates year over year"

## Benefits

### 1. **Flexibility**
- Handle novel queries not covered by deterministic methods
- Adapt to custom business logic
- Support domain-specific calculations

### 2. **Robustness**
- Preserve existing proven methods for standard queries
- Fallback mechanism ensures graceful degradation
- No breaking changes to current functionality

### 3. **Transparency**
- Generated code included in metadata
- Users can inspect and verify calculations
- Execution logs track decision reasoning

## Testing

### Run Test Suite
```bash
python test_specialist_code_gen.py
```

### Test Components
1. **Orchestrator Routing**: Verifies correct decision-making
2. **StatisticalAgent Standard**: Tests deterministic methods still work
3. **StatisticalAgent Code Gen**: Tests complex statistical queries
4. **FinancialAgent Standard**: Tests deterministic financial methods
5. **FinancialAgent Code Gen**: Tests complex financial analyses

## API Response Format

### Standard Method Response
```json
{
  "success": true,
  "result": "Statistical analysis result...",
  "agent": "StatisticalAgent",
  "test_type": "t-test",
  "p_value": 0.023
}
```

### Code Generation Response
```json
{
  "success": true,
  "result": "## Statistical Result\n\n**Analysis output...**",
  "metadata": {
    "agent": "StatisticalAgent",
    "execution_method": "code_generation",
    "generated_code": "import pandas as pd...",
    "executed_code": "# Cleaned code...",
    "execution_time_ms": 1234.5,
    "code_gen_model": "qwen2.5-coder:7b"
  }
}
```

## Configuration

### Enable/Disable Code Generation

In `config/cot_review_config.json`:

```json
{
  "query_orchestrator": {
    "enable_code_generation": true,
    "complexity_threshold": 0.6,
    "semantic_routing": true
  }
}
```

### Model Selection

The orchestrator automatically selects the best model based on:
- Query complexity
- Available models
- Model capabilities (code generation vs. general LLM)

## Fallback Behavior

### Failure Cascade

```
1. Code Generation Attempted
   ↓ (failure)
2. Exception caught in _execute_with_code_gen
   ↓ (re-raise)
3. Falls through to standard deterministic logic
   ↓
4. Intent parsing & specialized methods execute
```

### Graceful Degradation

If code generation fails:
- Error logged with details
- Execution continues with deterministic methods
- User receives valid result (not an error)

## Performance Considerations

### Code Generation
- **Latency**: 2-10 seconds (LLM generation + execution)
- **Use When**: Complex, multi-step, or novel analyses
- **Cost**: Higher (LLM inference + code execution)

### Deterministic Methods
- **Latency**: <100ms (direct calculation)
- **Use When**: Standard statistical tests, common metrics
- **Cost**: Minimal (no LLM calls)

## Troubleshooting

### Code Generation Not Triggering

**Check:**
1. Query complexity sufficient (>0.6 complexity score)
2. LLM client initialized (`model_manager.ensure_initialized()`)
3. Code generator available (`get_code_generator()` works)
4. Semantic routing enabled in config

### Code Generation Failing

**Check:**
1. Model supports code generation (e.g., `qwen2.5-coder`)
2. Data loaded correctly (DataFrame available)
3. File path valid if provided
4. No syntax errors in generated code (check logs)

### Always Using Standard Methods

**Possible causes:**
1. Queries too simple (orchestrator chooses DIRECT_LLM)
2. Semantic routing disabled
3. Orchestrator not initialized (`self._orchestrator = None`)

## Logs

### Successful Code Generation
```
INFO: QueryOrchestrator decision for StatisticalAgent: CODE_GENERATION - Complex multivariate analysis detected
INFO: StatisticalAgent code generation succeeded: 2341.5ms
```

### Fallback to Standard
```
INFO: QueryOrchestrator decision for StatisticalAgent: DIRECT_LLM - Simple correlation query
DEBUG: Using deterministic correlation_analysis method
```

## Future Enhancements

### Planned
- Cache generated code for similar queries
- User preference: force code generation or standard methods
- Multi-agent code generation (collaborate between agents)
- Code validation with unit tests before execution

### Under Consideration
- Fine-tune routing prompts per agent specialization
- Add code generation for TimeSeriesAgent and MLInsightsAgent
- Support streaming code generation progress
- Interactive code refinement based on user feedback

## Related Documentation

- [SEMANTIC_ROUTING_QUICKREF.md](SEMANTIC_ROUTING_QUICKREF.md) - Query orchestration
- [AGENT_STREAMING_QUICKREF.md](AGENT_STREAMING_QUICKREF.md) - Agent execution in streaming
- [COMPLETE_PROJECT_EXPLANATION.md](../docs/COMPLETE_PROJECT_EXPLANATION.md) - Full system architecture

## Summary

✅ **StatisticalAgent** and **FinancialAgent** now intelligently switch between:
- **Code Generation**: For complex, custom, or novel analyses
- **Deterministic Methods**: For standard, proven statistical/financial operations

This hybrid approach combines **flexibility** with **robustness**, ensuring both sophisticated capabilities and reliable performance.
