# Project Data-Agnostic Compliance Report

**Auditor Role:** Senior Architectural Compliance Auditor
**Date:** 2025-12-30
**Target:** Nexus LLM Analytics Codebase

---

## 1. Compliance Verdict
### 🔴 PARTIALLY COMPLIANT (High Risk in Domain Plugins)

While the **Core Infrastructure** (Ingestion, Routing, Sandboxing) is designed with commendable generic principles, the **Domain-Specific Agents** (specifically `FinancialAgent`) and **Optimization Utilities** contain severe violations of data-agnostic principles. They rely on hardcoded English vocabulary lists, arbitrary numeric thresholds, and "Magic String" heuristics that will fail on non-standard, non-English, or cross-domain datasets.

---

## 2. Evidence Summary

### ✅ Compliant Areas (The "Platform")
*   **Data Ingestion (`data_utils.py`)**: Uses robust, library-standard inference (`pd.read_csv`, `infer_data_types`). Does not presume column existence. The `get_column_properties` function generates a generic metadata schema (types, min/max, unique counts) that allows downstream agents to reason about data *abstractly*.
*   **Code Sandboxing**: The `SecurityGuards` and `RestrictedPython` implementation is purely syntactic (AST-based) and completely data-agnostic.
*   **Plugin Architecture**: The `AgentRegistry` and discovery mechanism are agnostic to the content of the agents.

### ❌ Non-Compliant Areas (The "Intelligence")
*   **Rigid Semantic Mapping**: The system attempts to "force" data understanding by checking against hardcoded lists of words like "Revenue", "Sales", "Profit".
*   **Arbitrary Thresholds**: Business logic defines "High Growth" as >20% regardless of whether the user is analyzing a stable utility company or a volatile crypto asset.
*   **Hidden Heuristics**: Optimization logic skips or highlights columns based on string matching against a fixed list of common English column names.

---

## 3. Violations Found

| Severity | Component | File Location | Violation Details |
| :--- | :--- | :--- | :--- |
| 🚨 **CRITICAL** | **Financial Agent** | `src/backend/plugins/financial_agent.py` (L163-171) | **Hardcoded Vocabulary**: `revenue`, `sales`, `income` mapped explicitly. Will fail on `GrossInflow` or `Revenue_EUR`. |
| 🚨 **CRITICAL** | **Financial Agent** | `src/backend/plugins/financial_agent.py` (L397, L503) | **Arbitrary Thresholds**: Hardcoded benchmarks (40% margin, 20% growth) baked into code. This is **Overfitting**. |
| ⚠️ **HIGH** | **Data Optimizer** | `src/backend/utils/data_optimizer.py` (L710) | **Magic Lists**: `['revenue', 'profit', 'margin'...]` used to identify "interesting" columns. Biases analysis towards financial datasets. |
| ⚠️ **HIGH** | **Validation** | `src/backend/core/automated_validation.py` (L329) | **Heuristic Filters**: Hardcoded ignored columns `['id', 'name', 'age']`. Will accidentally ignore a column named "Age" in a geological dating dataset. |
| 🟡 **MED** | **Financial Agent** | `src/backend/plugins/financial_agent.py` (L577) | **Fixed Stability Logic**: Assumes Coefficient of Variation < 0.2 is "stable". This statistical assumption is not universally true across domains. |

---

## 4. Risk Assessment

**Scenario**: User uploads a **biotech clinical trial dataset**.
*   **Data**: Columns like `Subject_ID`, `Dosage_mg`, `Titer_Level`, `Growth_Rate`.
*   **Failure**:
    1.  `FinancialAgent` might trigger on "Growth_Rate" but apply financial "20% = High" logic, which might be microscopically small or massive for cell growth, rendering insights misleading.
    2.  `DataOptimizer` might ignore `Subject_ID` if it matches heuristics similar to "ID", potentially losing key segmentation data.
    3.  If strict mode is on, the dataset might be rejected for not matching the "Revenue/Cost" schema expected by the overfitted logic.

**Impact**: The system works beautifully for the "Happy Path" (Sales Data) but provides **dangerous misinformation** or **silent failures** for Unseen Domains.

---

## 5. Required Fix Suggestions

### Fix A: Semantic Mapping Layer (Architecture Level)
*   **Current**: Hardcoded dict `{'revenue': ['sales', 'income']}`.
*   **Required**: Introduce a dynamic `SemanticMapper` step.
    *   Use the LLM to map *User Columns* -> *Abstract Concepts* at runtime.
    *   Example: LLM identifies `GrossInflow` maps to concept `Inflow_Metric`. Code then operates on `Inflow_Metric`.

### Fix B: Relative Thresholding (Logic Level)
*   **Current**: `if growth > 20%: return "High"`
*   **Required**: Calculate Distributional Statistics (Percentiles/Z-Scores).
    *   logic: `if growth > dataset.growth.percentile(75): return "High relative to this dataset"`.
    *   Alternatively: Prompt the user for domain constants during `AnalysisRequest`.

### Fix C: Configurable Heuristics (Robustness Level)
*   **Current**: `ignored_cols = ['id', 'name']` inside python code.
*   **Required**: Move all "magic lists" to an external `config/heuristics.yaml` or `system_prompt`.
    *   Allow the user to override: "Treat 'Date' as categorical".

---

## 6. Idea & Methodology Status

| Methodology | Status | Completeness | Notes |
| :--- | :--- | :--- | :--- |
| **Plugin System** | ✅ **Accurate** | 100% | Properly decouples logic. |
| **Generative Router** | ✅ **Accurate** | 100% | Text-based routing is naturally generic. |
| **Abstract Profiling**| ✅ **Accurate** | 90% | `data_utils` is strong, but needs to shed the few magic strings. |
| **Semantic Layer** | ❌ **Missing** | 0% | The missing link between generic code and specific data. |

---

## 7. Final Confidence Score

**Score: 6/10**

**Justification**:
The **Platform (Backend/Core)** is a solid 9/10. It handles generic files, validates AST, and routes intelligently.
However, the **Application Logic (Agents)** is a 3/10. It is currently built effectively as a "Financial Demo" rather than a true "Universal Analyst". It relies too heavily on the developer's assumptions about what data looks like.

To reach 10/10, the "Business Logic" must be extracted from the Python code and moved into **Prompts** or **Configuration**, turning the agents into pure reasoning engines rather than hardcoded calculators.
