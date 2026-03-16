# -*- coding: utf-8 -*-
"""
=============================================================================
  NEXUS LLM ANALYTICS - COMPREHENSIVE ACCURACY TEST (BATCH MODE)
  Tests CSV, JSON, TXT, PDF, DOCX from Simple to God level
  Runs one test at a time with memory checks and saves progress after each.
=============================================================================
"""
import sys, os, io, json, time, asyncio, traceback, gc, psutil
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

RESULTS_FILE = ROOT / "accuracy_test_results.json"
REPORT_FILE = ROOT / "ACCURACY_TEST_REPORT.md"

# ============================================================================
#  TEST SUITE DEFINITION
# ============================================================================
DIFFICULTY_LEVELS = ["Simple", "Intermediate", "Advanced", "Expert", "God"]

def get_test_suite():
    """Return the full test suite as a flat list of test dicts."""
    tests = []
    tid = 0

    # ----- CSV TESTS -----
    csv_tests = [
        # sales_data.csv (100 rows: product,region,sales,revenue,price,marketing_spend)
        ("CSV", "data/samples/sales_data.csv", "Simple", "How many rows are in this dataset?", ["100", "row"]),
        ("CSV", "data/samples/sales_data.csv", "Simple", "What are the column names in this file?", ["product", "region", "sales", "revenue"]),
        ("CSV", "data/samples/sales_data.csv", "Intermediate", "What is the total revenue across all products?", ["revenue", "total"]),
        ("CSV", "data/samples/sales_data.csv", "Intermediate", "Which region has the highest total sales?", ["region"]),
        ("CSV", "data/samples/sales_data.csv", "Advanced", "Calculate the average revenue per unit sold for each product", ["average", "product"]),
        ("CSV", "data/samples/sales_data.csv", "Advanced", "What is the correlation between marketing_spend and revenue?", ["correlation", "marketing"]),
        ("CSV", "data/samples/sales_data.csv", "Expert", "Which product has the best revenue-to-marketing-spend ratio? Provide a profitability ranking.", ["ratio", "product"]),
        ("CSV", "data/samples/sales_data.csv", "God", "Build a regression model to predict revenue from sales, price and marketing_spend. Report R-squared and the most significant predictor.", ["predict"]),

        # healthcare_patients.csv
        ("CSV", "data/samples/healthcare_patients.csv", "Simple", "How many patients are in the dataset?", ["patient"]),
        ("CSV", "data/samples/healthcare_patients.csv", "Intermediate", "What is the average treatment cost?", ["average", "cost"]),
        ("CSV", "data/samples/healthcare_patients.csv", "Advanced", "Compare average treatment costs between patients with and without insurance coverage", ["insurance", "cost"]),
        ("CSV", "data/samples/healthcare_patients.csv", "Expert", "Which comorbidities are associated with the highest readmission rates?", ["readmission"]),
        ("CSV", "data/samples/healthcare_patients.csv", "God", "Predict severity level using age, BMI, blood pressure, cholesterol and glucose. Which feature is most predictive?", ["predict", "severity"]),

        # hr_employee_data.csv
        ("CSV", "data/samples/hr_employee_data.csv", "Simple", "List all unique departments in the dataset", ["department"]),
        ("CSV", "data/samples/hr_employee_data.csv", "Intermediate", "What is the average salary by department?", ["salary", "average"]),
        ("CSV", "data/samples/hr_employee_data.csv", "Advanced", "Which employees have performance rating above 4.0 and are bonus eligible?", ["bonus", "performance"]),
        ("CSV", "data/samples/hr_employee_data.csv", "Expert", "Analyze the relationship between training hours and performance rating across departments", ["training", "performance"]),
        ("CSV", "data/samples/hr_employee_data.csv", "God", "Build a predictive model for bonus eligibility using salary, performance_rating, training_hours and projects_completed. Report accuracy.", ["predict", "accuracy"]),

        # time_series_stock.csv
        ("CSV", "data/samples/time_series_stock.csv", "Simple", "What stock symbols are in the dataset?", ["symbol"]),
        ("CSV", "data/samples/time_series_stock.csv", "Intermediate", "What is the average closing price for each stock symbol?", ["average", "close"]),
        ("CSV", "data/samples/time_series_stock.csv", "Advanced", "Calculate the 5-day moving average of closing prices for the TECH sector", ["moving", "average"]),
        ("CSV", "data/samples/time_series_stock.csv", "Expert", "Identify the most volatile stock based on daily returns", ["volatile", "return"]),
        ("CSV", "data/samples/time_series_stock.csv", "God", "Calculate a Sharpe-like ratio (mean daily return / std of daily return) for each sector. Which is best risk-adjusted?", ["ratio", "sector"]),

        # comprehensive_ecommerce.csv
        ("CSV", "data/samples/comprehensive_ecommerce.csv", "Simple", "How many orders are in the dataset?", ["order"]),
        ("CSV", "data/samples/comprehensive_ecommerce.csv", "Intermediate", "What is the most popular product category by order count?", ["category"]),
        ("CSV", "data/samples/comprehensive_ecommerce.csv", "Advanced", "What is the return rate by product category?", ["return", "category"]),
        ("CSV", "data/samples/comprehensive_ecommerce.csv", "Expert", "Segment customers by Premium vs Regular and compare average order values", ["segment", "premium"]),
        ("CSV", "data/samples/comprehensive_ecommerce.csv", "God", "Perform a customer lifetime value analysis: avg order value and total revenue by customer_segment and city", ["customer", "value", "segment"]),

        # university_academic_data.csv
        ("CSV", "data/samples/university_academic_data.csv", "Simple", "How many students are in the dataset?", ["student"]),
        ("CSV", "data/samples/university_academic_data.csv", "Intermediate", "What is the average GPA by major?", ["gpa", "average"]),
        ("CSV", "data/samples/university_academic_data.csv", "Advanced", "Is there a correlation between study hours per week and final score?", ["correlation", "study"]),
        ("CSV", "data/samples/university_academic_data.csv", "Expert", "Compare GPA between students with and without part-time jobs", ["part-time", "gpa"]),
        ("CSV", "data/samples/university_academic_data.csv", "God", "Predict final_score using GPA, attendance_rate, study_hours, midterm_score, participation_score. Which factor matters most?", ["predict", "factor"]),
    ]

    # ----- JSON TESTS -----
    json_tests = [
        ("JSON", "data/samples/simple.json", "Simple", "How many products are in the sales_data?", ["5", "product"]),
        ("JSON", "data/samples/simple.json", "Intermediate", "What is the total sales amount across all products?", ["total", "amount"]),
        ("JSON", "data/samples/simple.json", "Advanced", "Which product has the highest amount and by what percent does it exceed the average?", ["product", "average"]),

        ("JSON", "data/samples/financial_quarterly.json", "Simple", "How many quarters of data are available?", ["quarter"]),
        ("JSON", "data/samples/financial_quarterly.json", "Intermediate", "What is the total annual revenue?", ["revenue", "total"]),
        ("JSON", "data/samples/financial_quarterly.json", "Advanced", "Which quarter had the highest gross margin percentage?", ["margin", "quarter"]),
        ("JSON", "data/samples/financial_quarterly.json", "Expert", "Calculate quarter-over-quarter revenue growth rate", ["growth", "quarter"]),
        ("JSON", "data/samples/financial_quarterly.json", "God", "Analyze the full P&L: track revenue through COGS, expenses to net income. Which expense category grew fastest?", ["expense", "revenue"]),

        ("JSON", "data/samples/complex_nested.json", "Simple", "What is the company name?", ["techcorp"]),
        ("JSON", "data/samples/complex_nested.json", "Intermediate", "How many departments does the company have?", ["department"]),
        ("JSON", "data/samples/complex_nested.json", "Advanced", "What is the total salary expenditure across all departments?", ["salary", "total"]),
        ("JSON", "data/samples/complex_nested.json", "Expert", "Which department has the highest average salary?", ["salary", "department"]),
        ("JSON", "data/samples/complex_nested.json", "God", "Calculate headcount, avg salary, budget utilization per department. Flag departments where salaries exceed 50 percent of budget.", ["budget", "salary", "department"]),

        ("JSON", "data/samples/nested_manufacturing.json", "Simple", "How many manufacturing plants are listed?", ["plant"]),
        ("JSON", "data/samples/nested_manufacturing.json", "Intermediate", "What is the average efficiency rate across all plants?", ["efficiency"]),
        ("JSON", "data/samples/nested_manufacturing.json", "Advanced", "Which plant has the highest defect rate?", ["defect", "plant"]),
        ("JSON", "data/samples/nested_manufacturing.json", "Expert", "Which plant is most energy-efficient (lowest energy per unit of output)?", ["energy", "efficient"]),
        ("JSON", "data/samples/nested_manufacturing.json", "God", "Rank plants by composite score (efficiency, inverse defect rate, first pass yield, energy per unit). Report rankings.", ["rank", "plant", "score"]),
    ]

    # ----- TXT TESTS -----
    txt_tests = [
        ("TXT", "data/evaluation/market_report.txt", "Simple", "What type of document is this?", ["market", "report"]),
        ("TXT", "data/evaluation/market_report.txt", "Intermediate", "What was the growth rate of the technology sector?", ["15", "growth"]),
        ("TXT", "data/evaluation/market_report.txt", "Advanced", "Summarize the key findings: which segments grew and which declined?", ["technology", "growth"]),
        ("TXT", "data/evaluation/market_report.txt", "Expert", "What strategic recommendations would you make based on this report?", ["cloud", "recommendation"]),
        ("TXT", "data/evaluation/market_report.txt", "God", "Extract all quantitative metrics, categorize by segment, and assess overall market health.", ["metric", "segment"]),
    ]

    # ----- PDF TESTS (extracted text) -----
    pdf_tests = [
        ("PDF", "data/uploads/HARSHA_Kota_Resume.pdf.extracted.txt", "Simple", "Whose resume is this?", ["harsha", "kota"]),
        ("PDF", "data/uploads/HARSHA_Kota_Resume.pdf.extracted.txt", "Intermediate", "What are the key technical skills mentioned?", ["skill"]),
        ("PDF", "data/uploads/HARSHA_Kota_Resume.pdf.extracted.txt", "Advanced", "List all the internship experiences with company names", ["intern"]),
        ("PDF", "data/uploads/HARSHA_Kota_Resume.pdf.extracted.txt", "Expert", "What role would this candidate be best suited for?", ["role"]),
        ("PDF", "data/uploads/HARSHA_Kota_Resume.pdf.extracted.txt", "God", "Comprehensive resume analysis: extract achievements, assess skill depth, identify gaps for a senior ML engineer role. Give a score 1-10.", ["score", "skill"]),
    ]

    # ----- DOCX TESTS -----
    docx_tests = [
        ("DOCX", "data/samples/test_proposal.docx", "Simple", "What type of document is this?", ["proposal"]),
        ("DOCX", "data/samples/test_proposal.docx", "Intermediate", "Summarize the main content of this document", []),
        ("DOCX", "data/samples/test_proposal.docx", "Advanced", "Extract all key topics and sections from this document", []),
    ]

    # ----- EDGE CASE TESTS -----
    edge_tests = [
        ("EDGE", "data/evaluation/corrupted_data.csv", "Advanced", "Analyze this dataset and report any data quality issues", ["issue"]),
        ("EDGE", "data/evaluation/empty.csv", "Intermediate", "Analyze this file and describe its contents", []),
        ("EDGE", "data/samples/malformed.json", "Advanced", "Analyze this JSON file and report what you find", []),
    ]

    all_tests = csv_tests + json_tests + txt_tests + pdf_tests + docx_tests + edge_tests
    for t in all_tests:
        tid += 1
        tests.append({
            "id": tid,
            "dtype": t[0],
            "file": t[1],
            "difficulty": t[2],
            "query": t[3],
            "keywords": t[4],
        })
    return tests


# ============================================================================
#  EVALUATION
# ============================================================================
def evaluate_response(result, keywords):
    """Score a response on multiple dimensions."""
    scores = {}
    scores["success"] = 1.0 if result.get("success") else 0.0

    response_text = ""
    for key in ["result", "response", "interpretation", "analysis"]:
        val = result.get(key, "")
        if val and isinstance(val, str) and len(val) > len(response_text):
            response_text = val
    if not response_text and isinstance(result.get("result"), dict):
        response_text = json.dumps(result["result"])

    scores["has_response"] = 1.0 if len(response_text) > 20 else 0.0

    if keywords:
        lower_resp = response_text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in lower_resp)
        scores["keyword_match"] = matches / len(keywords) if keywords else 0.0
    else:
        scores["keyword_match"] = 1.0 if scores["has_response"] else 0.0

    resp_len = len(response_text)
    if resp_len > 500:
        scores["quality"] = 1.0
    elif resp_len > 200:
        scores["quality"] = 0.8
    elif resp_len > 50:
        scores["quality"] = 0.5
    elif resp_len > 0:
        scores["quality"] = 0.3
    else:
        scores["quality"] = 0.0

    error = result.get("error", "")
    scores["no_error"] = 0.0 if error else 1.0

    scores["overall"] = (
        scores["success"] * 0.25 +
        scores["has_response"] * 0.15 +
        scores["keyword_match"] * 0.30 +
        scores["quality"] * 0.15 +
        scores["no_error"] * 0.15
    )

    return scores, response_text


def get_available_ram_gb():
    """Return available RAM in GB."""
    try:
        return psutil.virtual_memory().available / (1024**3)
    except:
        return 999


def load_progress():
    """Load existing progress if any."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
            return data.get("detailed_results", [])
        except:
            pass
    return []


def save_progress(completed_results, all_tests):
    """Save current progress to JSON."""
    type_stats = {}
    level_stats = {}
    for r in completed_results:
        dtype = r["dtype"]
        level = r["difficulty"]
        score = r.get("scores", {}).get("overall", 0)
        passed = r["status"] == "PASS"

        for key, stats in [(dtype, type_stats), (level, level_stats)]:
            if key not in stats:
                stats[key] = {"passed": 0, "failed": 0, "total": 0, "scores": []}
            stats[key]["total"] += 1
            stats[key]["scores"].append(score)
            if passed:
                stats[key]["passed"] += 1
            else:
                stats[key]["failed"] += 1

    total_pass = sum(s["passed"] for s in type_stats.values())
    total_all = sum(s["total"] for s in type_stats.values())
    all_scores = [s for st in type_stats.values() for s in st["scores"]]

    report = {
        "test_date": datetime.now().isoformat(),
        "total_tests_defined": len(all_tests),
        "total_tests_completed": len(completed_results),
        "total_passed": total_pass,
        "total_failed": total_all - total_pass,
        "overall_accuracy_pct": round(total_pass / total_all * 100, 2) if total_all > 0 else 0,
        "overall_avg_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
        "by_data_type": {
            dtype: {
                "passed": s["passed"], "failed": s["failed"], "total": s["total"],
                "accuracy_pct": round(s["passed"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
                "avg_score": round(sum(s["scores"]) / len(s["scores"]), 4) if s["scores"] else 0,
            }
            for dtype, s in type_stats.items()
        },
        "by_difficulty": {
            level: {
                "passed": s["passed"], "failed": s["failed"], "total": s["total"],
                "accuracy_pct": round(s["passed"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
                "avg_score": round(sum(s["scores"]) / len(s["scores"]), 4) if s["scores"] else 0,
            }
            for level, s in level_stats.items()
        },
        "detailed_results": completed_results,
    }

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    return report


def generate_markdown(report):
    """Generate Markdown report from results."""
    lines = []
    lines.append("# NEXUS LLM Analytics - Comprehensive Accuracy Test Report\n")
    lines.append(f"**Date:** {report['test_date']}")
    lines.append(f"**Tests Completed:** {report['total_tests_completed']} / {report['total_tests_defined']}")
    lines.append(f"**Overall Accuracy:** {report['overall_accuracy_pct']}%")
    lines.append(f"**Average Score:** {report['overall_avg_score']}\n")

    lines.append("---\n## Results by Data Type\n")
    lines.append("| Type | Pass | Fail | Total | Accuracy | Avg Score |")
    lines.append("|------|------|------|-------|----------|-----------|")
    for dtype in ["CSV", "JSON", "TXT", "PDF", "DOCX", "EDGE"]:
        if dtype in report["by_data_type"]:
            d = report["by_data_type"][dtype]
            lines.append(f"| **{dtype}** | {d['passed']} | {d['failed']} | {d['total']} | {d['accuracy_pct']}% | {d['avg_score']} |")

    lines.append("\n## Results by Difficulty\n")
    lines.append("| Level | Pass | Fail | Total | Accuracy | Avg Score |")
    lines.append("|-------|------|------|-------|----------|-----------|")
    for level in DIFFICULTY_LEVELS:
        if level in report["by_difficulty"]:
            d = report["by_difficulty"][level]
            lines.append(f"| **{level}** | {d['passed']} | {d['failed']} | {d['total']} | {d['accuracy_pct']}% | {d['avg_score']} |")

    lines.append("\n---\n## Detailed Results\n")
    lines.append("| # | Type | Difficulty | File | Query | Status | Score | Agent | Time |")
    lines.append("|---|------|-----------|------|-------|--------|-------|-------|------|")
    for r in report["detailed_results"]:
        fname = Path(r["file"]).name
        q = r["query"][:45] + "..." if len(r["query"]) > 45 else r["query"]
        s = r.get("scores", {}).get("overall", 0)
        a = r.get("agent", "-")
        t = r.get("time", 0)
        lines.append(f"| {r['id']} | {r['dtype']} | {r['difficulty']} | {fname} | {q} | **{r['status']}** | {s:.2f} | {a} | {t}s |")

    # Failed details
    failed = [r for r in report["detailed_results"] if r["status"] != "PASS"]
    if failed:
        lines.append("\n---\n## Failed Tests Detail\n")
        for r in failed:
            lines.append(f"### Test #{r['id']} [{r['status']}] - {r['dtype']} / {r['difficulty']}")
            lines.append(f"- **File:** {r['file']}")
            lines.append(f"- **Query:** {r['query']}")
            if r.get("error"):
                lines.append(f"- **Error:** `{r['error']}`")
            if r.get("response_preview"):
                lines.append(f"- **Response:** {r['response_preview'][:300]}")
            lines.append("")

    lines.append("\n---\n## Scoring Methodology\n")
    lines.append("| Dimension | Weight | Description |")
    lines.append("|-----------|--------|-------------|")
    lines.append("| Success | 25% | Service returned success=true |")
    lines.append("| Has Response | 15% | Non-trivial response returned |")
    lines.append("| Keyword Match | 30% | Expected keywords in response |")
    lines.append("| Quality | 15% | Response length heuristic |")
    lines.append("| No Error | 15% | No error in response |")
    lines.append("\n**Pass threshold:** score >= 0.50")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================================
#  MAIN RUNNER
# ============================================================================
async def run_all_tests():
    """Run all tests one by one with progress saving."""
    all_tests = get_test_suite()
    total = len(all_tests)

    print("=" * 80)
    print("  NEXUS LLM ANALYTICS - COMPREHENSIVE ACCURACY TEST")
    print(f"  Total Tests: {total} | Date: {datetime.now()}")
    print(f"  RAM Available: {get_available_ram_gb():.1f} GB")
    print("=" * 80)

    # Load existing progress
    completed = load_progress()
    completed_ids = {r["id"] for r in completed}
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} tests already done")

    # Import service
    try:
        from backend.services.analysis_service import AnalysisService
        service = AnalysisService()
        print("  [OK] AnalysisService initialized\n")
    except Exception as e:
        print(f"  [FAIL] Cannot init AnalysisService: {e}")
        traceback.print_exc()
        return

    for test in all_tests:
        tid = test["id"]
        if tid in completed_ids:
            continue

        # Memory check
        ram = get_available_ram_gb()
        if ram < 1.0:
            print(f"\n  [LOW RAM] {ram:.1f} GB - forcing GC...")
            gc.collect()
            await asyncio.sleep(5)
            ram = get_available_ram_gb()
            if ram < 0.8:
                print(f"  [STOP] RAM too low ({ram:.1f} GB). Run again to resume.")
                break

        filepath = test["file"]
        full_path = str(ROOT / filepath)
        exists = os.path.exists(full_path)

        print(f"  [{tid}/{total}] {test['dtype']:5s} | {test['difficulty']:13s} | {test['query'][:60]}")

        if not exists:
            entry = {
                "id": tid, "dtype": test["dtype"], "file": filepath,
                "difficulty": test["difficulty"], "query": test["query"],
                "status": "SKIPPED", "reason": "File not found",
                "scores": {"overall": 0}, "time": 0,
            }
            completed.append(entry)
            save_progress(completed, all_tests)
            print(f"           SKIPPED (file missing)")
            continue

        # Build context
        fname = os.path.basename(filepath)
        context = {"filename": fname, "filepath": full_path}

        if filepath.endswith(".extracted.txt") or test["dtype"] == "TXT":
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    context["text_data"] = f.read()[:8000]
            except:
                pass

        start = time.time()
        try:
            result = await asyncio.wait_for(
                service.analyze(test["query"], context),
                timeout=180
            )
            elapsed = time.time() - start

            scores, response_text = evaluate_response(result, test["keywords"])
            passed = scores["overall"] >= 0.5
            status = "PASS" if passed else "FAIL"
            agent = result.get("agent", "unknown")

            print(f"           {status} | score={scores['overall']:.2f} kw={scores['keyword_match']:.0%} q={scores['quality']:.0%} | {agent} | {elapsed:.1f}s")

            entry = {
                "id": tid, "dtype": test["dtype"], "file": filepath,
                "difficulty": test["difficulty"], "query": test["query"],
                "status": status, "agent": agent,
                "scores": scores, "time": round(elapsed, 2),
                "keywords_expected": test["keywords"],
                "response_preview": response_text[:300],
            }

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"           TIMEOUT ({elapsed:.0f}s)")
            entry = {
                "id": tid, "dtype": test["dtype"], "file": filepath,
                "difficulty": test["difficulty"], "query": test["query"],
                "status": "TIMEOUT", "scores": {"overall": 0}, "time": round(elapsed, 2),
            }

        except Exception as e:
            elapsed = time.time() - start
            print(f"           ERROR: {str(e)[:80]}")
            entry = {
                "id": tid, "dtype": test["dtype"], "file": filepath,
                "difficulty": test["difficulty"], "query": test["query"],
                "status": "ERROR", "error": str(e)[:200],
                "scores": {"overall": 0}, "time": round(elapsed, 2),
            }

        completed.append(entry)
        save_progress(completed, all_tests)

        gc.collect()
        await asyncio.sleep(3)

    # Final
    report = save_progress(completed, all_tests)
    generate_markdown(report)

    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print(f"  Completed: {report['total_tests_completed']}/{report['total_tests_defined']}")
    print(f"  Passed: {report['total_passed']} | Failed: {report['total_failed']}")
    print(f"  Accuracy: {report['overall_accuracy_pct']}% | Avg Score: {report['overall_avg_score']}")
    print(f"\n  BY TYPE:")
    for dtype, d in report['by_data_type'].items():
        print(f"    {dtype:6s}: {d['accuracy_pct']:5.1f}% ({d['passed']}/{d['total']})")
    print(f"\n  BY DIFFICULTY:")
    for level in DIFFICULTY_LEVELS:
        if level in report['by_difficulty']:
            d = report['by_difficulty'][level]
            print(f"    {level:13s}: {d['accuracy_pct']:5.1f}% ({d['passed']}/{d['total']})")
    print("=" * 80)


_original_print = print

def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    kwargs.pop('flush', None)
    _original_print(*args, **kwargs, flush=True)

if __name__ == "__main__":
    # Override print globally
    import builtins
    builtins.print = flush_print

    print(f"Python {sys.version}")
    print(f"Starting test at {datetime.now()}")

    # Write a marker file to prove we started
    with open(ROOT / "_test_started.txt", 'w') as f:
        f.write(f"Started at {datetime.now()}")

    try:
        asyncio.run(run_all_tests())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        with open(ROOT / "_test_error.txt", 'w') as f:
            f.write(traceback.format_exc())
