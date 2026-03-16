#!/usr/bin/env python
"""
Nexus LLM Analytics — Performance & Accuracy Benchmark
=======================================================
Runs a focused set of queries through the FULL AnalysisService pipeline,
measuring latency, accuracy, and error handling.

Uses ThreadPoolExecutor for proper timeout enforcement (the service
internally makes blocking LLM calls inside async functions).

Usage:
    python examples/run_benchmark.py
"""

import sys, os, time, json, asyncio, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

# Setup path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluation"
REPORT_DIR = Path(__file__).parent.parent / "reports" / "benchmark"
QUERY_TIMEOUT = 180  # seconds per query (generous for phi3:mini)

# ── Datasets ────────────────────────────────────────────────────────────
def generate_datasets():
    """Create small, reproducible test datasets."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 1. Financial sample (100 rows)
    fp = EVAL_DIR / "financial_sample.csv"
    if not fp.exists():
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=100),
            "Revenue": np.random.uniform(1000, 5000, 100).round(2),
            "Expenses": np.random.uniform(500, 3000, 100).round(2),
            "Region": np.random.choice(["North", "South", "East", "West"], 100),
            "Category": np.random.choice(["Software", "Hardware", "Services"], 100),
        })
        df["Profit"] = (df["Revenue"] - df["Expenses"]).round(2)
        df.to_csv(fp, index=False)
        print(f"  Created {fp.name} ({len(df)} rows)")

    # 2. Large-scale sales (10k rows)
    fp2 = EVAL_DIR / "large_scale_sales.csv"
    if not fp2.exists():
        df2 = pd.DataFrame({
            "TransactionID": range(10_000),
            "Amount": np.random.uniform(10, 1000, 10_000).round(2),
            "CustomerID": np.random.randint(1, 500, 10_000),
            "Date": pd.date_range("2023-01-01", periods=10_000, freq="h"),
        })
        df2.to_csv(fp2, index=False)
        print(f"  Created {fp2.name} ({len(df2)} rows)")

    # Pre-compute ground truth values for validation
    df_fin = pd.read_csv(EVAL_DIR / "financial_sample.csv")
    ground_truth = {
        "total_revenue": df_fin["Revenue"].sum(),
        "avg_profit":    df_fin["Profit"].mean(),
        "max_revenue":   df_fin["Revenue"].max(),
        "num_rows":      len(df_fin),
    }
    print(f"  Ground truth: total_revenue={ground_truth['total_revenue']:.2f}, "
          f"avg_profit={ground_truth['avg_profit']:.2f}")
    return ground_truth


# ── Test Cases ──────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC_01",
        "query": "What is the total revenue?",
        "dataset": "financial_sample.csv",
        "keywords": ["revenue", "total"],
        "gt_key": "total_revenue",
    },
    {
        "id": "TC_02",
        "query": "What is the average profit?",
        "dataset": "financial_sample.csv",
        "keywords": ["profit", "average"],
        "gt_key": "avg_profit",
    },
    {
        "id": "TC_03",
        "query": "Show summary statistics for this dataset",
        "dataset": "financial_sample.csv",
        "keywords": ["mean", "std", "count"],
        "gt_key": None,
    },
    {
        "id": "TC_04",
        "query": "Which region has the highest total revenue?",
        "dataset": "financial_sample.csv",
        "keywords": ["region", "highest"],
        "gt_key": None,
    },
    {
        "id": "TC_05",
        "query": "What is the total transaction amount?",
        "dataset": "large_scale_sales.csv",
        "keywords": ["total", "amount"],
        "gt_key": None,
    },
]


# ── Runner ──────────────────────────────────────────────────────────────
def _run_one_query(service, query: str, filename: str) -> dict:
    """Run a single query synchronously (called from thread)."""
    filepath = str((EVAL_DIR / filename).absolute())
    file_type = "." + filename.rsplit(".", 1)[-1]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            service.analyze(
                query=query,
                context={
                    "filename": filename,
                    "filepath": filepath,
                    "file_type": file_type,
                },
            )
        )
        return result or {"success": False, "error": "Empty result"}
    except Exception as exc:
        return {"success": False, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        loop.close()


def validate_result(result: dict, tc: dict, ground_truth: dict) -> dict:
    """Validate a result against keywords and optional ground truth."""
    v = {"keywords_found": 0, "keywords_total": len(tc["keywords"]),
         "gt_match": None, "gt_error_pct": None}

    response_text = str(result.get("result", "") or result.get("response", "")).lower()

    # Keyword check
    for kw in tc["keywords"]:
        if kw.lower() in response_text:
            v["keywords_found"] += 1

    # Ground truth numeric check
    if tc["gt_key"] and tc["gt_key"] in ground_truth:
        expected = ground_truth[tc["gt_key"]]
        # Try to find the number in the response
        import re
        numbers = re.findall(r"[\d,]+\.?\d*", response_text.replace(",", ""))
        for n_str in numbers:
            try:
                found = float(n_str)
                if expected != 0:
                    error_pct = abs(found - expected) / abs(expected) * 100
                    if error_pct < 5.0:  # Within 5%
                        v["gt_match"] = True
                        v["gt_error_pct"] = round(error_pct, 2)
                        break
            except ValueError:
                continue
        if v["gt_match"] is None:
            v["gt_match"] = False

    return v


def run_benchmark():
    """Main benchmark runner."""
    print("=" * 70)
    print("  NEXUS LLM ANALYTICS — BENCHMARK")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 70)

    # 1. Generate datasets
    print("\n📦 Generating datasets...")
    ground_truth = generate_datasets()

    # 2. Initialize service
    print("\n🔧 Initializing AnalysisService...")
    from backend.services.analysis_service import get_analysis_service
    service = get_analysis_service()
    print("   ✅ Service ready")

    # 3. Run test cases
    results = []
    executor = ThreadPoolExecutor(max_workers=1)

    print(f"\n🚀 Running {len(TEST_CASES)} test cases (timeout={QUERY_TIMEOUT}s each)\n")

    for tc in TEST_CASES:
        tc_id = tc["id"]
        print(f"─── {tc_id}: {tc['query'][:60]} ───")
        print(f"    Dataset: {tc['dataset']}")

        start = time.time()
        try:
            future = executor.submit(_run_one_query, service, tc["query"], tc["dataset"])
            result = future.result(timeout=QUERY_TIMEOUT)
            elapsed = time.time() - start

            success = result.get("success", False)
            validation = validate_result(result, tc, ground_truth)

            entry = {
                "id": tc_id,
                "query": tc["query"],
                "dataset": tc["dataset"],
                "success": success,
                "duration_s": round(elapsed, 2),
                "keywords_found": validation["keywords_found"],
                "keywords_total": validation["keywords_total"],
                "gt_match": validation["gt_match"],
                "gt_error_pct": validation["gt_error_pct"],
                "error": result.get("error"),
                "response_preview": str(result.get("result", ""))[:200],
                "metadata": {k: str(v)[:100] for k, v in (result.get("metadata") or {}).items()},
            }
            results.append(entry)

            status = "✅" if success else "❌"
            kw = f"{validation['keywords_found']}/{validation['keywords_total']} keywords"
            gt = ""
            if validation["gt_match"] is not None:
                gt = f" | GT: {'✅' if validation['gt_match'] else '❌'}"
                if validation["gt_error_pct"] is not None:
                    gt += f" ({validation['gt_error_pct']}% err)"

            print(f"    {status} {elapsed:.1f}s | {kw}{gt}")
            if not success:
                print(f"    Error: {result.get('error', 'unknown')[:100]}")

        except FuturesTimeout:
            elapsed = time.time() - start
            results.append({
                "id": tc_id, "query": tc["query"], "dataset": tc["dataset"],
                "success": False, "duration_s": round(elapsed, 2),
                "error": f"TIMEOUT after {QUERY_TIMEOUT}s",
                "keywords_found": 0, "keywords_total": len(tc["keywords"]),
            })
            print(f"    ⏰ TIMEOUT after {elapsed:.0f}s")

        except Exception as exc:
            elapsed = time.time() - start
            results.append({
                "id": tc_id, "query": tc["query"], "dataset": tc["dataset"],
                "success": False, "duration_s": round(elapsed, 2),
                "error": str(exc),
            })
            print(f"    💥 ERROR: {exc}")

    executor.shutdown(wait=False)

    # 4. Summary
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    avg_latency = sum(r["duration_s"] for r in results) / total if total else 0
    kw_accuracy = (
        sum(r.get("keywords_found", 0) for r in results) /
        sum(r.get("keywords_total", 1) for r in results) * 100
    ) if total else 0
    gt_results = [r for r in results if r.get("gt_match") is not None]
    gt_accuracy = (sum(1 for r in gt_results if r["gt_match"]) / len(gt_results) * 100) if gt_results else None

    print(f"\n  Total:        {total}")
    print(f"  Passed:       {passed} ({passed/total*100:.0f}%)")
    print(f"  Failed:       {failed}")
    print(f"  Avg Latency:  {avg_latency:.1f}s")
    print(f"  KW Accuracy:  {kw_accuracy:.0f}%")
    if gt_accuracy is not None:
        print(f"  GT Accuracy:  {gt_accuracy:.0f}% ({len(gt_results)} cases)")
    print()

    # 5. Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total, "passed": passed, "failed": failed,
            "avg_latency_s": round(avg_latency, 2),
            "keyword_accuracy_pct": round(kw_accuracy, 1),
            "ground_truth_accuracy_pct": round(gt_accuracy, 1) if gt_accuracy else None,
        },
        "ground_truth": {k: round(v, 2) for k, v in ground_truth.items()},
        "results": results,
    }
    report_path = REPORT_DIR / f"benchmark_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  📄 Report saved: {report_path}")

    # Also write a markdown summary
    md_path = REPORT_DIR / f"benchmark_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Benchmark Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total Tests | {total} |\n")
        f.write(f"| Passed | {passed} ({passed/total*100:.0f}%) |\n")
        f.write(f"| Avg Latency | {avg_latency:.1f}s |\n")
        f.write(f"| Keyword Accuracy | {kw_accuracy:.0f}% |\n")
        if gt_accuracy is not None:
            f.write(f"| Ground Truth Accuracy | {gt_accuracy:.0f}% |\n")
        f.write(f"\n## Results\n\n")
        f.write(f"| ID | Query | Duration | Status | Keywords | GT |\n")
        f.write(f"|-----|-------|----------|--------|----------|----|\n")
        for r in results:
            s = "✅" if r["success"] else "❌"
            kw = f"{r.get('keywords_found',0)}/{r.get('keywords_total',0)}"
            gt_s = ""
            if r.get("gt_match") is not None:
                gt_s = "✅" if r["gt_match"] else "❌"
            f.write(f"| {r['id']} | {r['query'][:40]} | {r['duration_s']}s | {s} | {kw} | {gt_s} |\n")

    print(f"  📄 Markdown:  {md_path}")
    print("\n✨ Benchmark complete!")

    return report


if __name__ == "__main__":
    run_benchmark()
