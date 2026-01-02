"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - TEST COVERAGE RUNNER
═══════════════════════════════════════════════════════════════════════════════

Phase 4.5: Automated test coverage measurement and reporting.

Usage:
    python scripts/run_coverage.py              # Run all tests with coverage
    python scripts/run_coverage.py --html       # Generate HTML report
    python scripts/run_coverage.py --module core # Run specific module tests
    python scripts/run_coverage.py --threshold 80 # Set coverage threshold

Version: 1.0.0
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_coverage(
    test_path: str = "tests",
    modules: list = None,
    generate_html: bool = True,
    generate_xml: bool = True,
    threshold: int = 60,
    verbose: bool = True
) -> dict:
    """
    Run pytest with coverage measurement.
    
    Args:
        test_path: Path to tests directory
        modules: Specific modules to test (e.g., ['core', 'agents'])
        generate_html: Generate HTML coverage report
        generate_xml: Generate XML report (for CI)
        threshold: Minimum coverage percentage
        verbose: Show verbose output
    
    Returns:
        Dict with coverage results
    """
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        f"--cov=src/backend",
        "--cov-report=term-missing",
        f"--cov-fail-under={threshold}",
    ]
    
    if generate_html:
        cmd.append("--cov-report=html:reports/coverage")
    
    if generate_xml:
        cmd.append("--cov-report=xml:reports/coverage.xml")
    
    # Always generate JSON for programmatic access
    cmd.append("--cov-report=json:reports/coverage.json")
    
    if verbose:
        cmd.append("-v")
    
    # Filter by modules if specified
    if modules:
        module_filter = " or ".join([f"test_{m}" for m in modules])
        cmd.extend(["-k", module_filter])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run coverage
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=False
    )
    
    # Parse results
    coverage_data = parse_coverage_json(reports_dir / "coverage.json")
    
    return {
        "success": result.returncode == 0,
        "return_code": result.returncode,
        "threshold": threshold,
        "coverage": coverage_data,
        "reports": {
            "html": str(reports_dir / "coverage" / "index.html") if generate_html else None,
            "xml": str(reports_dir / "coverage.xml") if generate_xml else None,
            "json": str(reports_dir / "coverage.json")
        },
        "generated_at": datetime.now().isoformat()
    }


def parse_coverage_json(json_path: Path) -> dict:
    """Parse coverage.json to extract key metrics"""
    if not json_path.exists():
        return {"error": "Coverage JSON not found"}
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        totals = data.get("totals", {})
        files = data.get("files", {})
        
        # Calculate per-module coverage
        module_coverage = {}
        for filepath, file_data in files.items():
            # Extract module name from path
            parts = Path(filepath).parts
            if "backend" in parts:
                idx = parts.index("backend")
                if idx + 1 < len(parts):
                    module = parts[idx + 1]
                    if module not in module_coverage:
                        module_coverage[module] = {
                            "files": 0,
                            "covered_lines": 0,
                            "total_lines": 0
                        }
                    summary = file_data.get("summary", {})
                    module_coverage[module]["files"] += 1
                    module_coverage[module]["covered_lines"] += summary.get("covered_lines", 0)
                    module_coverage[module]["total_lines"] += summary.get("num_statements", 0)
        
        # Calculate module percentages
        for module, data in module_coverage.items():
            if data["total_lines"] > 0:
                data["percent_covered"] = round(
                    data["covered_lines"] / data["total_lines"] * 100, 2
                )
            else:
                data["percent_covered"] = 0.0
        
        return {
            "overall": {
                "percent_covered": totals.get("percent_covered", 0),
                "covered_lines": totals.get("covered_lines", 0),
                "num_statements": totals.get("num_statements", 0),
                "missing_lines": totals.get("missing_lines", 0),
                "excluded_lines": totals.get("excluded_lines", 0),
                "num_branches": totals.get("num_branches", 0),
                "covered_branches": totals.get("covered_branches", 0)
            },
            "by_module": module_coverage,
            "files_analyzed": len(files)
        }
    except Exception as e:
        return {"error": str(e)}


def print_coverage_summary(coverage_data: dict):
    """Print formatted coverage summary"""
    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    
    overall = coverage_data.get("overall", {})
    print(f"\n📊 Overall Coverage: {overall.get('percent_covered', 0):.2f}%")
    print(f"   Covered Lines: {overall.get('covered_lines', 0)}/{overall.get('num_statements', 0)}")
    print(f"   Missing Lines: {overall.get('missing_lines', 0)}")
    
    if overall.get("num_branches"):
        branch_pct = (overall.get("covered_branches", 0) / overall.get("num_branches", 1)) * 100
        print(f"   Branch Coverage: {branch_pct:.2f}%")
    
    print(f"\n📁 Files Analyzed: {coverage_data.get('files_analyzed', 0)}")
    
    by_module = coverage_data.get("by_module", {})
    if by_module:
        print("\n📦 Coverage by Module:")
        sorted_modules = sorted(
            by_module.items(), 
            key=lambda x: x[1].get("percent_covered", 0),
            reverse=True
        )
        for module, data in sorted_modules:
            pct = data.get("percent_covered", 0)
            status = "✅" if pct >= 80 else "⚠️" if pct >= 60 else "❌"
            print(f"   {status} {module}: {pct:.1f}% ({data.get('files', 0)} files)")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run test coverage for Nexus LLM Analytics"
    )
    parser.add_argument(
        "--test-path",
        default="tests",
        help="Path to tests directory"
    )
    parser.add_argument(
        "--module",
        "-m",
        action="append",
        dest="modules",
        help="Specific modules to test (can repeat)"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        default=True,
        help="Generate XML coverage report (for CI)"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=60,
        help="Minimum coverage percentage (default: 60)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    generate_html = not args.no_html
    
    results = run_coverage(
        test_path=args.test_path,
        modules=args.modules,
        generate_html=generate_html,
        generate_xml=args.xml,
        threshold=args.threshold,
        verbose=not args.quiet
    )
    
    # Print summary
    if not args.quiet and results.get("coverage"):
        print_coverage_summary(results["coverage"])
    
    # Print report locations
    if generate_html:
        print(f"\n📄 HTML Report: {results['reports']['html']}")
    if args.xml:
        print(f"📄 XML Report: {results['reports']['xml']}")
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
