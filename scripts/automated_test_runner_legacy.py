
"""
Automated Test Runner
=====================
Executes comprehensive test scenarios against the AnalysisService.
Reads questions from COMPREHENSIVE_TEST_QUESTIONS.md and asserts success.
"""

import asyncio
import re
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add src to path
SRC_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(SRC_DIR))

from backend.services.analysis_service import get_analysis_service

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_DATA_DIR = Path(__file__).parent / "data"

@dataclass
class TestScenario:
    id: str
    category: str
    filename: str
    query: str
    expected_outcome: str = "success"

@dataclass
class TestResult:
    scenario_id: str
    success: bool
    execution_time: float
    error: str = None
    result_summary: str = None
    agent_used: str = None

class AutomatedTestRunner:
    def __init__(self):
        self.service = get_analysis_service()
        self.results: List[TestResult] = []
        
    def parse_scenarios(self, md_path: Path) -> List[TestScenario]:
        """Parse scenarios from markdown file"""
        with open(md_path, 'r') as f:
            content = f.read()
            
        scenarios = []
        
        # Regex to find sections: ## X. Title (File: filename)
        sections = re.split(r'## \d+\. ', content)
        
        for section in sections[1:]: # Skip header
            lines = section.strip().split('\n')
            header = lines[0]
            
            # Extract filename
            file_match = re.search(r'\(File: (.*?)\)', header)
            if not file_match:
                continue
            filename = file_match.group(1)
            category = header.split('(')[0].strip()
            
            # Extract questions (numbered list)
            for line in lines:
                question_match = re.match(r'\d+\. "(.*?)"', line)
                if question_match:
                    query = question_match.group(1)
                    s_id = f"{category[:3].upper()}-{len(scenarios)+1}"
                    scenarios.append(TestScenario(
                        id=s_id,
                        category=category,
                        filename=filename,
                        query=query
                    ))
                    
        return scenarios

    async def run_scenario(self, scenario: TestScenario) -> TestResult:
        logger.info(f"Running {scenario.id}: {scenario.query} [{scenario.filename}]")
        start_time = time.time()
        
        try:
            # Check if file exists
            file_path = TEST_DATA_DIR / scenario.filename
            if not file_path.exists() and "No specific file" not in scenario.filename:
                return TestResult(
                    scenario_id=scenario.id,
                    success=False,
                    execution_time=0,
                    error=f"File not found: {scenario.filename}"
                )
            
            context = {'filename': scenario.filename}
            if file_path.exists():
                context['filepath'] = str(file_path)
            
            # Execute analysis
            response = await self.service.analyze(
                query=scenario.query,
                context=context
            )
            
            # Verify result
            success = response.get('success', False)
            error_msg = response.get('error')
            agent = response.get('agent', 'Unknown')
            
            # Summarize result
            result_content = response.get('result')
            summary = str(result_content)[:100] + "..." if result_content else "No result"
            
            return TestResult(
                scenario_id=scenario.id,
                success=success,
                execution_time=time.time() - start_time,
                error=error_msg,
                result_summary=summary,
                agent_used=agent
            )
            
        except Exception as e:
            logger.error(f"Scenario {scenario.id} failed with exception: {e}")
            return TestResult(
                scenario_id=scenario.id,
                success=False,
                execution_time=time.time() - start_time,
                error=str(e)
            )

    async def run_all(self):
        md_file = TEST_DATA_DIR / "COMPREHENSIVE_TEST_QUESTIONS.md"
        scenarios = self.parse_scenarios(md_file)
        logger.info(f"Loaded {len(scenarios)} scenarios.")
        
        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            self.results.append(result)
            
            # Brief pause to avoid rate limits if any
            await asyncio.sleep(0.5)
            
        self.generate_report()
        
    def generate_report(self):
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        failed = total - passed
        
        logger.info("="*50)
        logger.info(f"TEST COMPLETE. Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        logger.info("="*50)
        
        report_data = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed/total if total > 0 else 0
            },
            "details": [asdict(r) for r in self.results]
        }
        
        report_path = TEST_DATA_DIR.parent / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    runner = AutomatedTestRunner()
    asyncio.run(runner.run_all())
