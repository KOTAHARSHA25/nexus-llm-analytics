import pytest
import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path for proper imports
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from backend.agents.controller_agent import ControllerAgent
from backend.core.sandbox import Sandbox
from backend.agents.review_agent import ReviewAgent

def test_controller_summarize():
    controller = ControllerAgent()
    result = controller.handle_query('summarize', filename='analyze.json')
    assert 'preview' in result

def test_sandbox_exec():
    sandbox = Sandbox()
    df = pd.DataFrame({'a': [1,2]})
    code = 'result = data["a"].sum()'
    out = sandbox.execute(code, data=df)
    assert out['result']['result'] == 3

def test_review_agent_safe():
    review = ReviewAgent()
    code = 'result = 1 + 2'
    res = review.review(code)
    assert res['status'] == 'ok'
