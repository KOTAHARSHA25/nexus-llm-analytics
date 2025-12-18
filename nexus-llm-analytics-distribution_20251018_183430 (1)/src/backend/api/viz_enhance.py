"""
Enhanced Visualization API with LIDA-Inspired Features.
Provides editing, explanation, evaluation, repair, and recommendation capabilities.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import logging
import json
from backend.core.crew_singleton import get_crew_manager
from backend.visualization.scaffold import ChartScaffold, VisualizationGoal
from backend.utils.data_utils import clean_code_snippet, create_data_summary, read_dataframe
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)


# ==================== REQUEST/RESPONSE MODELS ====================

class ChartEditRequest(BaseModel):
    """Request to edit visualization with natural language instructions"""
    code: str
    instructions: Union[str, List[str]]
    filename: str
    library: str = "plotly"


class ChartExplainRequest(BaseModel):
    """Request to explain visualization code"""
    code: str
    library: str = "plotly"


class ChartEvaluateRequest(BaseModel):
    """Request to evaluate visualization quality"""
    code: str
    goal: str  # The visualization goal/question
    library: str = "plotly"


class ChartRepairRequest(BaseModel):
    """Request to repair visualization based on feedback"""
    code: str
    goal: str
    feedback: Union[str, List[str], List[Dict[str, Any]]]
    filename: str
    library: str = "plotly"


class ChartRecommendRequest(BaseModel):
    """Request visualization recommendations"""
    code: str
    filename: str
    n: int = 3  # Number of recommendations
    library: str = "plotly"


class PersonaGoalsRequest(BaseModel):
    """Request persona-based visualization goals"""
    filename: str
    persona: Optional[str] = None  # e.g., "CEO", "Data Analyst", "Doctor"
    n: int = 5


# ==================== HELPER FUNCTIONS ====================

def get_data_path(filename: str) -> str:
    """Get full path for uploaded file using centralized resolver"""
    from backend.utils.data_utils import DataPathResolver
    
    filepath = DataPathResolver.resolve_data_file(filename)
    if not filepath:
        raise FileNotFoundError(f"File '{filename}' not found")
    
    return str(filepath)


def preprocess_code(code: str) -> str:
    """Preprocess code for execution (uses centralized function)"""
    from backend.utils.data_utils import preprocess_visualization_code
    return preprocess_visualization_code(code, library="plotly")


# ==================== EDIT ENDPOINT ====================

@router.post("/edit")
async def edit_visualization(request: ChartEditRequest):
    """
    Edit a visualization using natural language instructions.
    
    Example instructions:
    - "Change the color to red"
    - "Convert to a bar chart"
    - "Add a trend line"
    - "Rotate x-axis labels"
    """
    logger.info(f"[VIZ_EDIT] Editing chart with {len(request.instructions) if isinstance(request.instructions, list) else 1} instructions")
    
    try:
        # Load data summary
        filepath = get_data_path(request.filename)
        df = read_dataframe(filepath)
        data_summary = create_data_summary(df, request.filename)
        
        # Convert instructions to list
        if isinstance(request.instructions, str):
            instructions = [request.instructions]
        else:
            instructions = request.instructions
        
        # Build instruction string
        instruction_string = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(instructions)])
        
        # Get scaffold template
        scaffold = ChartScaffold()
        template, library_instructions = scaffold.get_template(
            VisualizationGoal(question="", visualization=""),
            request.library
        )
        
        # Build prompt for CrewAI
        prompt = f"""
        Dataset Summary: {json.dumps(data_summary, indent=2)}
        
        Current Visualization Code:
        ```python
        {request.code}
        ```
        
        Instructions to Apply:
        {instruction_string}
        
        Library: {request.library}
        Template: {template}
        
        Task: Modify the visualization code to meet ALL the instructions above.
        
        Requirements:
        1. Keep ALL existing functionality unless explicitly changed
        2. Apply ONLY the requested modifications
        3. Ensure code is syntactically correct
        4. Use only fields from the dataset summary
        5. Follow {request.library} best practices
        6. Return COMPLETE executable code
        
        Return ONLY the modified Python code wrapped in ```python ``` markers.
        """
        
        # Get edited code from CrewAI
        crew_manager = get_crew_manager()
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary=data_summary
        )
        
        # Extract and clean code
        edited_code = result.get("analysis", "")
        edited_code = preprocess_code(edited_code)
        
        return {
            "success": True,
            "edited_code": edited_code,
            "instructions_applied": instructions,
            "library": request.library,
            "message": f"Applied {len(instructions)} modifications successfully"
        }
        
    except Exception as e:
        logger.error(f"[VIZ_EDIT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")


# ==================== EXPLAIN ENDPOINT ====================

@router.post("/explain")
async def explain_visualization(request: ChartExplainRequest):
    """
    Explain a visualization code in human-readable format.
    
    Returns explanations for:
    - Accessibility: Physical appearance, colors, insights
    - Transformation: Data filtering, aggregation, grouping
    - Visualization: Step-by-step code explanation
    """
    logger.info(f"[VIZ_EXPLAIN] Explaining {request.library} visualization")
    
    try:
        # Build prompt
        prompt = f"""
        Analyze this {request.library} visualization code:
        
        ```python
        {request.code}
        ```
        
        Provide a structured explanation in JSON format with exactly 3 sections:
        
        1. **accessibility**: Describe the physical appearance (colors, chart type), the goal of the chart, and main insights
        2. **transformation**: Describe data transformations (filtering, aggregation, grouping, null handling)
        3. **visualization**: Step-by-step description of how the chart is created
        
        Return ONLY a JSON array in this format:
        ```json
        [
            {{"section": "accessibility", "code": "relevant code snippet or None", "explanation": "detailed explanation"}},
            {{"section": "transformation", "code": "relevant code snippet", "explanation": "detailed explanation"}},
            {{"section": "visualization", "code": "relevant code snippet", "explanation": "detailed explanation"}}
        ]
        ```
        """
        
        crew_manager = get_crew_manager()
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary={}
        )
        
        # Extract explanation
        explanation_text = result.get("analysis", "")
        explanation_text = clean_code_snippet(explanation_text)
        
        # Try to parse as JSON
        try:
            explanations = json.loads(explanation_text)
        except json.JSONDecodeError:
            # If not valid JSON, return as structured text
            explanations = [
                {
                    "section": "general",
                    "code": None,
                    "explanation": explanation_text
                }
            ]
        
        return {
            "success": True,
            "explanations": explanations,
            "library": request.library,
            "code_length": len(request.code)
        }
        
    except Exception as e:
        logger.error(f"[VIZ_EXPLAIN] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# ==================== EVALUATE ENDPOINT ====================

@router.post("/evaluate")
async def evaluate_visualization(request: ChartEvaluateRequest):
    """
    Evaluate visualization quality on 6 dimensions.
    
    Dimensions:
    - bugs: Syntax errors, logic errors, potential failures
    - transformation: Appropriate data transformations
    - compliance: How well it meets the goal
    - type: Is the visualization type appropriate?
    - encoding: Are data encodings appropriate?
    - aesthetics: Visual appeal and effectiveness
    
    Each dimension scored 1 (bad) - 10 (good) with rationale.
    """
    logger.info(f"[VIZ_EVAL] Evaluating {request.library} visualization")
    
    try:
        prompt = f"""
        Evaluate this {request.library} visualization code:
        
        Goal: {request.goal}
        
        Code:
        ```python
        {request.code}
        ```
        
        Evaluate on 6 dimensions and return JSON array:
        
        1. **bugs**: Are there bugs, syntax errors, or reasons code may fail? (bugs exist = score < 5)
        2. **transformation**: Is data transformed appropriately for visualization type?
        3. **compliance**: How well does code meet the specified goal?
        4. **type**: Is visualization type appropriate? (if not = score < 5)
        5. **encoding**: Is data encoded appropriately?
        6. **aesthetics**: Are aesthetics appropriate?
        
        Score each 1-10. Return ONLY JSON:
        ```json
        [
            {{"dimension": "bugs", "score": X, "rationale": "..."}},
            {{"dimension": "transformation", "score": X, "rationale": "..."}},
            {{"dimension": "compliance", "score": X, "rationale": "..."}},
            {{"dimension": "type", "score": X, "rationale": "..."}},
            {{"dimension": "encoding", "score": X, "rationale": "..."}},
            {{"dimension": "aesthetics", "score": X, "rationale": "..."}}
        ]
        ```
        """
        
        crew_manager = get_crew_manager()
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary={}
        )
        
        # Extract evaluations
        eval_text = result.get("analysis", "")
        eval_text = clean_code_snippet(eval_text)
        
        try:
            evaluations = json.loads(eval_text)
            
            # Calculate average score
            avg_score = sum(e.get("score", 0) for e in evaluations) / len(evaluations)
            
            return {
                "success": True,
                "evaluations": evaluations,
                "average_score": round(avg_score, 2),
                "library": request.library,
                "goal": request.goal,
                "passed": avg_score >= 7.0  # Consider passing if avg >= 7
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse evaluation results",
                "raw_response": eval_text
            }
        
    except Exception as e:
        logger.error(f"[VIZ_EVAL] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ==================== REPAIR ENDPOINT ====================

@router.post("/repair")
async def repair_visualization(request: ChartRepairRequest):
    """
    Repair visualization based on feedback from evaluation or user.
    
    Feedback can be:
    - String: "The chart has syntax errors"
    - List of strings: ["Fix the colors", "Add labels"]
    - List of evaluation dicts: Output from /evaluate endpoint
    """
    logger.info(f"[VIZ_REPAIR] Repairing {request.library} visualization")
    
    try:
        # Load data summary
        filepath = get_data_path(request.filename)
        df = read_dataframe(filepath)
        data_summary = create_data_summary(df, request.filename)
        
        # Format feedback
        if isinstance(request.feedback, str):
            feedback_str = request.feedback
        elif isinstance(request.feedback, list):
            if all(isinstance(f, dict) for f in request.feedback):
                # Evaluation feedback
                feedback_str = "\n".join([
                    f"- {f.get('dimension', 'Issue')}: Score {f.get('score', 0)}/10 - {f.get('rationale', 'No details')}"
                    for f in request.feedback
                ])
            else:
                # List of strings
                feedback_str = "\n".join([f"- {f}" for f in request.feedback])
        else:
            feedback_str = str(request.feedback)
        
        # Get scaffold
        scaffold = ChartScaffold()
        template, _ = scaffold.get_template(
            VisualizationGoal(question=request.goal),
            request.library
        )
        
        prompt = f"""
        Dataset Summary: {json.dumps(data_summary, indent=2)}
        
        Original Goal: {request.goal}
        
        Current Code:
        ```python
        {request.code}
        ```
        
        Feedback to Address:
        {feedback_str}
        
        Library: {request.library}
        Template: {template}
        
        Task: Fix the code to address ALL valid feedback points.
        
        Requirements:
        1. Address each feedback point that is CORRECT
        2. Do NOT break existing working functionality
        3. Use only fields from dataset summary
        4. Follow {request.library} best practices
        5. Ensure code is syntactically correct
        
        Return ONLY the repaired Python code wrapped in ```python ``` markers.
        """
        
        crew_manager = get_crew_manager()
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary=data_summary
        )
        
        repaired_code = result.get("analysis", "")
        repaired_code = preprocess_code(repaired_code)
        
        return {
            "success": True,
            "repaired_code": repaired_code,
            "feedback_addressed": feedback_str,
            "library": request.library,
            "goal": request.goal
        }
        
    except Exception as e:
        logger.error(f"[VIZ_REPAIR] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Repair failed: {str(e)}")


# ==================== RECOMMEND ENDPOINT ====================

@router.post("/recommend")
async def recommend_visualizations(request: ChartRecommendRequest):
    """
    Recommend alternative visualizations based on current chart.
    
    Generates diverse recommendations considering:
    - Different chart types
    - Different data aggregations
    - Different variables from dataset
    - Clearer ways to display information
    """
    logger.info(f"[VIZ_RECOMMEND] Generating {request.n} recommendations")
    
    try:
        # Load data summary
        filepath = get_data_path(request.filename)
        df = read_dataframe(filepath)
        data_summary = create_data_summary(df, request.filename)
        
        # Get scaffold
        scaffold = ChartScaffold()
        template, _ = scaffold.get_template(
            VisualizationGoal(),
            request.library
        )
        
        prompt = f"""
        Dataset Summary: {json.dumps(data_summary, indent=2)}
        
        Current Visualization:
        ```python
        {request.code}
        ```
        
        Library: {request.library}
        Template: {template}
        
        Task: Recommend {request.n} DIVERSE alternative visualizations.
        
        Each recommendation should:
        1. Use different chart types where appropriate
        2. Consider different data aggregations
        3. Use different variables from the dataset
        4. Provide clearer insights than the original
        5. Follow visualization best practices
        
        Separate each code snippet with exactly 5 asterisks (*****).
        
        Return format:
        ```python
        # Recommendation 1: <brief description>
        import ...
        def plot(data):
            ...
        chart = plot(data)
        ```
        *****
        ```python
        # Recommendation 2: <brief description>
        import ...
        def plot(data):
            ...
        chart = plot(data)
        ```
        *****
        ... (continue for {request.n} recommendations)
        """
        
        crew_manager = get_crew_manager()
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary=data_summary
        )
        
        # Split recommendations
        recommendations_text = result.get("analysis", "")
        snippets = recommendations_text.split("*****")
        
        recommendations = []
        for i, snippet in enumerate(snippets[:request.n]):
            cleaned = preprocess_code(snippet)
            if len(cleaned.strip()) > 10:
                recommendations.append({
                    "index": i + 1,
                    "code": cleaned,
                    "description": f"Alternative visualization {i + 1}"
                })
        
        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations),
            "library": request.library,
            "based_on_code_length": len(request.code)
        }
        
    except Exception as e:
        logger.error(f"[VIZ_RECOMMEND] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


# ==================== PERSONA GOALS ENDPOINT ====================

@router.post("/persona-goals")
async def generate_persona_goals(request: PersonaGoalsRequest):
    """
    Generate visualization goals based on a persona.
    
    Personas can be:
    - CEO: High-level business metrics, KPIs, trends
    - Data Analyst: Detailed statistical analysis, correlations
    - Doctor: Patient outcomes, treatment effectiveness
    - Accountant: Financial metrics, cost analysis
    - (or auto-detect from data)
    """
    logger.info(f"[PERSONA_GOALS] Generating goals for persona: {request.persona or 'auto'}")
    
    try:
        # Load data summary
        filepath = get_data_path(request.filename)
        df = read_dataframe(filepath)
        data_summary = create_data_summary(df, request.filename)
        
        if request.persona:
            persona_instruction = f"The goals should be focused on the interests and perspective of a '{request.persona}' persona."
        else:
            persona_instruction = "Auto-detect the most relevant persona based on the dataset and generate appropriate goals."
        
        prompt = f"""
        Dataset Summary: {json.dumps(data_summary, indent=2)}
        
        Task: Generate {request.n} insightful visualization goals.
        
        {persona_instruction}
        
        Each goal should include:
        - question: What question does this visualization answer?
        - visualization: What type of chart? (be specific)
        - rationale: Why is this visualization useful? What insights will it reveal?
        - fields: Which exact dataset fields will be used?
        
        The visualizations MUST:
        1. Follow best practices (e.g., bar charts not pie charts for comparisons)
        2. Be meaningful and insightful
        3. Use exact field names from the dataset summary
        4. Be relevant to the persona's interests
        
        Return ONLY a JSON array:
        ```json
        [
            {{
                "index": 0,
                "question": "What is the distribution of X?",
                "visualization": "histogram of X",
                "rationale": "This shows the spread and identifies outliers",
                "fields": ["X"]
            }},
            ... ({request.n} total)
        ]
        ```
        """
        
        crew_manager = get_crew_manager()
        result = crew_manager.analyze_structured_data(
            query=prompt,
            data_summary=data_summary
        )
        
        # Parse goals
        goals_text = result.get("analysis", "")
        goals_text = clean_code_snippet(goals_text)
        
        try:
            goals = json.loads(goals_text)
            
            return {
                "success": True,
                "goals": goals,
                "count": len(goals),
                "persona": request.persona or "auto-detected",
                "dataset": request.filename
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse goals",
                "raw_response": goals_text
            }
        
    except Exception as e:
        logger.error(f"[PERSONA_GOALS] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Goal generation failed: {str(e)}")
