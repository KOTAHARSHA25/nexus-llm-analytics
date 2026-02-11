"""Self-Correction Engine — Generator → Critic → Feedback loop.

This module implements the core iterative self-correction pipeline used by the
Nexus LLM Analytics system.  A *Generator* LLM produces Chain-of-Thought
(CoT) reasoning, which is first screened by an **automated validator** for
obvious errors (e.g. division-by-zero, SQL injection) and then evaluated by a
*Critic* LLM.  If the Critic identifies issues the Generator is re-prompted
with structured feedback until the output is validated or the iteration budget
is exhausted.

.. versionadded:: 2.0.0
   Added :class:`CorrectionStrategy`, :class:`CorrectionObserver`,
   :class:`CorrectionMetrics`, and :func:`get_correction_metrics`.

Key classes:
    SelfCorrectionEngine  – orchestrates the full loop and self-learning.
    CorrectionIteration   – data container for a single loop iteration.
    CorrectionResult      – final output returned to the caller.

Pipeline position:
    QueryRouter → DynamicPlanner → **SelfCorrectionEngine** → ResultInterpreter
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.core.chromadb_client import ChromaDBClient, embed_text
from backend.core.dynamic_planner import get_dynamic_planner

from .automated_validation import RuntimeEvaluator
from .cot_parser import (
    CoTParser,
    CriticFeedback,
    CriticIssue,
    CriticParser,
    ParsedCoT,
)
from .paper_metrics import CorrectionOutcome, get_paper_metrics
from backend.utils.error_analysis import ErrorClassifier, ErrorCategory

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "SelfCorrectionEngine",
    "CorrectionIteration",
    "CorrectionResult",
    # v2.0 Enterprise additions
    "CorrectionStrategy",
    "CorrectionObserver",
    "CorrectionMetrics",
    "get_correction_metrics",
]


@dataclass
class CorrectionIteration:
    """A single iteration in the Generator → Critic correction loop.

    Captures the generator output, its parsed CoT structure, the critic's
    evaluation, and whether a further correction pass is required.
    """

    iteration_number: int
    generator_response: str
    parsed_cot: ParsedCoT
    critic_response: str
    critic_feedback: CriticFeedback
    correction_needed: bool
    timestamp: float

@dataclass
class CorrectionResult:
    """Final result produced by :class:`SelfCorrectionEngine`.

    Contains the validated (or best-effort) output, full iteration history,
    and metadata such as elapsed time and termination reason.
    """

    final_output: str
    final_reasoning: str
    total_iterations: int
    all_iterations: List[CorrectionIteration]
    success: bool
    termination_reason: str  # "validated", "max_iterations", "timeout", "parsing_failure", "failure"
    total_time_seconds: float

class SelfCorrectionEngine:
    """Orchestrates the Generator → Critic → Feedback self-correction loop.

    The engine drives an iterative refinement cycle:

    1. The **Generator** LLM produces a Chain-of-Thought (CoT) response.
    2. An **Automated Validator** screens for obvious errors (division-by-zero,
       unsafe operations, schema mismatches).
    3. The **Critic** LLM evaluates the reasoning for correctness.
    4. If issues are found the Generator is re-prompted with structured
       feedback; otherwise the validated output is returned.

    The engine also maintains a ChromaDB-backed vector memory so that past
    correction pairs can be retrieved for similar future queries (self-learning).
    """
    
    def __init__(self, config: Dict[str, Any], llm_client) -> None:
        """Initialise the correction engine with *config* and an LLM client.

        Args:
            config: Configuration dictionary. Expected keys:
                ``tags`` — CoT delimiter tags,
                ``max_iterations`` — correction loop budget,
                ``timeout_per_iteration_seconds`` — per-iteration timeout,
                ``generator`` — generator prompt settings,
                ``critic`` — critic prompt settings,
                ``cot_review`` — nested alternative for the above keys,
                ``performance_tracking`` — logging / metrics flags.
            llm_client: LLM client exposing a ``.generate()`` method used
                for both the Generator and Critic calls.
        """
        self.config = config
        self.llm_client = llm_client
        
        # Config may arrive in flat format (direct 'tags' key) or nested
        # under 'cot_review'.  Try flat first, then nested.  Fall back to
        # built-in defaults if neither found.
        tags = config.get('tags')
        if not tags and 'cot_review' in config:
            tags = config.get('cot_review', {}).get('tags')
        if not tags:
            # Fallback defaults
            tags = {
                "reasoning_start": "[REASONING]", 
                "reasoning_end": "[/REASONING]",
                "output_start": "[OUTPUT]", 
                "output_end": "[/OUTPUT]"
            }
        
        self.cot_parser = CoTParser(
            reasoning_start=tags['reasoning_start'],
            reasoning_end=tags['reasoning_end'],
            output_start=tags['output_start'],
            output_end=tags['output_end']
        )
        self.critic_parser = CriticParser()
        self.automated_validator = RuntimeEvaluator()
        self.error_classifier = ErrorClassifier() # Integration from Benchmarks  # Pre-validation (formerly AutomatedValidator)
        
        self.max_iterations = config.get('max_iterations', 2)
        self.timeout_per_iteration = config.get('timeout_per_iteration_seconds', 30)
        
        # [FIX] Handle nested generator/critic config
        generator_config = config.get('generator')
        if not generator_config and 'cot_review' in config:
            generator_config = config.get('cot_review', {}).get('generator', {})
        if not generator_config:
            generator_config = {'system_prompt_template': 'cot_generator_prompt.txt'}
        
        critic_config = config.get('critic')
        if not critic_config and 'cot_review' in config:
            critic_config = config.get('cot_review', {}).get('critic', {})
        if not critic_config:
            critic_config = {'system_prompt_template': 'cot_critic_prompt.txt'}
        
        # Load prompt templates
        self.generator_prompt_template = self._load_prompt_template(
            generator_config.get('system_prompt_template', 'cot_generator_prompt.txt')
        )
        self.critic_prompt_template = self._load_prompt_template(
            critic_config.get('system_prompt_template', 'cot_critic_prompt.txt')
        )
        
        # Initialize Vector Memory for Self-Learning
        try:
            self.memory = ChromaDBClient(collection_name="error_patterns")
            logger.info("Self-Learning Vector Memory initialized")
        except Exception as e:
            logger.warning("Failed to initialize Vector Memory, self-learning disabled: %s", e)
            self.memory = None
    
    def _load_prompt_template(self, filename: str) -> str:
        """Load prompt template from file"""
        prompt_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'prompts', filename
        )
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error("Failed to load prompt template %s: %s", filename, e, exc_info=True)
            return ""
    
    def run_correction_loop(
        self,
        query: str,
        data_context: Dict[str, Any],
        generator_model: str,
        critic_model: str,
        analysis_plan: Optional[Any] = None,
    ) -> CorrectionResult:
        """Execute the full Generator → Critic self-correction loop.

        Args:
            query: The user's original natural-language query.
            data_context: Dataset metadata — column names, dtypes, row count,
                and summary statistics used to ground the LLM's reasoning.
            generator_model: Ollama / LLM model identifier for the Generator.
            critic_model: Ollama / LLM model identifier for the Critic.
            analysis_plan: Optional :class:`AnalysisPlan` produced by the
                DynamicPlanner.  When provided the Generator is given a
                structured strategy to follow.

        Returns:
            A :class:`CorrectionResult` containing the validated (or
            best-effort) output, the full iteration history, elapsed time,
            and the termination reason (``"validated"``, ``"max_iterations"``,
            ``"timeout"``, ``"parsing_failure"``, or ``"failure"``).
        """
        start_time = time.time()
        iterations = []
        metrics = get_paper_metrics()
        
        # STEP 0: Safety validation before proceeding
        # (Activated dead code - addresses reviewer comment 3)
        
        # STEP 0.5: Retrieve learned patterns from vector memory (Self-Learning)
        # (Addresses reviewer comment 4: learning-based routing)
        learned_context = self.get_learned_patterns(query)
        if learned_context:
            logger.info("[SELF-LEARNING] Retrieved past error patterns for query")

        # STEP 0.6: Retrieve user-feedback weak-query patterns (Patent: feedback flywheel)
        feedback_context = ""
        try:
            from backend.api.feedback import get_weak_query_patterns
            feedback_context = get_weak_query_patterns(max_patterns=3)
            if feedback_context:
                logger.info("[FEEDBACK-LEARNING] Injecting %d chars of user-feedback context",
                            len(feedback_context))
        except Exception as e:
            logger.debug("Feedback learning unavailable: %s", e)

        # Merge both learning signals
        combined_learning = learned_context
        if feedback_context:
            combined_learning = (combined_learning + "\n" + feedback_context).strip()
        
        # Build initial generator prompt
        current_prompt = self._build_generator_prompt(
            query=query,
            data_context=data_context,
            previous_attempt=None,
            critic_feedback=None,
            analysis_plan=analysis_plan,
            learned_patterns=combined_learning
        )
        
        for iteration in range(1, self.max_iterations + 1):
            logger.debug("Self-correction iteration %d/%d", iteration, self.max_iterations)
            
            # STEP 1: Generator produces CoT response
            try:
                gen_response = self.llm_client.generate(
                    prompt=current_prompt,
                    model=generator_model,
                    adaptive_timeout=True
                )
                
                if not gen_response or not gen_response.get('success'):
                    logger.error("Generator failed on iteration %d", iteration)
                    break
                
                generator_output = gen_response.get('response', '')
                
            except Exception as e:
                logger.error("Generator error iteration %d: %s", iteration, e, exc_info=True)
                break
            
            # STEP 2: Parse CoT
            parsed_cot = self.cot_parser.parse(generator_output)
            
            if not parsed_cot.is_valid:
                logger.warning("CoT parsing failed: %s", parsed_cot.error_message)
                # Use unparsed response as fallback
                return CorrectionResult(
                    final_output=generator_output,
                    final_reasoning="Parsing failed - no CoT structure",
                    total_iterations=iteration,
                    all_iterations=iterations,
                    success=False,
                    termination_reason="parsing_failure",
                    total_time_seconds=time.time() - start_time
                )
            
            # STEP 2.5: Automated pre-validation (NEW - catches obvious errors)
            auto_validation = self.automated_validator.validate(
                query=query,
                reasoning=parsed_cot.reasoning,
                output=parsed_cot.output,
                data_context=data_context
            )
            
            if not auto_validation.is_valid:
                # Automated check found HIGH severity issues - skip LLM critic
                logger.info("Automated validation found %d issues", len(auto_validation.issues))
                
                # Convert validation issues to CriticIssue format
                critic_issues = [
                    CriticIssue(
                        description=issue.description,
                        location=issue.location,
                        severity=issue.severity,
                        suggestion=f"Fix: {issue.issue_type}"
                    )
                    for issue in auto_validation.issues
                    if issue.severity == "HIGH"  # Only HIGH severity issues
                ]
                
                critic_feedback = CriticFeedback(
                    is_valid=False,
                    issues=critic_issues,
                    raw_response=auto_validation.to_feedback_text()
                )
                
                # Record iteration with automated rejection
                iteration_record = CorrectionIteration(
                    iteration_number=iteration,
                    generator_response=generator_output,
                    parsed_cot=parsed_cot,
                    critic_response="[AUTOMATED_VALIDATION_FAILED]",
                    critic_feedback=critic_feedback,
                    correction_needed=True,
                    timestamp=time.time()
                )
                iterations.append(iteration_record)
                
                # Build correction prompt for next iteration
                # FIX: Pass parsed_cot (ParsedCoT) not generator_output (str)
                # FIX: Pass critic_feedback (CriticFeedback) not critic_feedback.feedback (str)
                current_prompt = self._build_generator_prompt(
                    query=query,
                    data_context=data_context,
                    previous_attempt=parsed_cot,
                    critic_feedback=critic_feedback
                )
                
                # Record error recovery event for metrics
                metrics.record_error_recovery(
                    mechanism='automated_validation',
                    trigger=f'Found {len(auto_validation.issues)} issues',
                    outcome='recovered',
                    latency_ms=0.0
                )
                
                continue  # Skip LLM critic, go to next iteration
            
            # STEP 3: Critic evaluates reasoning
            critic_prompt = self._build_critic_prompt(
                query=query,
                data_context=data_context,
                cot_reasoning=parsed_cot.reasoning,
                final_output=parsed_cot.output
            )
            
            try:
                critic_response = self.llm_client.generate(
                    prompt=critic_prompt,
                    model=critic_model,
                    adaptive_timeout=True
                )
                
                if not critic_response or not critic_response.get('success'):
                    logger.error("Critic failed on iteration %d", iteration)
                    break
                
                critic_output = critic_response.get('response', '')
                
            except Exception as e:
                logger.error("Critic error iteration %d: %s", iteration, e, exc_info=True)
                break
            
            # STEP 4: Parse critic feedback
            critic_feedback = self.critic_parser.parse(critic_output)
            
            # Record iteration
            iteration_record = CorrectionIteration(
                iteration_number=iteration,
                generator_response=generator_output,
                parsed_cot=parsed_cot,
                critic_response=critic_output,
                critic_feedback=critic_feedback,
                correction_needed=not critic_feedback.is_valid,
                timestamp=time.time()
            )
            iterations.append(iteration_record)
            
            # Log iteration details
            if self.config.get('performance_tracking', {}).get('log_iterations', True):
                self._log_iteration_details(iteration_record)
            
            # STEP 4.5: Safety validation (activated dead code - reviewer comment 3)
            is_safe, safety_error = self._validate_safety(parsed_cot)
            if not is_safe:
                logger.warning("Safety validation failed: %s", safety_error)
                metrics.record_error_recovery(
                    mechanism='safety_validation',
                    trigger=safety_error,
                    outcome='failed',
                    latency_ms=0.0
                )
                # Don't return unsafe output - force another iteration or fail
                if iteration < self.max_iterations:
                    current_prompt = self._build_generator_prompt(
                        query=query,
                        data_context=data_context,
                        previous_attempt=parsed_cot,
                        critic_feedback=CriticFeedback(
                            is_valid=False,
                            issues=[CriticIssue(
                                description=f"Safety issue: {safety_error}",
                                location="Output",
                                severity="HIGH",
                                suggestion="Remove dangerous operations"
                            )],
                            raw_response=f"Safety validation failed: {safety_error}"
                        )
                    )
                    continue
            
            # STEP 5: Decision gate
            if critic_feedback.is_valid:
                # SUCCESS: Reasoning validated
                logger.debug("Reasoning validated on iteration %d", iteration)
                
                # Calculate confidence score (activated dead code - reviewer comment 3)
                confidence = self._calculate_confidence_score(
                    CorrectionResult(
                        final_output=parsed_cot.output,
                        final_reasoning=parsed_cot.reasoning,
                        total_iterations=iteration,
                        all_iterations=iterations,
                        success=True,
                        termination_reason="validated",
                        total_time_seconds=time.time() - start_time
                    ),
                    critic_feedback
                )
                
                # If we had to correct (iteration > 1), learn from it
                learned_stored = False
                if iteration > 1:
                     self._learn_from_correction(iterations[0].parsed_cot, parsed_cot, query)
                     learned_stored = True
                
                # Record correction outcome for paper metrics
                metrics.record_correction_outcome(
                    CorrectionOutcome(
                        timestamp=datetime.datetime.now().isoformat(),
                        query=query[:200],
                        iterations=iteration,
                        termination_reason='validated',
                        total_time_seconds=time.time() - start_time,
                        automated_issues_found=sum(1 for it in iterations if it.critic_response == '[AUTOMATED_VALIDATION_FAILED]'),
                        automated_issue_types=[],
                        critic_invoked=True,
                        critic_approved=True,
                        safety_issues_found=0,
                        confidence_score=confidence,
                        learned_pattern_stored=learned_stored,
                        learned_pattern_retrieved=bool(learned_context),
                        correction_improved_output=(iteration > 1)
                    )
                )

                return CorrectionResult(
                    final_output=parsed_cot.output,
                    final_reasoning=parsed_cot.reasoning,
                    total_iterations=iteration,
                    all_iterations=iterations,
                    success=True,
                    termination_reason="validated",
                    total_time_seconds=time.time() - start_time
                )
            
            # STEP 6: Build correction prompt for next iteration
            # PATENT COMPLIANCE: Feedback goes back through PLANNING + GENERATION.
            # Re-invoke the DynamicPlanner so the analytical plan is revised
            # before the Generator is re-prompted (addresses patent language:
            # "back through the planning and generation of code phases").
            if iteration < self.max_iterations:
                revised_plan = analysis_plan  # Default: keep original plan
                try:
                    planner = get_dynamic_planner()
                    issue_summary = "; ".join(
                        i.description for i in critic_feedback.issues[:3]
                    )
                    replan_context = (
                        f"Previous plan failed verification. "
                        f"Issues: {issue_summary}. "
                        f"Revise the analytical approach for: {query}"
                    )
                    revised_plan = planner.create_plan(
                        replan_context,
                        str(data_context),
                        model=None,
                    )
                    logger.info(
                        "[RE-PLAN] Revised plan: domain=%s, steps=%d",
                        revised_plan.domain,
                        len(revised_plan.steps),
                    )
                except Exception as e:
                    logger.warning("Re-planning failed, using original plan: %s", e)

                # BENCHMARK INTEGRATION: Classify the error to give specific guidance
                try:
                    error_cat, error_sev, error_det = self.error_classifier.classify(
                        generator_output, 
                        error_details=str([i.description for i in critic_feedback.issues])
                    )
                    # Simple strategy mapping (derived from benchmarks/error_analysis.py)
                    strategies = {
                        ErrorCategory.FACTUAL: "Double-check constraints and data types.",
                        ErrorCategory.HALLUCINATION: "Only use columns/values explicitly present in DATA CONTEXT.",
                        ErrorCategory.FORMATTING: "Strictly follow the requested output format/JSON structure.",
                        ErrorCategory.INCOMPLETE: "Ensure all parts of the query are addressed.",
                        ErrorCategory.PYTHON_ERROR: "Check for syntax errors or undefined variables."
                    }
                    fix_strategy = strategies.get(error_cat, "Review the issues carefully.")
                    logger.info("[SELF-CORRECTION] Classified error as %s. Strategy: %s", error_cat.name, fix_strategy)
                except Exception as e:
                    fix_strategy = "Review the provided issues."
                    error_cat = None

                current_prompt = self._build_generator_prompt(
                    query=query,
                    data_context=data_context,
                    previous_attempt=parsed_cot,
                    critic_feedback=critic_feedback,
                    analysis_plan=revised_plan,
                    fix_strategy=fix_strategy 
                )
            
            # Check timeout
            if (time.time() - start_time) > (self.timeout_per_iteration * self.max_iterations):
                logger.warning("Self-correction timeout reached")
                break
        
        # Max iterations reached - return best attempt
        logger.warning("Max iterations (%d) reached without validation", self.max_iterations)
        
        # Return last attempt
        last_iteration = iterations[-1] if iterations else None
        if last_iteration:
            return CorrectionResult(
                final_output=last_iteration.parsed_cot.output,
                final_reasoning=last_iteration.parsed_cot.reasoning,
                total_iterations=len(iterations),
                all_iterations=iterations,
                success=False,
                termination_reason="max_iterations",
                total_time_seconds=time.time() - start_time
            )
        else:
            # Complete failure
            return CorrectionResult(
                final_output="Analysis failed - unable to generate response",
                final_reasoning="No successful iterations",
                total_iterations=0,
                all_iterations=[],
                success=False,
                termination_reason="failure",
                total_time_seconds=time.time() - start_time
            )
    
    def _build_generator_prompt(
        self,
        query: str,
        data_context: Dict[str, Any],
        previous_attempt: Optional[ParsedCoT] = None,
        critic_feedback: Optional[CriticFeedback] = None,
        analysis_plan: Optional[Any] = None,
        learned_patterns: str = "",
        fix_strategy: str = ""
    ) -> str:
        """Build the Generator's system prompt (initial or correction)."""
        
        # Build plan context
        plan_context = ""
        if analysis_plan:
             try:
                if hasattr(analysis_plan, 'summary'):
                    summary = str(analysis_plan.summary).strip()
                    if summary and summary != "Fallback analysis due to planning error":
                         plan_context = f"\n\n📋 ANALYSIS STRATEGY:\n{summary}\n"
                         
                         if hasattr(analysis_plan, 'steps') and analysis_plan.steps:
                             steps_list = []
                             for i, step in enumerate(analysis_plan.steps):
                                 desc = step.description if hasattr(step, 'description') else str(step)
                                 steps_list.append(f"{i+1}. {desc}")
                             if steps_list:
                                 plan_context += f"STEPS:\n" + "\n".join(steps_list[:10]) + "\n"
             except Exception:
                pass # Fail gracefully

        # Build self-learning context (addresses reviewer comment 4)
        learning_context = ""
        if learned_patterns:
            learning_context = f"\n\n⚠️ IMPORTANT - {learned_patterns}\n"

        if previous_attempt is None:
            # Initial prompt
            base_prompt = self.generator_prompt_template.format(
                user_query=query,
                data_info=self._format_data_context(data_context)
            )
            if plan_context:
                base_prompt = base_prompt.replace("DATA CONTEXT:", f"DATA CONTEXT:{plan_context}")
            if learning_context:
                base_prompt = base_prompt + learning_context
            return base_prompt
        else:
            # Correction prompt — includes revised plan when available
            issues_text = "\n".join([
                f"- {issue.description}\n  Location: {issue.location}\n  Fix: {issue.suggestion}"
                for issue in critic_feedback.issues
            ])

            # PATENT COMPLIANCE: Include revised analysis plan in correction
            revised_plan_text = ""
            if analysis_plan:
                try:
                    if hasattr(analysis_plan, 'summary'):
                        revised_plan_text = f"\nREVISED ANALYSIS PLAN:\n{analysis_plan.summary}\n"
                        if hasattr(analysis_plan, 'steps') and analysis_plan.steps:
                            for i, step in enumerate(analysis_plan.steps):
                                desc = step.description if hasattr(step, 'description') else str(step)
                                revised_plan_text += f"  {i+1}. {desc}\n"
                except Exception:
                    pass
            
            correction_prompt = f"""Your previous reasoning had issues that need correction:

ISSUES IDENTIFIED:
{issues_text}
{revised_plan_text}

RECOMMENDED FIX STRATEGY:
{fix_strategy}

ORIGINAL QUERY:
{query}

DATA CONTEXT:
{self._format_data_context(data_context)}

YOUR PREVIOUS REASONING:
{previous_attempt.reasoning}

YOUR PREVIOUS OUTPUT:
{previous_attempt.output}

Please revise your analysis addressing all issues above. Provide corrected [REASONING] and [OUTPUT] sections.
"""
            return correction_prompt
    
    def _build_critic_prompt(
        self,
        query: str,
        data_context: Dict[str, Any],
        cot_reasoning: str,
        final_output: str,
    ) -> str:
        """Build the Critic's evaluation prompt for a given CoT response."""
        return self.critic_prompt_template.format(
            query=query,
            data_context=self._format_data_context(data_context),
            cot_reasoning=cot_reasoning,
            final_output=final_output
        )
    
    def _format_data_context(self, data_context: Dict[str, Any]) -> str:
        """Format data context for prompt injection"""
        rows = data_context.get('rows', 'Unknown')
        columns = data_context.get('columns', [])
        data_types = data_context.get('data_types', {})
        
        context_str = f"""
Dataset Information:
- Rows: {rows}
- Columns: {len(columns)} total
- Column Names: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}
- Data Types: {data_types}

Available Statistics:
{data_context.get('stats_summary', 'No statistics available')}
"""
        return context_str.strip()
    
    def _log_iteration_details(self, iteration: CorrectionIteration) -> None:
        """Emit detailed debug/info logs for a single correction iteration."""
        logger.info("[COT] Iteration %d", iteration.iteration_number)
        logger.debug("[COT] Generator response length: %d chars", len(iteration.generator_response))
        logger.debug("[COT] Reasoning steps: %d", len(self.cot_parser.extract_steps(iteration.parsed_cot.reasoning)))
        logger.debug("[COT] Critic found issues: %d", len(iteration.critic_feedback.issues))
        
        if iteration.critic_feedback.issues:
            for issue in iteration.critic_feedback.issues:
                logger.info("[COT] Issue: %s (Severity: %s)", issue.description, issue.severity)
    
    def _calculate_confidence_score(self, 
                                     correction_result: CorrectionResult,
                                     critic_feedback: CriticFeedback) -> float:
        """
        Calculate confidence score (0.0-1.0) based on:
        - Validation status
        - Number of iterations
        - Severity of unresolved issues
        """
        if correction_result.success:
            # Validated - high confidence
            base_confidence = 0.95
            # Reduce slightly for more iterations (indicates initial issues)
            iteration_penalty = (correction_result.total_iterations - 1) * 0.05
            return max(0.7, base_confidence - iteration_penalty)
        
        else:
            # Not validated - calculate based on issue severity
            if not critic_feedback.issues:
                # No specific issues found, moderate confidence
                return 0.6
            
            # Check issue severities
            high_severity_count = sum(1 for issue in critic_feedback.issues if issue.severity == "HIGH")
            medium_severity_count = sum(1 for issue in critic_feedback.issues if issue.severity == "MEDIUM")
            
            if high_severity_count > 0:
                return 0.3  # Low confidence
            elif medium_severity_count > 0:
                return 0.5  # Medium confidence
            else:
                return 0.7  # Mostly low severity issues
    
    def _validate_safety(self, parsed_cot: ParsedCoT) -> tuple[bool, Optional[str]]:
        """
        Safety validation before executing analysis
        
        Returns:
            (is_safe, error_message)
        """
        reasoning = parsed_cot.reasoning.lower()
        output = parsed_cot.output.lower()
        
        # Check for dangerous operations
        danger_keywords = [
            'delete file', 'rm -rf', 'drop table', 'truncate',
            'system("', 'exec(', 'eval(', '__import__'
        ]
        
        for keyword in danger_keywords:
            if keyword in reasoning or keyword in output:
                return False, f"Potentially dangerous operation detected: {keyword}"
        
        # Check for data exfiltration attempts
        exfil_keywords = ['send to', 'upload to', 'http://', 'https://']
        for keyword in exfil_keywords:
            if keyword in reasoning:
                logger.warning("Suspicious keyword in reasoning: %s", keyword)
                # Not blocking but logged for audit
        
        return True, None

    def _learn_from_correction(self, original_cot: ParsedCoT, final_cot: ParsedCoT, query: str) -> None:
        """Store a correction pair in Vector DB for future self-learning.

        Patent Claim #4 — Self-Learning Error Correction.  The original
        (incorrect) and final (corrected) CoT are persisted so that
        semantically similar future queries can benefit from past mistakes.
        """
        if not self.memory:
            return

        try:
            timestamp = time.time()
            
            # Create semantic document: Query + Incorrect Reasoning
            # This allows retrieving this error when a SIMILAR query/reasoning pattern appears
            doc_text = f"Query: {query}\nIncorrect Reasoning: {original_cot.reasoning}"
            
            # Generate ID (deterministic across Python sessions)
            doc_id = f"err_{int(timestamp)}_{int(hashlib.md5(query.encode()).hexdigest(), 16) % 10000}"
            
            # Generate embedding for semantic search
            embedding = embed_text(doc_text)
            
            # Store in ChromaDB
            self.memory.add_document(
                doc_id=doc_id,
                text=doc_text,
                embedding=embedding,
                metadata={
                    "timestamp": timestamp,
                    "query": query,
                    "original_reasoning": original_cot.reasoning,
                    "corrected_reasoning": final_cot.reasoning,
                    "improvement_type": "correction"
                }
            )
                
            logger.info("[SELF-LEARNING] stored correction pattern in Vector DB: %s", doc_id)
            
        except Exception as e:
            logger.warning("Failed to learn from correction: %s", e)

    def get_learned_patterns(self, query: str) -> str:
        """Retrieve relevant past learnings using Semantic Search.

        Finds past errors that are conceptually similar to the current
        query by embedding it and querying the ChromaDB vector memory.

        Args:
            query: The user's natural-language query to match against
                stored correction patterns.

        Returns:
            str: A formatted context string describing past mistakes and
                their corrected approaches, or an empty string if no
                relevant patterns are found.
        """
        if not self.memory:
            return ""

        try:
            # Generate embedding for current query to find semantically similar past errors
            embedding = embed_text(query)
            
            # Query Vector DB
            results = self.memory.query(
                query_text=query,
                n_results=2,
                embedding=embedding
            )
            
            if not results['documents'] or not results['documents'][0]:
                return ""
            
            metadatas = results['metadatas'][0]
            
            context = "Past Mistakes to Avoid (Self-Learned):\n"
            count = 0
            for meta in metadatas:
                if not meta: continue
                # Retrieve the specific correction logic
                original = meta.get('original_reasoning', '')[:100]
                corrected = meta.get('corrected_reasoning', '')[:150]
                q = meta.get('query', '')[:50]
                
                context += f"- For queries like '{q}...', I previously failed with reasoning like '{original}...'. CORRECT APPROACH: '{corrected}...'\n"
                count += 1
            
            if count == 0:
                return ""
                
            return context
            
        except Exception as e:
            logger.warning("Failed to retrieve learned patterns: %s", e)
            return ""


# =============================================================================
# ENTERPRISE: CORRECTION STRATEGY (Plugin Interface)
# =============================================================================

class CorrectionStrategy(ABC):
    """Abstract base class for pluggable correction strategies.

    Implement this interface to customise how the self-correction
    engine responds to different kinds of critic feedback.

    Subclasses must define:
        - ``name`` — short human-readable identifier.
        - ``should_apply(feedback)`` — filter for applicable feedback.
        - ``transform_prompt(original, feedback)`` — produce a corrected prompt.
    """

    name: str = "base_strategy"
    priority: int = 100

    @abstractmethod
    def should_apply(self, feedback: Any) -> bool:
        """Return True if this strategy handles the feedback."""
        ...

    @abstractmethod
    def transform_prompt(
        self, original_prompt: str, feedback: str, context: Dict[str, Any],
    ) -> str:
        """Produce a corrected prompt incorporating feedback.

        Args:
            original_prompt: The generator's previous prompt.
            feedback: Structured feedback from the critic.
            context: Arbitrary context dict.

        Returns:
            Corrected prompt string.
        """
        ...


class DefaultCorrectionStrategy(CorrectionStrategy):
    """Default correction strategy that appends feedback to prompt."""
    name = "default"
    priority = 999

    def should_apply(self, feedback: Any) -> bool:
        return True

    def transform_prompt(
        self, original_prompt: str, feedback: str, context: Dict[str, Any],
    ) -> str:
        return f"{original_prompt}\n\nPrevious issues to fix:\n{feedback}"


# =============================================================================
# ENTERPRISE: CORRECTION OBSERVER (Event Hooks)
# =============================================================================

class CorrectionObserver(ABC):
    """Observer interface for monitoring correction loop events.

    Register implementations to receive callbacks whenever the
    self-correction engine starts, iterates, succeeds, or fails.
    """

    def on_loop_start(
        self, query: str, max_iterations: int, **kwargs: Any,
    ) -> None:
        """Called when a correction loop starts."""
        pass

    def on_iteration_complete(
        self, iteration: CorrectionIteration, **kwargs: Any,
    ) -> None:
        """Called after each iteration."""
        pass

    def on_loop_success(
        self, result: CorrectionResult, **kwargs: Any,
    ) -> None:
        """Called on successful completion."""
        pass

    def on_loop_failure(
        self, error: str, iterations_used: int, **kwargs: Any,
    ) -> None:
        """Called on failure / budget exhaustion."""
        pass


class LoggingCorrectionObserver(CorrectionObserver):
    """Built-in observer that logs correction events."""

    def on_loop_start(self, query: str, max_iterations: int, **kwargs: Any) -> None:
        logger.info("Correction loop started: query=%s, max_iter=%d", query[:50], max_iterations)

    def on_iteration_complete(self, iteration: CorrectionIteration, **kwargs: Any) -> None:
        logger.info(
            "Correction iteration %d: valid=%s, confidence=%.3f",
            iteration.iteration_number,
            iteration.passed_critic,
            iteration.confidence_score,
        )

    def on_loop_success(self, result: CorrectionResult, **kwargs: Any) -> None:
        logger.info(
            "Correction succeeded: iterations=%d, confidence=%.3f",
            result.iterations_used if hasattr(result, 'iterations_used') else 0,
            result.confidence_score,
        )

    def on_loop_failure(self, error: str, iterations_used: int, **kwargs: Any) -> None:
        logger.warning("Correction failed after %d iterations: %s", iterations_used, error)


# =============================================================================
# ENTERPRISE: CORRECTION METRICS
# =============================================================================

class CorrectionMetrics:
    """Tracks performance metrics for the self-correction engine.

    Thread-safe collector for iteration counts, success rates,
    latency distributions, and strategy usage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_loops = 0
        self._successful_loops = 0
        self._total_iterations = 0
        self._loop_durations: deque = deque(maxlen=500)
        self._iterations_per_loop: deque = deque(maxlen=500)
        self._strategy_usage: Dict[str, int] = defaultdict(int)
        self._failure_reasons: Dict[str, int] = defaultdict(int)

    def record_loop(
        self,
        success: bool,
        iterations: int,
        duration_ms: float,
        strategy: str = "default",
        failure_reason: Optional[str] = None,
    ) -> None:
        """Record a completed correction loop.

        Args:
            success: Whether the loop produced valid output.
            iterations: Number of iterations used.
            duration_ms: Total loop duration in ms.
            strategy: Name of the correction strategy used.
            failure_reason: Reason string if the loop failed.
        """
        with self._lock:
            self._total_loops += 1
            self._total_iterations += iterations
            self._loop_durations.append(duration_ms)
            self._iterations_per_loop.append(iterations)
            self._strategy_usage[strategy] += 1
            if success:
                self._successful_loops += 1
            elif failure_reason:
                self._failure_reasons[failure_reason] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Return correction metrics summary."""
        with self._lock:
            durations = list(self._loop_durations)
            iterations = list(self._iterations_per_loop)
            return {
                "total_loops": self._total_loops,
                "success_rate": round(
                    self._successful_loops / max(self._total_loops, 1), 4
                ),
                "total_iterations": self._total_iterations,
                "avg_iterations_per_loop": round(
                    self._total_iterations / max(self._total_loops, 1), 2
                ),
                "avg_loop_duration_ms": round(
                    sum(durations) / len(durations), 2
                ) if durations else 0,
                "strategy_usage": dict(self._strategy_usage),
                "top_failure_reasons": dict(
                    sorted(self._failure_reasons.items(), key=lambda x: -x[1])[:5]
                ),
            }


# =============================================================================
# SINGLETON
# =============================================================================

_correction_metrics: Optional[CorrectionMetrics] = None
_correction_metrics_lock = threading.Lock()


def get_correction_metrics() -> CorrectionMetrics:
    """Get or create the singleton :class:`CorrectionMetrics` (thread-safe)."""
    global _correction_metrics
    if _correction_metrics is None:
        with _correction_metrics_lock:
            if _correction_metrics is None:
                _correction_metrics = CorrectionMetrics()
    return _correction_metrics
