"""
Self-Correction Engine
Manages the Generator → Critic → Feedback loop
"""
import logging
import time
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from .cot_parser import CoTParser, CriticParser, ParsedCoT, CriticFeedback

@dataclass
class CorrectionIteration:
    """Single iteration in the correction loop"""
    iteration_number: int
    generator_response: str
    parsed_cot: ParsedCoT
    critic_response: str
    critic_feedback: CriticFeedback
    correction_needed: bool
    timestamp: float

@dataclass
class CorrectionResult:
    """Final result after correction loop"""
    final_output: str
    final_reasoning: str
    total_iterations: int
    all_iterations: List[CorrectionIteration]
    success: bool
    termination_reason: str  # "validated", "max_iterations", "timeout", "parsing_failure", "failure"
    total_time_seconds: float

class SelfCorrectionEngine:
    """
    Implements the Generator → Critic → Feedback self-correction loop
    """
    
    def __init__(self, config: Dict[str, Any], llm_client):
        self.config = config
        self.llm_client = llm_client
        self.cot_parser = CoTParser(
            reasoning_start=config['tags']['reasoning_start'],
            reasoning_end=config['tags']['reasoning_end'],
            output_start=config['tags']['output_start'],
            output_end=config['tags']['output_end']
        )
        self.critic_parser = CriticParser()
        
        self.max_iterations = config['max_iterations']
        self.timeout_per_iteration = config['timeout_per_iteration_seconds']
        
        # Load prompt templates
        self.generator_prompt_template = self._load_prompt_template(
            config['generator']['system_prompt_template']
        )
        self.critic_prompt_template = self._load_prompt_template(
            config['critic']['system_prompt_template']
        )
    
    def _load_prompt_template(self, filename: str) -> str:
        """Load prompt template from file"""
        prompt_path = os.path.join(
            os.path.dirname(__file__), '..', 'prompts', filename
        )
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to load prompt template {filename}: {e}")
            return ""
    
    def run_correction_loop(self, 
                           query: str, 
                           data_context: Dict[str, Any],
                           generator_model: str,
                           critic_model: str) -> CorrectionResult:
        """
        Execute the full self-correction loop
        
        Args:
            query: User's original query
            data_context: Data information (structure, stats, etc.)
            generator_model: Model name for generation
            critic_model: Model name for critique
            
        Returns:
            CorrectionResult with final validated output
        """
        start_time = time.time()
        iterations = []
        
        # Build initial generator prompt
        current_prompt = self._build_generator_prompt(
            query=query,
            data_context=data_context,
            previous_attempt=None,
            critic_feedback=None
        )
        
        for iteration in range(1, self.max_iterations + 1):
            logging.debug(f"Self-correction iteration {iteration}/{self.max_iterations}")
            
            # STEP 1: Generator produces CoT response
            try:
                gen_response = self.llm_client.generate(
                    prompt=current_prompt,
                    model=generator_model,
                    adaptive_timeout=True
                )
                
                if not gen_response or not gen_response.get('success'):
                    logging.error(f"Generator failed on iteration {iteration}")
                    break
                
                generator_output = gen_response.get('response', '')
                
            except Exception as e:
                logging.error(f"Generator error iteration {iteration}: {e}")
                break
            
            # STEP 2: Parse CoT
            parsed_cot = self.cot_parser.parse(generator_output)
            
            if not parsed_cot.is_valid:
                logging.warning(f"CoT parsing failed: {parsed_cot.error_message}")
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
                    logging.error(f"Critic failed on iteration {iteration}")
                    break
                
                critic_output = critic_response.get('response', '')
                
            except Exception as e:
                logging.error(f"Critic error iteration {iteration}: {e}")
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
            
            # STEP 5: Decision gate
            if critic_feedback.is_valid:
                # SUCCESS: Reasoning validated
                logging.debug(f"Reasoning validated on iteration {iteration}")
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
            if iteration < self.max_iterations:
                current_prompt = self._build_generator_prompt(
                    query=query,
                    data_context=data_context,
                    previous_attempt=parsed_cot,
                    critic_feedback=critic_feedback
                )
            
            # Check timeout
            if (time.time() - start_time) > (self.timeout_per_iteration * self.max_iterations):
                logging.warning("Self-correction timeout reached")
                break
        
        # Max iterations reached - return best attempt
        logging.warning(f"Max iterations ({self.max_iterations}) reached without validation")
        
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
    
    def _build_generator_prompt(self, 
                               query: str,
                               data_context: Dict[str, Any],
                               previous_attempt: Optional[ParsedCoT] = None,
                               critic_feedback: Optional[CriticFeedback] = None) -> str:
        """Build generator prompt (initial or correction)"""
        if previous_attempt is None:
            # Initial prompt
            return self.generator_prompt_template.format(
                user_query=query,
                data_info=self._format_data_context(data_context)
            )
        else:
            # Correction prompt
            issues_text = "\n".join([
                f"- {issue.description}\n  Location: {issue.location}\n  Fix: {issue.suggestion}"
                for issue in critic_feedback.issues
            ])
            
            correction_prompt = f"""Your previous reasoning had issues that need correction:

ISSUES IDENTIFIED:
{issues_text}

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
    
    def _build_critic_prompt(self,
                            query: str,
                            data_context: Dict[str, Any],
                            cot_reasoning: str,
                            final_output: str) -> str:
        """Build critic evaluation prompt"""
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
    
    def _log_iteration_details(self, iteration: CorrectionIteration):
        """Detailed logging for debugging"""
        logging.info(f"[COT] Iteration {iteration.iteration_number}")
        logging.debug(f"[COT] Generator response length: {len(iteration.generator_response)} chars")
        logging.debug(f"[COT] Reasoning steps: {len(self.cot_parser.extract_steps(iteration.parsed_cot.reasoning))}")
        logging.debug(f"[COT] Critic found issues: {len(iteration.critic_feedback.issues)}")
        
        if iteration.critic_feedback.issues:
            for issue in iteration.critic_feedback.issues:
                logging.info(f"[COT] Issue: {issue.description} (Severity: {issue.severity})")
    
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
    
    def _validate_safety(self, parsed_cot: ParsedCoT) -> tuple:
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
                logging.warning(f"Suspicious keyword in reasoning: {keyword}")
                # Not blocking but logged for audit
        
        return True, None
