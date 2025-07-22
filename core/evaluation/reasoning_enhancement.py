"""
Reasoning Enhancement Module for Log Anomaly Detection
Implements self-consistency and verifier feedback techniques to improve SLM performance on AD tasks.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import statistics
from pathlib import Path

from .data_loading import enrich_text_for_rag, enrich_unsw_for_rag

logger = logging.getLogger(__name__)


class ReasoningEnhancer:
    """Enhanced reasoning techniques for anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.self_consistency_config = config.get('self_consistency', {})
        self.verifier_config = config.get('verifier_feedback', {})
        
    async def enhance_prediction(self, backend, log_entry: str, dataset_type: str, 
                               template_mapping: Optional[Dict[str, str]] = None,
                               unsw_features: Optional[Dict[str, str]] = None,
                               model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply reasoning enhancements to a single log entry prediction
        
        Args:
            backend: The inference backend
            log_entry: The log entry to analyze
            dataset_type: Type of dataset (eventtraces, unsw-nb15)
            template_mapping: Event template mapping for enrichment
            unsw_features: UNSW feature mapping for enrichment
            model_name: Name of the model for model-specific handling
            
        Returns:
            Enhanced prediction with reasoning details
        """
        # Enrich log entry based on dataset type
        enriched_log_entry = self._enrich_log_entry(log_entry, dataset_type, template_mapping, unsw_features)
        
        # Standard prediction first
        standard_result = await self._standard_prediction(backend, enriched_log_entry, dataset_type, model_name)
        
        enhanced_result = {
            'log_entry': log_entry,
            'standard_prediction': standard_result,
            'enhancements': {}
        }
        
        # Apply self-consistency if enabled
        if self.self_consistency_config.get('enabled', False):
            logger.info("Applying self-consistency enhancement")
            consistency_result = await self._apply_self_consistency(
                backend, enriched_log_entry, dataset_type, standard_result, model_name
            )
            enhanced_result['enhancements']['self_consistency'] = consistency_result
            
        # Apply verifier feedback if enabled
        if self.verifier_config.get('enabled', False):
            logger.info("Applying verifier feedback enhancement")
            
            # Use self-consistency result as input if available, otherwise standard
            input_prediction = (
                enhanced_result['enhancements'].get('self_consistency', {}).get('final_prediction')
                or standard_result
            )
            
            verifier_result = await self._apply_verifier_feedback(
                backend, enriched_log_entry, dataset_type, input_prediction, model_name
            )
            enhanced_result['enhancements']['verifier_feedback'] = verifier_result
            
        # Determine final prediction
        enhanced_result['final_prediction'] = self._get_final_prediction(enhanced_result)
        
        return enhanced_result
    
    def _enrich_log_entry(self, log_entry: str, dataset_type: str, 
                         template_mapping: Optional[Dict[str, str]] = None,
                         unsw_features: Optional[Dict[str, str]] = None) -> str:
        """Enrich log entry with templates for better reasoning"""
        
        if dataset_type == "unsw-nb15" and unsw_features:
            return enrich_unsw_for_rag(log_entry, unsw_features)
        elif dataset_type == "eventtraces" and template_mapping:
            return enrich_text_for_rag(log_entry, template_mapping)
        else:
            return log_entry
    
    async def _standard_prediction(self, backend, log_entry: str, dataset_type: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate standard anomaly detection prediction"""
        
        # Detect DeepSeek models for special handling
        is_deepseek = model_name and "deepseek" in model_name.lower()
        
        if dataset_type in ["eventtraces", "bgl"]:
            system_type = "event traces" if dataset_type == "eventtraces" else "supercomputer system logs"
            base_prompt = f"""You are a distributed systems expert analyzing {system_type}.

Log data: {log_entry}

Analyze this log data and determine if it represents an anomaly:

1. Consider the sequence of events or system messages
2. Look for unusual patterns or violations of normal system behavior
3. Identify any suspicious system conditions or failures

Respond with:
- "ANOMALY" if the log shows abnormal system behavior
- "NORMAL" if the log shows normal system behavior
- Provide a brief explanation of your reasoning

Be concise.
Your response:"""
        
        else:  # unsw-nb15
            base_prompt = f"""You are a network security expert analyzing network traffic data.

Network data: {log_entry}

Analyze this network traffic and determine if it represents an anomaly:

1. Look for suspicious network patterns
2. Check for signs of attacks or malicious activity
3. Consider normal vs abnormal traffic characteristics

Respond with:
- "ANOMALY" if the traffic shows malicious or suspicious behavior
- "NORMAL" if the traffic shows normal network behavior
- Provide a brief explanation of your reasoning

Be concise.
Your response:"""

        # Use standard prompt (backend handles DeepSeek thinking pattern)
        prompt = base_prompt
        
        response = await backend.generate_text(prompt, max_new_tokens=512, use_thinking=False)
        
        # Parse response
        prediction = "ANOMALY" if "ANOMALY" in response.text.upper() else "NORMAL"
        
        return {
            'prediction': prediction,
            'explanation': response.text,
            'confidence': 0.5,  # Default
            'prompt': prompt
        }
    
    async def _apply_self_consistency(self, backend, log_entry: str, dataset_type: str, 
                                     standard_result: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Apply self-consistency reasoning enhancement"""
        
        num_samples = self.self_consistency_config.get('num_samples', 5)
        temperature = self.self_consistency_config.get('temperature', 0.7)
        
        predictions = []
        explanations = []
        prompts = []
        
        for i in range(num_samples):
            # Generate varied prompts for different reasoning paths
            varied_prompt = self._generate_varied_prompt(log_entry, dataset_type, i, model_name)
            
            # Generate prediction with higher temperature for diversity (no thinking for self-consistency)
            response = await backend.generate_text(
                varied_prompt, 
                temperature=temperature,
                max_new_tokens=512,
                use_thinking=False
            )
            
            # Parse prediction
            prediction = "ANOMALY" if "ANOMALY" in response.text.upper() else "NORMAL"
            
            predictions.append(prediction)
            explanations.append(response.text)
            prompts.append(varied_prompt)
        
        # Calculate consistency metrics
        prediction_counts = Counter(predictions)
        majority_prediction = prediction_counts.most_common(1)[0][0]
        consistency_score = prediction_counts[majority_prediction] / len(predictions)
        
        return {
            'individual_predictions': predictions,
            'individual_explanations': explanations,
            'individual_prompts': prompts,
            'final_prediction': {
                'prediction': majority_prediction,
                'confidence': consistency_score,
                'explanation': f"Majority vote: {majority_prediction} ({prediction_counts[majority_prediction]}/{len(predictions)} samples)"
            },
            'consistency_score': consistency_score,
            'prediction_distribution': dict(prediction_counts)
        }
    
    async def _apply_verifier_feedback(self, backend, log_entry: str, dataset_type: str,
                                      initial_prediction: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Apply verifier feedback reasoning enhancement"""
        
        verifier_temperature = self.verifier_config.get('verifier_temperature', 0.3)
        
        # Generate verifier prompt
        verifier_prompt = self._generate_verifier_prompt(
            log_entry, dataset_type, initial_prediction, model_name
        )
        
        # Get verifier response (WITH thinking for critical review)
        verifier_response = await backend.generate_text(
            verifier_prompt, 
            temperature=verifier_temperature,
            max_new_tokens=1024,
            use_thinking=True
        )
        
        # Parse verifier decision
        verifier_text = verifier_response.text.upper()
        
        if "REJECT" in verifier_text:
            verification_decision = "REJECT"
        elif "CONFIRM" in verifier_text:
            verification_decision = "CONFIRM"
        else:
            verification_decision = "UNCERTAIN"
        
        # Handle verifier decision
        if verification_decision == "REJECT":
            # Generate refined prediction based on verifier feedback
            refined_prediction = await self._generate_refined_prediction(
                backend, log_entry, dataset_type, initial_prediction, verifier_response.text, model_name
            )
            final_prediction = refined_prediction
        else:
            # Enhance confidence based on verification
            confidence_boost = 0.2 if verification_decision == "CONFIRM" else 0.0
            final_prediction = {
                'prediction': initial_prediction['prediction'],
                'confidence': min(1.0, initial_prediction.get('confidence', 0.5) + confidence_boost),
                'explanation': f"Verified: {initial_prediction.get('explanation', '')}"
            }
        
        return {
            'verifier_prompt': verifier_prompt,
            'verifier_response': verifier_response.text,
            'verification_decision': verification_decision,
            'initial_prediction': initial_prediction,
            'final_prediction': final_prediction
        }
    
    def _generate_varied_prompt(self, log_entry: str, dataset_type: str, sample_idx: int, model_name: Optional[str] = None) -> str:
        """Generate varied prompts for self-consistency"""
        
        if dataset_type in ["eventtraces", "bgl"]:
            if dataset_type == "eventtraces":
                variations = [
                    "You are a distributed systems expert analyzing log event traces.",
                    "You are a system administrator investigating potential anomalies in event logs.",
                    "You are a cybersecurity analyst examining system event sequences.",
                    "You are a software engineer debugging distributed system behavior.",
                    "You are a DevOps specialist monitoring system health through logs."
                ]
            else:  # bgl
                variations = [
                    "You are a distributed systems expert analyzing supercomputer system logs.",
                    "You are an HPC system administrator investigating potential system anomalies.",
                    "You are a supercomputing specialist examining system behavior logs.",
                    "You are a systems engineer debugging high-performance computing infrastructure.",
                    "You are an operations specialist monitoring supercomputer system health."
                ]
            
            reasoning_styles = [
                "Think step by step about the system behavior:",
                "Consider the following aspects:",
                "Analyze this systematically:",
                "Examine the logs carefully:",
                "Investigate the following:"
            ]
        else:  # unsw-nb15
            variations = [
                "You are a network security expert analyzing network traffic data.",
                "You are a cybersecurity analyst investigating network anomalies.",
                "You are a network administrator monitoring traffic patterns.",
                "You are a security engineer examining network behavior.",
                "You are a threat hunter analyzing network communications."
            ]
            
            reasoning_styles = [
                "Think step by step about the network traffic:",
                "Consider the following network aspects:",
                "Analyze this traffic systematically:",
                "Examine the network data carefully:",
                "Investigate the following network indicators:"
            ]
        
        role = variations[sample_idx % len(variations)]
        reasoning_style = reasoning_styles[sample_idx % len(reasoning_styles)]
        
        # Detect DeepSeek models for special handling
        is_deepseek = model_name and "deepseek" in model_name.lower()
        
        if dataset_type in ["eventtraces", "bgl"]:
            data_label = "Event trace" if dataset_type == "eventtraces" else "System log"
            base_prompt = f"""{role}

{data_label}: {log_entry}

{reasoning_style}
1. What type of system behavior is this?
2. Are there any unusual patterns or indicators?
3. Does this follow normal distributed system behavior?
4. What could indicate an anomaly?

Final classification: NORMAL or ANOMALY
Explanation: Provide your reasoning

Be concise.
Your response:"""
            
            # Backend handles DeepSeek thinking pattern
            return base_prompt
        else:
            base_prompt = f"""{role}

Network data: {log_entry}

{reasoning_style}
1. What type of network traffic is this?
2. Are there any suspicious network indicators?
3. Does this follow normal network behavior?
4. What could indicate malicious activity?

Final classification: NORMAL or ANOMALY
Explanation: Provide your reasoning

Be concise.
Your response:"""
            
            # Backend handles DeepSeek thinking pattern
            return base_prompt
    
    def _generate_verifier_prompt(self, log_entry: str, dataset_type: str, 
                                 initial_prediction: Dict[str, Any], model_name: Optional[str] = None) -> str:
        """Generate verifier prompt for feedback"""
        
        if dataset_type in ["eventtraces", "bgl"]:
            domain = "distributed systems"
        else:  # unsw-nb15
            domain = "network security"
        
        # Detect DeepSeek models for special handling
        is_deepseek = model_name and "deepseek" in model_name.lower()
        
        base_prompt = f"""You are a senior {domain} expert conducting a critical review of an anomaly detection analysis.

Original Data: {log_entry}
Initial Prediction: {initial_prediction['prediction']}
Initial Reasoning: {initial_prediction.get('explanation', 'No explanation provided')}

Review this analysis using available evidence. Only use:
- CONFIRM: If reasoning is sound for the available evidence
- REJECT: If there are clear logical flaws or contradictions
- UNCERTAIN: Only if genuinely ambiguous with conflicting evidence

Don't demand perfect evidence - work with what's available.

Critical Questions:
- Is the reasoning logical given the available data?
- Are there clear errors in the analysis?
- Does the conclusion reasonably follow from the evidence?

Final Verification Decision: CONFIRM, REJECT, or UNCERTAIN

Be concise.
Provide your critical assessment and final verification decision:"""

        # Backend handles DeepSeek thinking pattern
        return base_prompt
    
    async def _generate_refined_prediction(self, backend, log_entry: str, dataset_type: str,
                                          initial_prediction: Dict[str, Any], 
                                          verifier_feedback: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate refined prediction based on verifier feedback"""
        
        # Detect DeepSeek models for special handling
        is_deepseek = model_name and "deepseek" in model_name.lower()
        
        base_prompt = f"""Based on the critical feedback below, provide a refined analysis of this data:

Original Data: {log_entry}
Initial Prediction: {initial_prediction['prediction']}
Initial Reasoning: {initial_prediction.get('explanation', '')}

Critical Feedback:
{verifier_feedback}

Considering the feedback, provide a refined analysis:

1. Address the specific concerns raised in the feedback
2. Incorporate any overlooked evidence or alternative perspectives
3. Provide a more robust conclusion

Refined Classification: NORMAL or ANOMALY
Improved Explanation: Your enhanced reasoning

Be concise.
Your response:"""
        
        # Backend handles DeepSeek thinking pattern
        refinement_prompt = base_prompt
        
        refined_response = await backend.generate_text(refinement_prompt, max_new_tokens=1024, use_thinking=True)
        
        # Parse refined response
        refined_prediction = "ANOMALY" if "ANOMALY" in refined_response.text.upper() else "NORMAL"
        
        return {
            'prediction': refined_prediction,
            'confidence': 0.7,  # Higher confidence for refined prediction
            'explanation': refined_response.text,
            'refinement_prompt': refinement_prompt
        }
    
    def _get_final_prediction(self, enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the final prediction from all enhancements"""
        
        # Priority order: 
        # ... verifier_feedback > self_consistency > standard
        if 'verifier_feedback' in enhanced_result['enhancements']:
            return enhanced_result['enhancements']['verifier_feedback']['final_prediction']
        elif 'self_consistency' in enhanced_result['enhancements']:
            return enhanced_result['enhancements']['self_consistency']['final_prediction']
        else:
            return enhanced_result['standard_prediction']