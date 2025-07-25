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

from .data_loading import enrich_log_entry_context, enrich_unsw_for_rag

logger = logging.getLogger(__name__)


class ReasoningEnhancer:
    """Enhanced reasoning techniques for anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.self_consistency_config = config.get('self_consistency', {})
        self.verifier_config = config.get('verifier_feedback', {})
        self.cross_validation_config = config.get('cross_validation_inference', {})
        
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
        
        # Use cross-validation inference if enabled, otherwise standard prediction
        if self.cross_validation_config.get('enabled', False):
            logger.info("Using cross-validation inference for initial prediction")
            standard_result = await self._cross_validation_prediction(
                backend, enriched_log_entry, dataset_type, model_name
            )
        else:
            # Standard prediction
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
            return enrich_log_entry_context(log_entry, template_mapping)
        elif dataset_type == "assuremoss":
            # AssureMOSS enrichment is handled in data_loading.py
            return enrich_log_entry_context(log_entry, None, dataset_type)
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
        
        elif dataset_type == "assuremoss":
            base_prompt = f"""You are a Kubernetes security expert analyzing NetFlow data from containerized microservices.

Kubernetes NetFlow data: {log_entry}

Analyze this container network traffic and determine if it represents an anomaly:

1. Consider Kubernetes service mesh patterns and pod-to-pod communication
2. Look for suspicious container behavior or lateral movement attempts
3. Check for abnormal data volumes or connection patterns between microservices
4. Identify potential container escape, privilege escalation, or data exfiltration indicators

Respond with:
- "ANOMALY" if the traffic shows malicious or suspicious Kubernetes behavior
- "NORMAL" if the traffic shows normal microservice communication
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
    
    async def _cross_validation_prediction(self, backend, log_entry: str, dataset_type: str, 
                                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate prediction using multi-perspective cross-validation"""
        
        # Get perspectives for this dataset type
        perspectives = self.cross_validation_config.get('perspectives', {}).get(dataset_type, [])
        if not perspectives:
            logger.warning(f"No perspectives configured for {dataset_type}, falling back to standard prediction")
            return await self._standard_prediction(backend, log_entry, dataset_type, model_name)
        
        temperature = self.cross_validation_config.get('temperature', 0.6)
        aggregation = self.cross_validation_config.get('aggregation', 'weighted_vote')
        
        # Collect predictions from each perspective
        perspective_results = []
        
        for perspective in perspectives:
            # Generate perspective-specific prompt
            prompt = self._generate_perspective_prompt(
                log_entry, dataset_type, perspective, model_name
            )
            
            # Get prediction from this perspective
            response = await backend.generate_text(
                prompt, 
                temperature=temperature,
                max_new_tokens=512,
                use_thinking=False
            )
            
            # Parse prediction
            prediction = "ANOMALY" if "ANOMALY" in response.text.upper() else "NORMAL"
            
            # Extract confidence if mentioned (simple heuristic)
            confidence = 0.7  # Default confidence
            if "high confidence" in response.text.lower():
                confidence = 0.9
            elif "low confidence" in response.text.lower():
                confidence = 0.5
            elif "uncertain" in response.text.lower():
                confidence = 0.3
            
            perspective_results.append({
                'perspective': perspective['name'],
                'weight': perspective.get('weight', 1.0),
                'focus': perspective.get('focus', ''),
                'prediction': prediction,
                'confidence': confidence,
                'explanation': response.text,
                'prompt': prompt
            })
        
        # Aggregate predictions based on method
        if aggregation == 'weighted_vote':
            final_result = self._weighted_vote_aggregation(perspective_results)
        elif aggregation == 'confidence_weighted':
            final_result = self._confidence_weighted_aggregation(perspective_results)
        else:  # majority_vote
            final_result = self._majority_vote_aggregation(perspective_results)
        
        # Add cross-validation details to result
        final_result['cross_validation_details'] = {
            'perspectives': perspective_results,
            'aggregation_method': aggregation
        }
        
        return final_result
    
    def _generate_perspective_prompt(self, log_entry: str, dataset_type: str, 
                                   perspective: Dict[str, Any], model_name: Optional[str] = None) -> str:
        """Generate prompt for a specific expert perspective"""
        
        expert_name = perspective['name'].replace('_', ' ').title()
        focus_areas = perspective.get('focus', '')
        
        if dataset_type in ["eventtraces", "bgl"]:
            system_type = "event traces" if dataset_type == "eventtraces" else "supercomputer system logs"
            prompt = f"""You are a {expert_name} analyzing {system_type}.
Your expertise focuses on: {focus_areas}

Log data: {log_entry}

From your specialized perspective, analyze this log data:

1. Apply your domain expertise to identify relevant patterns
2. Look for indicators specific to your area of focus
3. Determine if this represents an anomaly based on your experience

Respond with:
- "ANOMALY" if abnormal based on your expertise
- "NORMAL" if normal based on your expertise
- Brief explanation from your perspective
- Your confidence level (high/medium/low)

Be concise but specific to your expertise.
Your response:"""
            
        elif dataset_type == "assuremoss":
            prompt = f"""You are a {expert_name} analyzing Kubernetes NetFlow data.
Your expertise focuses on: {focus_areas}

Kubernetes NetFlow data: {log_entry}

From your specialized perspective, analyze this container network traffic:

1. Apply your domain expertise to identify relevant patterns
2. Look for indicators specific to your area of focus
3. Determine if this represents an anomaly based on your experience

Respond with:
- "ANOMALY" if suspicious based on your expertise
- "NORMAL" if normal based on your expertise
- Brief explanation from your perspective
- Your confidence level (high/medium/low)

Be concise but specific to your expertise.
Your response:"""
            
        else:  # unsw-nb15
            prompt = f"""You are a {expert_name} analyzing network traffic data.
Your expertise focuses on: {focus_areas}

Network data: {log_entry}

From your specialized perspective, analyze this network traffic:

1. Apply your domain expertise to identify relevant patterns
2. Look for indicators specific to your area of focus
3. Determine if this represents an anomaly based on your experience

Respond with:
- "ANOMALY" if malicious based on your expertise
- "NORMAL" if normal based on your expertise
- Brief explanation from your perspective
- Your confidence level (high/medium/low)

Be concise but specific to your expertise.
Your response:"""
        
        return prompt
    
    def _weighted_vote_aggregation(self, perspective_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate predictions using weighted voting"""
        
        weighted_scores = {'NORMAL': 0.0, 'ANOMALY': 0.0}
        total_weight = 0.0
        
        for result in perspective_results:
            weight = result['weight']
            prediction = result['prediction']
            weighted_scores[prediction] += weight
            total_weight += weight
        
        # Normalize scores
        for pred in weighted_scores:
            weighted_scores[pred] /= total_weight if total_weight > 0 else 1.0
        
        # Determine winner
        final_prediction = max(weighted_scores.items(), key=lambda x: x[1])[0]
        confidence = weighted_scores[final_prediction]
        
        # Create explanation
        explanations = []
        for result in perspective_results:
            if result['prediction'] == final_prediction:
                explanations.append(f"{result['perspective']}: {result['explanation'][:100]}...")
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'explanation': f"Cross-validation consensus ({final_prediction} with {confidence:.2f} weighted score). " + 
                          f"Key perspectives: {'; '.join(explanations[:2])}",
            'weighted_scores': weighted_scores
        }
    
    def _confidence_weighted_aggregation(self, perspective_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate predictions weighted by confidence and perspective weight"""
        
        weighted_scores = {'NORMAL': 0.0, 'ANOMALY': 0.0}
        total_weight = 0.0
        
        for result in perspective_results:
            weight = result['weight'] * result['confidence']
            prediction = result['prediction']
            weighted_scores[prediction] += weight
            total_weight += weight
        
        # Normalize scores
        for pred in weighted_scores:
            weighted_scores[pred] /= total_weight if total_weight > 0 else 1.0
        
        # Determine winner
        final_prediction = max(weighted_scores.items(), key=lambda x: x[1])[0]
        confidence = weighted_scores[final_prediction]
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'explanation': f"Confidence-weighted cross-validation: {final_prediction} ({confidence:.2f})",
            'weighted_scores': weighted_scores
        }
    
    def _majority_vote_aggregation(self, perspective_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple majority vote aggregation"""
        
        predictions = [r['prediction'] for r in perspective_results]
        prediction_counts = Counter(predictions)
        
        final_prediction = prediction_counts.most_common(1)[0][0]
        confidence = prediction_counts[final_prediction] / len(predictions)
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'explanation': f"Majority vote: {final_prediction} ({prediction_counts[final_prediction]}/{len(predictions)} experts)",
            'prediction_counts': dict(prediction_counts)
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
        elif dataset_type == "assuremoss":
            variations = [
                "You are a Kubernetes security expert analyzing containerized microservice NetFlows.",
                "You are a cloud-native security analyst investigating Kubernetes cluster anomalies.",
                "You are a DevSecOps engineer monitoring container network patterns.",
                "You are a container security specialist examining pod-to-pod communications.",
                "You are a cloud security architect analyzing microservice mesh traffic."
            ]
            
            reasoning_styles = [
                "Think step by step about the Kubernetes network behavior:",
                "Consider the following container security aspects:",
                "Analyze this microservice traffic systematically:",
                "Examine the pod communication patterns carefully:",
                "Investigate the following Kubernetes indicators:"
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
        elif dataset_type == "assuremoss":
            base_prompt = f"""{role}

Kubernetes NetFlow data: {log_entry}

{reasoning_style}
1. What type of Kubernetes network traffic is this?
2. Are there any suspicious container or pod behaviors?
3. Does this follow normal microservice communication patterns?
4. What could indicate container compromise or lateral movement?

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
            context = "system behavior patterns"
        elif dataset_type == "assuremoss":
            domain = "Kubernetes security"
            context = "container network traffic patterns"
        else:  # unsw-nb15
            domain = "network security"
            context = "network traffic characteristics"
        
        # Detect DeepSeek models for special handling
        is_deepseek = model_name and "deepseek" in model_name.lower()
        
        base_prompt = f"""You are a senior {domain} expert providing peer review of an anomaly detection analysis.

Original Data: {log_entry}

Analysis Under Review:
- Prediction: {initial_prediction['prediction']}
- Reasoning: {initial_prediction.get('explanation', 'No explanation provided')}

As a peer reviewer, evaluate whether:
1. The analysis correctly interprets the {context}
2. The reasoning follows logically from the observed data
3. Alternative interpretations should be considered

Provide your verification decision:
- CONFIRM: The analysis is well-reasoned and the conclusion follows from the data
- REJECT: The analysis contains significant errors or misinterpretations
- UNCERTAIN: The data supports multiple valid interpretations

Be concise.

Decision and brief justification:"""

        # Backend handles DeepSeek thinking pattern
        return base_prompt
    
    async def _generate_refined_prediction(self, backend, log_entry: str, dataset_type: str,
                                          initial_prediction: Dict[str, Any], 
                                          verifier_feedback: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate refined prediction based on verifier feedback"""
        
        # Detect DeepSeek models for special handling
        is_deepseek = model_name and "deepseek" in model_name.lower()
        
        base_prompt = f"""You are refining an anomaly detection analysis based on peer review feedback.

Original Data: {log_entry}

Your Initial Analysis:
- Prediction: {initial_prediction['prediction']}
- Reasoning: {initial_prediction.get('explanation', '')}

Peer Review Feedback:
{verifier_feedback}

Provide a refined analysis that:
1. Addresses specific points raised in the review
2. Reconsiders the data with the feedback in mind
3. Provides a well-justified final conclusion

Be concise.

Final Classification: NORMAL or ANOMALY
Refined Explanation: Your improved reasoning incorporating the feedback

Your refined analysis:"""
        
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