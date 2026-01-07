"""
Confidence Tracking Module
Quantifies prediction uncertainty by running multiple trials
"""
import numpy as np
from collections import defaultdict


class ConfidenceTracker:
    """
    Tracks confidence in predictions by measuring consistency across multiple trials
    """
    
    def __init__(self, num_trials=10):
        """
        num_trials: How many times to run prediction to measure variance
        Higher = more accurate confidence but slower (10-20 recommended)
        """
        self.num_trials = num_trials
    
    def predict_with_confidence(self, predictor, **predict_kwargs):
        """
        Run prediction multiple times and return probabilities + confidence
        
        Args:
            predictor: The predictor object (hybrid or GPU)
            **predict_kwargs: Arguments to pass to predict_next_category()
        
        Returns:
            dict with format:
            {
                'Travel': {
                    'probability': 0.70,
                    'confidence': 0.95,
                    'std': 0.012,
                    'trials': [0.72, 0.69, 0.71, ...]
                },
                ...
            }
        """
        # Run prediction multiple times
        trial_results = []
        
        for trial in range(self.num_trials):
            prediction = predictor.predict_next_category(**predict_kwargs)
            trial_results.append(prediction)
        
        # Aggregate results per category
        categories = trial_results[0].keys()
        
        confidence_results = {}
        
        for category in categories:
            # Collect probabilities across trials
            probs = [trial[category] for trial in trial_results]
            
            # Calculate statistics
            mean_prob = np.mean(probs)
            std_prob = np.std(probs)
            
            # Calculate confidence
            # Method 1: Coefficient of Variation (CV) based
            if mean_prob > 0:
                cv = std_prob / mean_prob  # Lower CV = higher confidence
                confidence = 1.0 / (1.0 + cv)  # Scale to 0-1
            else:
                confidence = 0.0
            
            # Clamp confidence to reasonable range
            confidence = np.clip(confidence, 0.0, 1.0)
            
            confidence_results[category] = {
                'probability': mean_prob,
                'confidence': confidence,
                'std': std_prob,
                'trials': probs
            }
        
        return confidence_results
    
    def predict_with_evidence(self, predictor, knowledge_graph, **predict_kwargs):
        """
        Enhanced version that also tracks EVIDENCE supporting the prediction
        
        Returns prediction with:
        - probability
        - confidence (from variance)
        - evidence_count (historical support)
        """
        # Get base confidence prediction
        confidence_results = self.predict_with_confidence(predictor, **predict_kwargs)
        
        # Add evidence counts
        for category in confidence_results.keys():
            # Count historical evidence
            evidence_count = 0
            
            # Evidence 1: How many searches in this category?
            category_searches = sum(
                1 for search in knowledge_graph.search_history
                if category in search.get('categories', {})
            )
            evidence_count += category_searches
            
            # Evidence 2: How many entities connected to this category?
            if category in knowledge_graph.G:
                entity_connections = sum(
                    1 for neighbor in knowledge_graph.G.predecessors(category)
                    if knowledge_graph.G.nodes[neighbor].get('node_type') == 'entity'
                )
                evidence_count += entity_connections
            
            # Evidence 3: Strength of USER connection
            if knowledge_graph.G.has_edge("USER", category):
                user_connection_strength = knowledge_graph.G["USER"][category]['weight']
                evidence_count += int(user_connection_strength * 100)
            
            confidence_results[category]['evidence_count'] = evidence_count
        
        return confidence_results
    
    def get_confidence_summary(self, confidence_results):
        """
        Get human-readable summary of confidence
        
        Returns:
            {
                'most_confident': ('Travel', 0.95),
                'least_confident': ('Fashion', 0.42),
                'high_confidence_categories': ['Travel', 'Daily_Life'],
                'low_confidence_categories': ['Fashion', 'News_Politics'],
                'overall_confidence': 0.73
            }
        """
        # Sort by confidence
        sorted_by_conf = sorted(
            confidence_results.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        # Calculate overall confidence (weighted by probability)
        total_prob = sum(r['probability'] for r in confidence_results.values())
        if total_prob > 0:
            overall_confidence = sum(
                r['probability'] * r['confidence'] 
                for r in confidence_results.values()
            ) / total_prob
        else:
            overall_confidence = 0.0
        
        # Categorize by confidence level
        high_conf = [cat for cat, data in confidence_results.items() 
                     if data['confidence'] > 0.8]
        low_conf = [cat for cat, data in confidence_results.items() 
                    if data['confidence'] < 0.5]
        
        return {
            'most_confident': sorted_by_conf[0] if sorted_by_conf else (None, 0),
            'least_confident': sorted_by_conf[-1] if sorted_by_conf else (None, 0),
            'high_confidence_categories': high_conf,
            'low_confidence_categories': low_conf,
            'overall_confidence': overall_confidence
        }
    
    def should_trust_prediction(self, confidence_results, min_confidence=0.7, 
                               min_evidence=50):
        """
        Decision helper: Should we trust this prediction?
        
        Returns:
            {
                'trust': True/False,
                'reason': 'explanation',
                'recommendation': 'suggested action'
            }
        """
        # Get top prediction
        top_category = max(confidence_results.items(), 
                          key=lambda x: x[1]['probability'])
        cat_name, cat_data = top_category
        
        prob = cat_data['probability']
        conf = cat_data['confidence']
        evidence = cat_data.get('evidence_count', 0)
        
        # Decision logic
        if conf >= min_confidence and evidence >= min_evidence:
            return {
                'trust': True,
                'reason': f"High confidence ({conf:.2f}) and strong evidence ({evidence} data points)",
                'recommendation': f"Confidently predict: {cat_name}"
            }
        
        elif conf >= min_confidence and evidence < min_evidence:
            return {
                'trust': False,
                'reason': f"High confidence ({conf:.2f}) but weak evidence ({evidence} data points)",
                'recommendation': f"Likely {cat_name}, but gather more data"
            }
        
        elif conf < min_confidence and evidence >= min_evidence:
            return {
                'trust': False,
                'reason': f"Strong evidence ({evidence} data points) but low confidence ({conf:.2f})",
                'recommendation': f"Uncertain between multiple options - ask user for clarification"
            }
        
        else:
            return {
                'trust': False,
                'reason': f"Low confidence ({conf:.2f}) and weak evidence ({evidence} data points)",
                'recommendation': "Insufficient data - defer to user or request more information"
            }
    
    def format_prediction_report(self, confidence_results, top_n=5):
        """
        Format results as human-readable report
        """
        sorted_results = sorted(
            confidence_results.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        
        report = []
        report.append("="*60)
        report.append("PREDICTION WITH CONFIDENCE")
        report.append("="*60)
        report.append("")
        
        for i, (category, data) in enumerate(sorted_results[:top_n], 1):
            prob = data['probability']
            conf = data['confidence']
            evidence = data.get('evidence_count', 0)
            
            # Confidence indicator
            if conf > 0.8:
                conf_indicator = "█████ (Very High)"
            elif conf > 0.6:
                conf_indicator = "████░ (High)"
            elif conf > 0.4:
                conf_indicator = "███░░ (Medium)"
            elif conf > 0.2:
                conf_indicator = "██░░░ (Low)"
            else:
                conf_indicator = "█░░░░ (Very Low)"
            
            report.append(f"{i}. {category}")
            report.append(f"   Probability: {prob*100:.1f}%")
            report.append(f"   Confidence:  {conf_indicator} ({conf:.2f})")
            report.append(f"   Evidence:    {evidence} data points")
            report.append(f"   Std Dev:     {data['std']:.3f}")
            report.append("")
        
        summary = self.get_confidence_summary(confidence_results)
        report.append("-"*60)
        report.append(f"Overall Confidence: {summary['overall_confidence']:.2f}")
        
        if summary['high_confidence_categories']:
            report.append(f"High Confidence: {', '.join(summary['high_confidence_categories'][:3])}")
        
        if summary['low_confidence_categories']:
            report.append(f"Low Confidence:  {', '.join(summary['low_confidence_categories'][:3])}")
        
        report.append("="*60)
        
        return "\n".join(report)