"""
Transition Detection
Identifies life transitions from sustained high surprise
"""
import numpy as np
from config import TRANSITION_WINDOW_SIZE, SURPRISE_THRESHOLD


class TransitionDetector:
    """
    Detects life transitions by monitoring prediction surprise over time
    """
    
    def __init__(self, window_size=TRANSITION_WINDOW_SIZE, surprise_threshold=SURPRISE_THRESHOLD):
        self.window_size = window_size
        self.surprise_threshold = surprise_threshold
        self.transitions = []
        
    def detect_transition(self, surprise_history):
        """
        Check if recent surprise indicates a life transition
        Returns: (is_transition, transition_score)
        """
        if len(surprise_history) < self.window_size:
            return False, 0.0
        
        # Get recent surprise scores
        recent_surprise = surprise_history[-self.window_size:]
        avg_surprise = np.mean(recent_surprise)
        
        # Calculate overall baseline
        if len(surprise_history) > self.window_size:
            baseline_surprise = np.mean(surprise_history[:-self.window_size])
        else:
            baseline_surprise = avg_surprise
        
        # Transition score
        if baseline_surprise > 0:
            transition_score = avg_surprise / baseline_surprise
        else:
            transition_score = 1.0
        
        # Detect transition
        is_transition = (avg_surprise > self.surprise_threshold and 
                        transition_score > 1.5)
        
        return is_transition, transition_score
    
    def log_transition(self, timestamp, search_index, categories, transition_score):
        """Record a detected transition"""
        transition = {
            'timestamp': timestamp,
            'search_index': search_index,
            'new_categories': categories,
            'transition_score': transition_score
        }
        self.transitions.append(transition)
        return transition
    
    def __repr__(self):
        return f"TransitionDetector(window={self.window_size}, threshold={self.surprise_threshold}, transitions={len(self.transitions)})"