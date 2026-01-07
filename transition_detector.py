"""
Transition Detection - WITH ADAPTIVE THRESHOLDS
Identifies life transitions using statistics derived from user's own data
"""
import numpy as np
from config import TRANSITION_WINDOW_SIZE, SURPRISE_THRESHOLD_SIGMA


class TransitionDetector:
    """
    Detects life transitions with ADAPTIVE thresholds
    No more magic numbers - thresholds derived from user's surprise distribution
    """
    
    def __init__(self, window_size=TRANSITION_WINDOW_SIZE, 
                 surprise_sigma=SURPRISE_THRESHOLD_SIGMA):
        self.window_size = window_size
        self.surprise_sigma = surprise_sigma
        self.transitions = []
        
        # Cache for adaptive threshold
        self._cached_threshold = None
        self._last_history_length = 0
        
    def calculate_adaptive_threshold(self, surprise_history):
        """
        Calculate threshold from user's own data
        threshold = mean(surprise) + k * std(surprise)
        where k is the only tunable parameter (interpretable!)
        """
        if len(surprise_history) < self.window_size * 2:
            # Not enough data yet, use a conservative default
            return np.mean(surprise_history) * 1.5 if surprise_history else 3.0
        
        # Calculate statistics from ALL historical data
        mean_surprise = np.mean(surprise_history)
        std_surprise = np.std(surprise_history)
        
        # Threshold = mean + k*std (only k is a parameter, rest is data-driven)
        threshold = mean_surprise + self.surprise_sigma * std_surprise
        
        return threshold
    
    def detect_transition(self, surprise_history):
        """
        Check if recent surprise indicates a life transition
        NOW USES ADAPTIVE THRESHOLD from user's own data
        """
        if len(surprise_history) < self.window_size:
            return False, 0.0
        
        # Get recent surprise scores
        recent_surprise = surprise_history[-self.window_size:]
        avg_surprise = np.mean(recent_surprise)
        
        # Calculate ADAPTIVE baseline (exclude recent window)
        if len(surprise_history) > self.window_size:
            baseline_surprise = np.mean(surprise_history[:-self.window_size])
        else:
            baseline_surprise = avg_surprise
        
        # Calculate ADAPTIVE threshold (only recalc periodically for efficiency)
        if (self._cached_threshold is None or 
            len(surprise_history) - self._last_history_length > 100):
            self._cached_threshold = self.calculate_adaptive_threshold(surprise_history)
            self._last_history_length = len(surprise_history)
        
        adaptive_threshold = self._cached_threshold
        
        # Transition score (relative to baseline)
        if baseline_surprise > 0:
            transition_score = avg_surprise / baseline_surprise
        else:
            transition_score = 1.0
        
        # Detect transition using ADAPTIVE threshold
        is_transition = (avg_surprise > adaptive_threshold and 
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
    
    def get_current_threshold(self, surprise_history):
        """Get the current adaptive threshold for reporting/visualization"""
        return self.calculate_adaptive_threshold(surprise_history)
    
    def __repr__(self):
        return f"TransitionDetector(window={self.window_size}, sigma={self.surprise_sigma}, transitions={len(self.transitions)})"