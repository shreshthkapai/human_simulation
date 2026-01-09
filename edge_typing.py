"""
Edge Type Detection from Temporal Patterns
Automatically tags edges as causal, associative, or temporal based on timing
"""
import numpy as np
from collections import defaultdict
from datetime import timedelta
from config import (EDGE_TYPE_CAUSAL_WINDOW_HOURS, 
                    EDGE_TYPE_SIMULTANEOUS_WINDOW_HOURS, 
                    EDGE_TYPE_MIN_SAMPLES)


def _default_co_occurrence_data():
    """Helper for defaultdict to allow pickling"""
    return {
        'sequential': [],      # List of time deltas (A → B)
        'simultaneous': 0,     # Count of simultaneous appearances
        'total_pairs': 0       # Total observations
    }


class EdgeTypeDetector:
    """
    Analyzes temporal patterns to infer edge relationship types
    - Causal/Sequential: A consistently appears BEFORE B
    - Associative: A and B appear simultaneously
    - Temporal Correlation: A and B appear at similar times but not together
    """
    
    def __init__(self, knowledge_graph, 
                 causal_window_hours=EDGE_TYPE_CAUSAL_WINDOW_HOURS,
                 simultaneous_window_hours=EDGE_TYPE_SIMULTANEOUS_WINDOW_HOURS,
                 min_samples=EDGE_TYPE_MIN_SAMPLES):
        
        self.kg = knowledge_graph
        self.causal_window = timedelta(hours=causal_window_hours)
        self.simultaneous_window = timedelta(hours=simultaneous_window_hours)
        self.min_samples = min_samples
        
        # Track co-occurrence patterns
        self.co_occurrence_data = defaultdict(_default_co_occurrence_data)
    
    def analyze_search_history(self):
        """
        Analyze entire search history to detect temporal patterns
        This runs ONCE after training completes
        """
        print("Analyzing temporal patterns for edge typing...")
        
        # Build timeline of entity appearances
        entity_timeline = defaultdict(list)
        
        for idx, search in enumerate(self.kg.search_history):
            timestamp = search.get('timestamp')
            if not timestamp:
                continue
            
            entities = search.get('entities', [])
            categories = list(search.get('categories', {}).keys())
            
            # Track all nodes (entities + categories)
            all_nodes = []
            for entity in entities:
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                if entity_id in self.kg.G:
                    all_nodes.append(entity_id)
            
            for cat in categories:
                if cat in self.kg.G:
                    all_nodes.append(cat)
            
            # Record appearance time for each node
            for node in all_nodes:
                entity_timeline[node].append({
                    'timestamp': timestamp,
                    'search_idx': idx
                })
        
        # Analyze pairwise temporal relationships
        analyzed_pairs = 0
        nodes_with_timeline = list(entity_timeline.keys())
        
        for i, node_A in enumerate(nodes_with_timeline):
            # Only analyze nodes connected in graph
            if node_A not in self.kg.G:
                continue
            
            neighbors = list(self.kg.G.neighbors(node_A))
            
            for node_B in neighbors:
                if node_B not in entity_timeline:
                    continue
                
                # Skip if we've already analyzed this pair
                pair_key = tuple(sorted([node_A, node_B]))
                
                # Analyze temporal relationship
                appearances_A = entity_timeline[node_A]
                appearances_B = entity_timeline[node_B]
                
                sequential_deltas = []
                simultaneous_count = 0
                
                for app_A in appearances_A:
                    for app_B in appearances_B:
                        time_diff = app_B['timestamp'] - app_A['timestamp']
                        abs_time_diff = abs(time_diff.total_seconds())
                        
                        # Check if simultaneous
                        if abs_time_diff <= self.simultaneous_window.total_seconds():
                            simultaneous_count += 1
                        
                        # Check if sequential (A before B)
                        elif timedelta(0) < time_diff <= self.causal_window:
                            sequential_deltas.append(time_diff.total_seconds() / 3600)  # Convert to hours
                
                # Store pattern data
                self.co_occurrence_data[pair_key]['sequential'] = sequential_deltas
                self.co_occurrence_data[pair_key]['simultaneous'] = simultaneous_count
                self.co_occurrence_data[pair_key]['total_pairs'] = len(appearances_A) * len(appearances_B)
                
                analyzed_pairs += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Analyzed {i+1}/{len(nodes_with_timeline)} nodes...")
        
        print(f"Temporal analysis complete: {analyzed_pairs} edge pairs analyzed")
    
    def classify_edge_type(self, node_A, node_B):
        """
        Classify the relationship type between two nodes
        Returns: ('causal', strength) or ('associative', strength) or ('temporal', strength)
        """
        pair_key = tuple(sorted([node_A, node_B]))
        
        if pair_key not in self.co_occurrence_data:
            return ('unknown', 0.5)  # No data
        
        data = self.co_occurrence_data[pair_key]
        sequential = data['sequential']
        simultaneous = data['simultaneous']
        total = data['total_pairs']
        
        if total < self.min_samples:
            return ('unknown', 0.5)  # Insufficient data
        
        # Calculate proportions
        sequential_ratio = len(sequential) / total if total > 0 else 0
        simultaneous_ratio = simultaneous / total if total > 0 else 0
        
        # Classification logic
        if simultaneous_ratio > 0.3:
            # Frequently appear together → Associative
            return ('associative', simultaneous_ratio)
        
        elif sequential_ratio > 0.2:
            # A often before B → Causal/Sequential
            # Check consistency of timing
            if len(sequential) >= 3:
                std_hours = np.std(sequential)
                consistency = 1.0 / (1.0 + std_hours)  # Low std = high consistency
                
                if consistency > 0.5:
                    return ('causal', sequential_ratio * consistency)
                else:
                    return ('temporal_correlation', sequential_ratio * 0.5)
            else:
                return ('causal', sequential_ratio * 0.5)
        
        else:
            # Rare co-occurrence or weak pattern → Temporal correlation
            # Check if they appear at similar hours across different days
            return ('temporal_correlation', 0.3)
    
    def tag_all_edges(self):
        """
        Tag all edges in the graph with relationship types
        Adds 'relationship_type' and 'type_strength' to edge data
        """
        print("Tagging edges with relationship types...")
        
        tagged_count = 0
        type_distribution = defaultdict(int)
        
        for u, v, data in self.kg.G.edges(data=True):
            edge_type, strength = self.classify_edge_type(u, v)
            
            # Add to edge data
            data['relationship_type'] = edge_type
            data['type_strength'] = strength
            
            tagged_count += 1
            type_distribution[edge_type] += 1
        
        print(f"Tagged {tagged_count} edges:")
        for edge_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {edge_type}: {count} edges ({count/tagged_count*100:.1f}%)")
        
        return type_distribution
    
    def get_edge_type_weight_multiplier(self, edge_type, reasoning_mode='forward'):
        """
        Get weight multiplier based on edge type and reasoning mode
        
        reasoning_mode:
        - 'forward': Follow causal edges strongly, associative moderately
        - 'associative': Follow associative edges strongly, others equally
        - 'backward': Follow all types equally (explanation mode)
        """
        if reasoning_mode == 'forward':
            # "What happens next?" - prefer causal
            multipliers = {
                'causal': 1.5,              # Strong boost for causal
                'associative': 1.1,         # Slight boost for associative
                'temporal_correlation': 1.0, # Neutral
                'unknown': 1.0              # Neutral
            }
        
        elif reasoning_mode == 'associative':
            # "What's related?" - prefer associative
            multipliers = {
                'causal': 1.0,              # Neutral
                'associative': 1.5,         # Strong boost for associative
                'temporal_correlation': 1.2, # Slight boost
                'unknown': 1.0              # Neutral
            }
        
        elif reasoning_mode == 'backward':
            # "Why did this happen?" - all types equally valid
            multipliers = {
                'causal': 1.0,
                'associative': 1.0,
                'temporal_correlation': 1.0,
                'unknown': 1.0
            }
        
        else:
            # Default: neutral
            multipliers = {
                'causal': 1.0,
                'associative': 1.0,
                'temporal_correlation': 1.0,
                'unknown': 1.0
            }
        
        return multipliers.get(edge_type, 1.0)