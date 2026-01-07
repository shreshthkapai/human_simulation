"""
Path Coherence Module
Calculates coherence between nodes to prefer logical reasoning paths
"""
import numpy as np
from collections import defaultdict
from datetime import datetime


class CoherenceCalculator:
    """
    Efficiently calculates coherence between nodes
    Uses caching and on-demand computation to avoid O(n²) complexity
    """
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.coherence_cache = {}  # Cache computed coherences
        
        # Pre-compute metadata for fast lookups
        self.node_metadata = {}
        self._build_metadata()
    
    def _build_metadata(self):
        """
        Pre-compute metadata for each node (done once during init)
        This enables O(1) coherence lookups instead of O(n) iteration
        """
        print("Building coherence metadata...")
        
        for node in self.kg.G.nodes():
            node_data = self.kg.G.nodes[node]
            node_type = node_data.get('node_type')
            
            if node_type in ['entity', 'category']:
                # Get connected categories
                connected_cats = set()
                for neighbor in self.kg.G.neighbors(node):
                    neighbor_data = self.kg.G.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'category':
                        connected_cats.add(neighbor)
                
                # Get searches this node appears in
                search_indices = []
                search_hours = []
                
                for idx, search in enumerate(self.kg.search_history):
                    # Check if this node appears in search
                    appears = False
                    
                    if node_type == 'category':
                        if node in search.get('categories', {}):
                            appears = True
                    elif node_type == 'entity':
                        node_label = node_data.get('label', '')
                        if node_label in search.get('entities', []):
                            appears = True
                    
                    if appears:
                        search_indices.append(idx)
                        timestamp = search.get('timestamp')
                        if timestamp:
                            search_hours.append(timestamp.hour)
                
                self.node_metadata[node] = {
                    'categories': connected_cats,
                    'search_indices': set(search_indices),
                    'typical_hours': search_hours,
                    'node_type': node_type
                }
        
        print(f"Metadata built for {len(self.node_metadata)} nodes")
    
    def calculate_shared_categories(self, node_A, node_B):
        """
        Calculate Jaccard similarity of connected categories
        Formula: |A ∩ B| / |A ∪ B|
        """
        meta_A = self.node_metadata.get(node_A, {})
        meta_B = self.node_metadata.get(node_B, {})
        
        cats_A = meta_A.get('categories', set())
        cats_B = meta_B.get('categories', set())
        
        if not cats_A and not cats_B:
            return 0.0
        
        intersection = len(cats_A & cats_B)
        union = len(cats_A | cats_B)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_co_occurrence(self, node_A, node_B):
        """
        Calculate how often these nodes appear in same searches
        Formula: |searches with both| / |searches with either|
        """
        meta_A = self.node_metadata.get(node_A, {})
        meta_B = self.node_metadata.get(node_B, {})
        
        searches_A = meta_A.get('search_indices', set())
        searches_B = meta_B.get('search_indices', set())
        
        if not searches_A and not searches_B:
            return 0.0
        
        both = len(searches_A & searches_B)
        either = len(searches_A | searches_B)
        
        return both / either if either > 0 else 0.0
    
    def calculate_temporal_overlap(self, node_A, node_B):
        """
        Simple temporal overlap: do they appear at similar hours?
        """
        meta_A = self.node_metadata.get(node_A, {})
        meta_B = self.node_metadata.get(node_B, {})
        
        hours_A = meta_A.get('typical_hours', [])
        hours_B = meta_B.get('typical_hours', [])
        
        if not hours_A or not hours_B:
            return 0.5  # Neutral if no data
        
        # Count overlaps within ±2 hours
        overlaps = 0
        total_pairs = 0
        
        for hour_a in hours_A[-10:]:  # Only check last 10 for efficiency
            for hour_b in hours_B[-10:]:
                total_pairs += 1
                if abs(hour_a - hour_b) <= 2:
                    overlaps += 1
        
        return overlaps / total_pairs if total_pairs > 0 else 0.5
    
    def get_coherence(self, node_A, node_B):
        """
        Get coherence score between two nodes (with caching)
        Returns value between 0.0 (unrelated) and 1.0 (highly coherent)
        """
        # Skip if same node
        if node_A == node_B:
            return 1.0
        
        # Check cache (bidirectional)
        cache_key = tuple(sorted([node_A, node_B]))
        if cache_key in self.coherence_cache:
            return self.coherence_cache[cache_key]
        
        # Skip if either node is USER (always coherent with USER)
        if node_A == "USER" or node_B == "USER":
            self.coherence_cache[cache_key] = 0.8  # High default coherence with USER
            return 0.8
        
        # Calculate sub-scores
        shared_cats = self.calculate_shared_categories(node_A, node_B)
        co_occurrence = self.calculate_co_occurrence(node_A, node_B)
        temporal = self.calculate_temporal_overlap(node_A, node_B)
        
        # Weighted combination
        coherence = (
            shared_cats * 0.4 +      # 40% weight on shared categories
            co_occurrence * 0.4 +    # 40% weight on co-occurrence
            temporal * 0.2           # 20% weight on temporal overlap
        )
        
        # Cache for future use
        self.coherence_cache[cache_key] = coherence
        
        return coherence
    
    def clear_cache(self):
        """Clear coherence cache (e.g., after graph updates)"""
        self.coherence_cache = {}
    
    def rebuild_metadata(self):
        """Rebuild metadata (e.g., after significant graph changes)"""
        self.node_metadata = {}
        self.coherence_cache = {}
        self._build_metadata()