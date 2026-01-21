"""
Path Coherence Module
"""
import numpy as np
from collections import defaultdict
from datetime import datetime
from config import (COHERENCE_SHARED_CATEGORIES_WEIGHT, 
                    COHERENCE_CO_OCCURRENCE_WEIGHT, 
                    COHERENCE_TEMPORAL_WEIGHT)


class CoherenceCalculator:
    """
    Efficiently calculates coherence between nodes
    Uses incremental updates to avoid O(n²) rebuilds
    """
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.coherence_cache = {}
        self.node_metadata = {}
        self._last_processed_search_idx = 0
        self._build_metadata()
    
    def _build_metadata(self):
        """
        Pre-compute metadata for each node (done once during init)
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
                    appears = False
                    
                    if node_type == 'category':
                        if node in search.get('categories', {}):
                            appears = True
                    elif node_type == 'entity':
                        #  Use entity_ids if available, fallback consistently
                        entity_ids = search.get('entity_ids', [])
                        if node in entity_ids:
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
        
        self._last_processed_search_idx = len(self.kg.search_history) - 1
        print(f"Metadata built for {len(self.node_metadata)} nodes")
    
    def incremental_update(self):
        """
        Update metadata incrementally from new searches only
        Only processes searches added since last update - O(n) instead of O(n²)
        """
        current_search_count = len(self.kg.search_history)
        
        # Nothing new to process
        if current_search_count <= self._last_processed_search_idx + 1:
            return
        
        # Process only NEW searches
        new_searches = self.kg.search_history[self._last_processed_search_idx + 1:]
        new_nodes_found = False
        
        for idx_offset, search in enumerate(new_searches):
            actual_idx = self._last_processed_search_idx + 1 + idx_offset
            
            # Process categories
            for category in search.get('categories', {}).keys():
                if category not in self.node_metadata:
                    # New category node - initialize metadata
                    self.node_metadata[category] = {
                        'categories': set(),
                        'search_indices': set(),
                        'typical_hours': [],
                        'node_type': 'category'
                    }
                    new_nodes_found = True
                
                # Update metadata
                self.node_metadata[category]['search_indices'].add(actual_idx)
                timestamp = search.get('timestamp')
                if timestamp:
                    self.node_metadata[category]['typical_hours'].append(timestamp.hour)
            
            # Process entities using entity_ids consistently
            entity_ids = search.get('entity_ids', [])
            
            for entity_id in entity_ids:
                if entity_id not in self.node_metadata:
                    # New entity - initialize metadata
                    connected_cats = set()
                    if entity_id in self.kg.G:
                        for neighbor in self.kg.G.neighbors(entity_id):
                            neighbor_data = self.kg.G.nodes.get(neighbor, {})
                            if neighbor_data.get('node_type') == 'category':
                                connected_cats.add(neighbor)
                    
                    self.node_metadata[entity_id] = {
                        'categories': connected_cats,
                        'search_indices': set(),
                        'typical_hours': [],
                        'node_type': 'entity'
                    }
                    new_nodes_found = True
                
                # Update metadata
                self.node_metadata[entity_id]['search_indices'].add(actual_idx)
                timestamp = search.get('timestamp')
                if timestamp:
                    self.node_metadata[entity_id]['typical_hours'].append(timestamp.hour)
                
                # Update category connections (may have changed)
                if entity_id in self.kg.G:
                    current_cats = set()
                    for neighbor in self.kg.G.neighbors(entity_id):
                        neighbor_data = self.kg.G.nodes.get(neighbor, {})
                        if neighbor_data.get('node_type') == 'category':
                            current_cats.add(neighbor)
                    self.node_metadata[entity_id]['categories'] = current_cats
        
        self._last_processed_search_idx = current_search_count - 1
        
        # Note: Cache is now cleared explicitly by training pipeline after this call
    
    def calculate_shared_categories(self, node_A, node_B):
        """Calculate Jaccard similarity of connected categories"""
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
        """Calculate how often these nodes appear in same searches"""
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
        """Simple temporal overlap: do they appear at similar hours?"""
        meta_A = self.node_metadata.get(node_A, {})
        meta_B = self.node_metadata.get(node_B, {})
        
        hours_A = meta_A.get('typical_hours', [])
        hours_B = meta_B.get('typical_hours', [])
        
        if not hours_A or not hours_B:
            return 0.5
        
        # Count overlaps within ±2 hours
        overlaps = 0
        total_pairs = 0
        
        for hour_a in hours_A[-10:]:
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
        if node_A == node_B:
            return 1.0
        
        cache_key = tuple(sorted([node_A, node_B]))
        if cache_key in self.coherence_cache:
            return self.coherence_cache[cache_key]
        
        if node_A == "USER" or node_B == "USER":
            self.coherence_cache[cache_key] = 0.8
            return 0.8
        
        shared_cats = self.calculate_shared_categories(node_A, node_B)
        co_occurrence = self.calculate_co_occurrence(node_A, node_B)
        temporal = self.calculate_temporal_overlap(node_A, node_B)
        
        coherence = (
            shared_cats * COHERENCE_SHARED_CATEGORIES_WEIGHT +
            co_occurrence * COHERENCE_CO_OCCURRENCE_WEIGHT +
            temporal * COHERENCE_TEMPORAL_WEIGHT
        )
        
        self.coherence_cache[cache_key] = coherence
        return coherence
    
    def clear_cache(self):
        """Clear coherence cache (called periodically during training)"""
        self.coherence_cache = {}
    
    def rebuild_metadata(self):
        """
        DEPRECATED: Use incremental_update() instead
        Full rebuild kept for backward compatibility but should be avoided
        """
        print("WARNING: Full rebuild is expensive. Use incremental_update() instead.")
        self.node_metadata = {}
        self.coherence_cache = {}
        self._last_processed_search_idx = 0
        self._build_metadata()