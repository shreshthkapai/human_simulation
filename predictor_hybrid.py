"""
Hybrid Predictor with CONTEXT-SENSITIVE ACTIVATION
No longer always starts from USER - starts from current mental state
"""
import random
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from config import NUM_WALKS, WALK_LENGTH, LEARNING_RATE


class GraphPredictorHybrid:
    """
    CPU-based predictor with CONTEXT-AWARE activation
    Simulates "what's on my mind right now" before predicting
    """
    
    def __init__(self, knowledge_graph, num_walks=30, walk_length=4):
        self.kg = knowledge_graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        
        self.surprise_scores = []
        
        print(f"GraphPredictorHybrid initialized (Context-Sensitive Activation)")
    
    def weighted_random_walk(self, start_node, length):
        """CPU-based random walk"""
        current = start_node
        path = [current]
        
        for _ in range(length):
            neighbors = list(self.kg.G.neighbors(current))
            
            if not neighbors:
                break
            
            weights = []
            for neighbor in neighbors:
                edge_data = self.kg.G[current][neighbor]
                weight = edge_data.get('weight', 0.1)
                weights.append(max(weight, 0.01))
            
            total = sum(weights)
            if total == 0:
                probs = [1.0 / len(weights)] * len(weights)
            else:
                probs = [w / total for w in weights]
            
            current = random.choices(neighbors, weights=probs)[0]
            path.append(current)
        
        return path
    
    def get_context_nodes(self, current_timestamp=None, num_context_nodes=3):
        """
        NEW: Determine which nodes are currently "activated" in mind
        Context = recent activity + temporal priming + transition state
        """
        context_nodes = ["USER"]  # Always include USER as base
        
        if not self.kg.search_history or len(self.kg.search_history) < 10:
            return context_nodes
        
        # 1. RECENT ACTIVATION: Entities from last N searches
        recent_searches = self.kg.search_history[-5:]
        recent_entities = []
        for search in recent_searches:
            for entity in search.get('entities', []):
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                if entity_id in self.kg.G:
                    # Weight by recency
                    mention_count = self.kg.G.nodes[entity_id].get('mention_count', 1)
                    score = 1.0 + np.log1p(mention_count)
                    recent_entities.append((entity_id, score))
        
        # Sort by score and take top
        recent_entities.sort(key=lambda x: x[1], reverse=True)
        for entity_id, _ in recent_entities[:2]:
            if entity_id not in context_nodes:
                context_nodes.append(entity_id)
        
        # 2. TEMPORAL PRIMING: What's typical for this time?
        if current_timestamp:
            hour = current_timestamp.hour
            day_of_week = current_timestamp.weekday()
            
            # Find entities/categories active at similar times historically
            temporal_matches = []
            for search in self.kg.search_history[-1000:]:  # Look at last 1000
                search_time = search.get('timestamp')
                if search_time:
                    # Same hour of day (±2 hours)
                    if abs(search_time.hour - hour) <= 2:
                        for entity in search.get('entities', []):
                            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                            if entity_id in self.kg.G:
                                temporal_matches.append(entity_id)
                    
                    # Same day of week
                    if search_time.weekday() == day_of_week:
                        for entity in search.get('entities', []):
                            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                            if entity_id in self.kg.G:
                                temporal_matches.append(entity_id)
            
            # Count frequency
            if temporal_matches:
                temporal_freq = Counter(temporal_matches)
                top_temporal = temporal_freq.most_common(1)
                if top_temporal and top_temporal[0][0] not in context_nodes:
                    context_nodes.append(top_temporal[0][0])
        
        # 3. TRANSITION STATE: If in transition, emphasize recent over stable
        if len(self.surprise_scores) > 25:
            recent_surprise = np.mean(self.surprise_scores[-25:])
            baseline_surprise = np.mean(self.surprise_scores[:-25])
            
            if recent_surprise > baseline_surprise * 1.3:
                # In transition - weight recent more heavily
                # Add most recent category
                if recent_searches:
                    last_categories = recent_searches[-1].get('categories', {})
                    if last_categories:
                        top_cat = max(last_categories.items(), key=lambda x: x[1])[0]
                        if top_cat not in context_nodes:
                            context_nodes.append(top_cat)
        
        # Limit total context nodes to avoid dilution
        return context_nodes[:num_context_nodes + 1]  # +1 for USER
    
    def predict_next_category(self, current_timestamp=None, use_context=True):
        """
        Predict using context-sensitive activation
        NOW: Starts walks from MULTIPLE context nodes, not just USER
        """
        if not use_context:
            # Legacy mode: just start from USER
            category_visits = Counter()
            for _ in range(self.num_walks):
                path = self.weighted_random_walk("USER", self.walk_length)
                for node in path:
                    node_data = self.kg.G.nodes.get(node, {})
                    if node_data.get('node_type') == 'category':
                        category_visits[node] += 1
        else:
            # CONTEXT-SENSITIVE MODE
            context_nodes = self.get_context_nodes(current_timestamp)
            
            # Distribute walks across context nodes
            walks_per_node = self.num_walks // len(context_nodes)
            remaining_walks = self.num_walks % len(context_nodes)
            
            category_visits = Counter()
            
            for i, start_node in enumerate(context_nodes):
                num_walks = walks_per_node + (1 if i < remaining_walks else 0)
                
                for _ in range(num_walks):
                    path = self.weighted_random_walk(start_node, self.walk_length)
                    
                    for node in path:
                        node_data = self.kg.G.nodes.get(node, {})
                        if node_data.get('node_type') == 'category':
                            category_visits[node] += 1
        
        total_visits = sum(category_visits.values())
        
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        prediction = {cat: category_visits.get(cat, 0) / total_visits 
                     for cat in self.kg.categories}
        
        return prediction
    
    def predict_from_context(self, start_nodes, num_walks_per_node=10):
        """
        ENHANCED: Explicit context-based prediction
        Useful for simulation: "given I'm thinking about X, Y, Z..."
        """
        category_visits = Counter()
        
        for start_node in start_nodes:
            if start_node not in self.kg.G:
                continue
                
            for _ in range(num_walks_per_node):
                path = self.weighted_random_walk(start_node, self.walk_length)
                
                for node in path:
                    node_data = self.kg.G.nodes.get(node, {})
                    if node_data.get('node_type') == 'category':
                        category_visits[node] += 1
        
        total_visits = sum(category_visits.values())
        
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        prediction = {cat: category_visits.get(cat, 0) / total_visits 
                     for cat in self.kg.categories}
        
        return prediction
    
    def explain_why(self, target_node, num_walks=30):
        """Backward reasoning - "Why am I thinking about this?" """
        incoming_nodes = list(self.kg.G.predecessors(target_node))
        
        if not incoming_nodes:
            return {}
        
        cause_visits = Counter()
        
        for _ in range(num_walks):
            path = self.weighted_random_walk(target_node, self.walk_length)
            
            for node in path:
                node_data = self.kg.G.nodes.get(node, {})
                node_type = node_data.get('node_type')
                
                if node_type in ['category', 'entity']:
                    cause_visits[node] += 1
        
        total = sum(cause_visits.values())
        if total == 0:
            return {}
        
        return {node: count/total for node, count in cause_visits.most_common(10)}
    
    def calculate_surprise(self, predicted_dist, actual_categories):
        """Measure surprise using KL divergence"""
        actual_dist = {cat: 0.0 for cat in self.kg.categories}
        
        total_confidence = sum(actual_categories.values())
        if total_confidence > 0:
            for cat, conf in actual_categories.items():
                if cat in actual_dist:
                    actual_dist[cat] = conf / total_confidence
        else:
            for cat in actual_dist:
                actual_dist[cat] = 1.0 / len(self.kg.categories)
        
        epsilon = 1e-10
        
        predicted_array = np.array([predicted_dist.get(cat, epsilon) + epsilon 
                                    for cat in self.kg.categories])
        actual_array = np.array([actual_dist.get(cat, epsilon) + epsilon 
                                for cat in self.kg.categories])
        
        predicted_array = predicted_array / predicted_array.sum()
        actual_array = actual_array / actual_array.sum()
        
        kl_div = np.sum(actual_array * np.log(actual_array / predicted_array))
        
        return kl_div
    
    def update_graph_weights(self, predicted_dist, actual_categories, learning_rate=LEARNING_RATE):
        """Update edge weights based on prediction accuracy"""
        for cat in self.kg.categories:
            predicted_prob = predicted_dist.get(cat, 0)
            actual_conf = actual_categories.get(cat, 0)
            
            error = actual_conf - predicted_prob
            
            if self.kg.G.has_edge("USER", cat):
                current = self.kg.G["USER"][cat]['weight']
                adjustment = learning_rate * error
                new_weight = np.clip(current + adjustment, 0.0, 1.0)
                self.kg.G["USER"][cat]['weight'] = new_weight
            
            if self.kg.G.has_edge(cat, "USER"):
                current = self.kg.G[cat]["USER"]['weight']
                adjustment = learning_rate * error * 0.3
                new_weight = np.clip(current + adjustment, 0.0, 1.0)
                self.kg.G[cat]["USER"]['weight'] = new_weight


print("Hybrid predictor ready (Context-Sensitive)")