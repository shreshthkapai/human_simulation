"""
Hybrid Predictor with TRUE bidirectional graph support
Can now actually traverse in both directions
"""
import random
import numpy as np
from collections import Counter
from config import NUM_WALKS, WALK_LENGTH, LEARNING_RATE


class GraphPredictorHybrid:
    """
    CPU-based predictor with TRUE bidirectional traversal
    """
    
    def __init__(self, knowledge_graph, num_walks=30, walk_length=4):
        self.kg = knowledge_graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        
        self.surprise_scores = []
        
        print(f"GraphPredictorHybrid initialized (CPU walks, TRUE bidirectional)")
    
    def weighted_random_walk(self, start_node, length):
        """
        CPU-based random walk - now can traverse in ANY direction
        because all edges exist in both directions!
        """
        current = start_node
        path = [current]
        
        for _ in range(length):
            neighbors = list(self.kg.G.neighbors(current))
            
            if not neighbors:
                break
            
            weights = []
            for neighbor in neighbors:
                # Just use weight - the edge direction is already correct
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
    
    def predict_next_category(self):
        """
        Predict using CPU random walks
        Now naturally follows whatever edges exist (including reverse!)
        """
        category_visits = Counter()
        
        for _ in range(self.num_walks):
            path = self.weighted_random_walk("USER", self.walk_length)
            
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
        NEW: Predict starting from multiple context nodes
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
        """
        NEW: Backward reasoning - "Why am I thinking about this?"
        Walks backward from target to find causes
        """
        # Get nodes that point TO target (reverse edges that actually exist!)
        incoming_nodes = list(self.kg.G.predecessors(target_node))
        
        if not incoming_nodes:
            return {}
        
        cause_visits = Counter()
        
        for _ in range(num_walks):
            # Start from target, walk backward
            path = self.weighted_random_walk(target_node, self.walk_length)
            
            for node in path:
                node_data = self.kg.G.nodes.get(node, {})
                node_type = node_data.get('node_type')
                
                if node_type in ['category', 'entity']:
                    cause_visits[node] += 1
        
        # Return top causes
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
        """
        Update edge weights based on prediction accuracy
        Updates USER → category edges
        """
        for cat in self.kg.categories:
            predicted_prob = predicted_dist.get(cat, 0)
            actual_conf = actual_categories.get(cat, 0)
            
            error = actual_conf - predicted_prob
            
            # Update forward edge (USER → category)
            if self.kg.G.has_edge("USER", cat):
                current = self.kg.G["USER"][cat]['weight']
                adjustment = learning_rate * error
                new_weight = np.clip(current + adjustment, 0.0, 1.0)
                self.kg.G["USER"][cat]['weight'] = new_weight
            
            # Update backward edge (category → USER) more conservatively
            if self.kg.G.has_edge(cat, "USER"):
                current = self.kg.G[cat]["USER"]['weight']
                adjustment = learning_rate * error * 0.3  # Weaker backward update
                new_weight = np.clip(current + adjustment, 0.0, 1.0)
                self.kg.G[cat]["USER"]['weight'] = new_weight


print("Hybrid predictor ready (TRUE bidirectional)")