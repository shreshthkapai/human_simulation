"""
Hybrid Predictor: CPU random walks + GPU LLM
Memory-safe alternative to full GPU version
"""
import random
import numpy as np
from collections import Counter
from config import NUM_WALKS, WALK_LENGTH, LEARNING_RATE

class GraphPredictorHybrid:
    """
    CPU-based predictor
    """
    
    def __init__(self, knowledge_graph, num_walks=30, walk_length=4):
        self.kg = knowledge_graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        
        self.surprise_scores = []
        
        print(f"GraphPredictorHybrid initialized (CPU walks, GPU LLM)")
    
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
    
    def predict_next_category(self):
        """Predict using CPU random walks"""
        from collections import Counter
        
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
    
    def calculate_surprise(self, predicted_dist, actual_categories):
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
    
    def update_graph_weights(self, predicted_dist, actual_categories, learning_rate=0.15):
        """Same as before"""
        for cat in self.kg.categories:
            predicted_prob = predicted_dist.get(cat, 0)
            actual_conf = actual_categories.get(cat, 0)
            
            error = actual_conf - predicted_prob
            
            if self.kg.G.has_edge("USER", cat):
                current_weight = self.kg.G["USER"][cat]['weight']
                adjustment = learning_rate * error
                new_weight = np.clip(current_weight + adjustment, 0.0, 1.0)
                self.kg.G["USER"][cat]['weight'] = new_weight

print("Hybrid predictor ready")