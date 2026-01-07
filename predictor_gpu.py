"""
GPU-accelerated Graph Predictor - CONTEXT-SENSITIVE
Uses PyTorch for GPU-accelerated random walks with context activation
"""
import torch
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from config import NUM_WALKS, WALK_LENGTH, LEARNING_RATE


class GraphPredictorGPU:
    """
    GPU predictor with context-aware activation
    """
    
    def __init__(self, knowledge_graph, num_walks=NUM_WALKS, walk_length=WALK_LENGTH):
        self.kg = knowledge_graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.surprise_scores = []
        
        print(f"GraphPredictorGPU initialized on: {self.device} (Context-Sensitive)")
    
    def graph_to_adjacency_tensor(self):
        """Convert NetworkX graph to PyTorch adjacency tensor"""
        nodes = list(self.kg.G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)
        
        adj = torch.zeros((n, n), device=self.device)
        
        for u, v, data in self.kg.G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            weight = data.get('weight', 0.1)
            adj[u_idx, v_idx] = max(weight, 0.01)
        
        row_sums = adj.sum(dim=1, keepdim=True)
        zero_rows = (row_sums.squeeze() == 0)
        adj[zero_rows, :] = 1.0 / n
        
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums
        
        return adj, nodes, node_to_idx
    
    def random_walks_gpu(self, start_idx, adj, num_walks, walk_length):
        """Perform multiple random walks on GPU"""
        n = adj.shape[0]
        paths = torch.zeros((num_walks, walk_length + 1), dtype=torch.long, device=self.device)
        paths[:, 0] = start_idx
        
        for step in range(walk_length):
            current = paths[:, step]
            probs = adj[current]
            
            probs = torch.clamp(probs, min=1e-10)
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            next_nodes = torch.multinomial(probs, 1).squeeze(1)
            paths[:, step + 1] = next_nodes
        
        return paths
    
    def get_context_nodes(self, current_timestamp=None, num_context_nodes=3):
        """
        NEW: Determine which nodes are currently "activated" in mind
        Same logic as hybrid predictor
        """
        context_nodes = ["USER"]
        
        if not self.kg.search_history or len(self.kg.search_history) < 10:
            return context_nodes
        
        # Recent activation
        recent_searches = self.kg.search_history[-5:]
        recent_entities = []
        for search in recent_searches:
            for entity in search.get('entities', []):
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                if entity_id in self.kg.G:
                    mention_count = self.kg.G.nodes[entity_id].get('mention_count', 1)
                    score = 1.0 + np.log1p(mention_count)
                    recent_entities.append((entity_id, score))
        
        recent_entities.sort(key=lambda x: x[1], reverse=True)
        for entity_id, _ in recent_entities[:2]:
            if entity_id not in context_nodes:
                context_nodes.append(entity_id)
        
        # Temporal priming
        if current_timestamp:
            hour = current_timestamp.hour
            day_of_week = current_timestamp.weekday()
            
            temporal_matches = []
            for search in self.kg.search_history[-1000:]:
                search_time = search.get('timestamp')
                if search_time:
                    if abs(search_time.hour - hour) <= 2:
                        for entity in search.get('entities', []):
                            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                            if entity_id in self.kg.G:
                                temporal_matches.append(entity_id)
                    
                    if search_time.weekday() == day_of_week:
                        for entity in search.get('entities', []):
                            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                            if entity_id in self.kg.G:
                                temporal_matches.append(entity_id)
            
            if temporal_matches:
                temporal_freq = Counter(temporal_matches)
                top_temporal = temporal_freq.most_common(1)
                if top_temporal and top_temporal[0][0] not in context_nodes:
                    context_nodes.append(top_temporal[0][0])
        
        # Transition state
        if len(self.surprise_scores) > 25:
            recent_surprise = np.mean(self.surprise_scores[-25:])
            baseline_surprise = np.mean(self.surprise_scores[:-25])
            
            if recent_surprise > baseline_surprise * 1.3:
                if recent_searches:
                    last_categories = recent_searches[-1].get('categories', {})
                    if last_categories:
                        top_cat = max(last_categories.items(), key=lambda x: x[1])[0]
                        if top_cat not in context_nodes:
                            context_nodes.append(top_cat)
        
        return context_nodes[:num_context_nodes + 1]
    
    def predict_next_category(self, current_timestamp=None, use_context=True):
        """
        GPU prediction with context-sensitive activation
        """
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor()
        
        if not use_context:
            # Legacy mode
            if "USER" not in node_to_idx:
                return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
            
            user_idx = node_to_idx["USER"]
            paths = self.random_walks_gpu(user_idx, adj, self.num_walks, self.walk_length)
            
            category_visits = {cat: 0 for cat in self.kg.categories}
            
            paths_cpu = paths.cpu().numpy()
            for path in paths_cpu:
                for node_idx in path:
                    node = nodes[node_idx]
                    node_data = self.kg.G.nodes.get(node, {})
                    if node_data.get('node_type') == 'category':
                        category_visits[node] += 1
        else:
            # CONTEXT-SENSITIVE MODE
            context_nodes = self.get_context_nodes(current_timestamp)
            
            walks_per_node = self.num_walks // len(context_nodes)
            remaining_walks = self.num_walks % len(context_nodes)
            
            category_visits = {cat: 0 for cat in self.kg.categories}
            
            for i, start_node in enumerate(context_nodes):
                if start_node not in node_to_idx:
                    continue
                
                num_walks = walks_per_node + (1 if i < remaining_walks else 0)
                start_idx = node_to_idx[start_node]
                
                paths = self.random_walks_gpu(start_idx, adj, num_walks, self.walk_length)
                
                paths_cpu = paths.cpu().numpy()
                for path in paths_cpu:
                    for node_idx in path:
                        node = nodes[node_idx]
                        node_data = self.kg.G.nodes.get(node, {})
                        if node_data.get('node_type') == 'category':
                            category_visits[node] += 1
        
        total_visits = sum(category_visits.values())
        
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        prediction = {cat: category_visits[cat] / total_visits 
                     for cat in self.kg.categories}
        
        return prediction
    
    def predict_from_context(self, start_nodes, num_walks_per_node=10):
        """Context-based prediction"""
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor()
        
        category_visits = {cat: 0 for cat in self.kg.categories}
        
        for start_node in start_nodes:
            if start_node not in node_to_idx:
                continue
            
            start_idx = node_to_idx[start_node]
            paths = self.random_walks_gpu(start_idx, adj, num_walks_per_node, self.walk_length)
            
            paths_cpu = paths.cpu().numpy()
            for path in paths_cpu:
                for node_idx in path:
                    node = nodes[node_idx]
                    node_data = self.kg.G.nodes.get(node, {})
                    if node_data.get('node_type') == 'category':
                        category_visits[node] += 1
        
        total_visits = sum(category_visits.values())
        
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        prediction = {cat: category_visits[cat] / total_visits 
                     for cat in self.kg.categories}
        
        return prediction
    
    def explain_why(self, target_node, num_walks=30):
        """Backward reasoning"""
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor()
        
        if target_node not in node_to_idx:
            return {}
        
        target_idx = node_to_idx[target_node]
        paths = self.random_walks_gpu(target_idx, adj, num_walks, self.walk_length)
        
        cause_visits = {}
        
        paths_cpu = paths.cpu().numpy()
        for path in paths_cpu:
            for node_idx in path:
                node = nodes[node_idx]
                node_data = self.kg.G.nodes.get(node, {})
                node_type = node_data.get('node_type')
                
                if node_type in ['category', 'entity']:
                    cause_visits[node] = cause_visits.get(node, 0) + 1
        
        total = sum(cause_visits.values())
        if total == 0:
            return {}
        
        sorted_causes = sorted(cause_visits.items(), key=lambda x: x[1], reverse=True)[:10]
        return {node: count/total for node, count in sorted_causes}
    
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