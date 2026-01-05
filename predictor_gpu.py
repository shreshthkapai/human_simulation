"""
GPU-accelerated Graph Predictor
Uses PyTorch for GPU-accelerated random walks
"""
import torch
import numpy as np
from config import NUM_WALKS, WALK_LENGTH, LEARNING_RATE


class GraphPredictorGPU:
    """
    Predicts next search using GPU-accelerated random walks
    Note: Can cause CUDA out of memory on large graphs
    """
    
    def __init__(self, knowledge_graph, num_walks=NUM_WALKS, walk_length=WALK_LENGTH):
        self.kg = knowledge_graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.surprise_scores = []
        
        print(f"GraphPredictorGPU initialized on: {self.device}")
    
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
        
        # Row-normalize for probability distribution
        row_sums = adj.sum(dim=1, keepdim=True)
        
        # Fix dead ends
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
    
    def predict_next_category(self):
        """Predict next search category using GPU random walks"""
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor()
        
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
        
        total_visits = sum(category_visits.values())
        
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        prediction = {cat: category_visits[cat] / total_visits 
                     for cat in self.kg.categories}
        
        return prediction
    
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
                current_weight = self.kg.G["USER"][cat]['weight']
                adjustment = learning_rate * error
                new_weight = np.clip(current_weight + adjustment, 0.0, 1.0)
                self.kg.G["USER"][cat]['weight'] = new_weight