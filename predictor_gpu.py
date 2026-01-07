"""
GPU Predictor - All Features
"""
import torch
import numpy as np
from collections import Counter
from config import NUM_WALKS, WALK_LENGTH, LEARNING_RATE, USE_COHERENCE, USE_EDGE_TYPES, USE_COMPETITION
from coherence import CoherenceCalculator
from edge_typing import EdgeTypeDetector
from competition import CompetitionManager


class GraphPredictorGPU:
    
    def __init__(self, knowledge_graph, num_walks=NUM_WALKS, walk_length=WALK_LENGTH,
                 use_coherence=True, use_edge_types=True, use_competition=True):
        self.kg = knowledge_graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_coherence = use_coherence
        self.use_edge_types = use_edge_types
        self.use_competition = use_competition
        
        self.surprise_scores = []
        
        if self.use_coherence:
            self.coherence_calc = CoherenceCalculator(knowledge_graph)
        else:
            self.coherence_calc = None
        
        if self.use_edge_types:
            self.edge_type_detector = EdgeTypeDetector(knowledge_graph)
        else:
            self.edge_type_detector = None
        
        if self.use_competition:
            self.competition_manager = CompetitionManager(knowledge_graph)
        else:
            self.competition_manager = None
        
        features = []
        if self.use_coherence: features.append("Coherence")
        if self.use_edge_types: features.append("Edge Types")
        if self.use_competition: features.append("Competition")
        
        feature_str = " + ".join(features) if features else "Basic"
        print(f"GraphPredictorGPU on {self.device} ({feature_str})")
    
    def graph_to_adjacency_tensor(self, reasoning_mode='forward'):
        nodes = list(self.kg.G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)
        
        adj = torch.zeros((n, n), device=self.device)
        
        for u, v, data in self.kg.G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            weight = data.get('weight', 0.1)
            
            if self.use_edge_types and self.edge_type_detector:
                edge_type = data.get('relationship_type', 'unknown')
                type_multiplier = self.edge_type_detector.get_edge_type_weight_multiplier(
                    edge_type, reasoning_mode
                )
                weight *= type_multiplier
            
            adj[u_idx, v_idx] = max(weight, 0.01)
        
        row_sums = adj.sum(dim=1, keepdim=True)
        zero_rows = (row_sums.squeeze() == 0)
        adj[zero_rows, :] = 1.0 / n
        
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums
        
        return adj, nodes, node_to_idx
    
    def random_walks_gpu_with_coherence(self, start_idx, adj, nodes, num_walks, walk_length):
        n = adj.shape[0]
        paths = torch.zeros((num_walks, walk_length + 1), dtype=torch.long, device=self.device)
        paths[:, 0] = start_idx
        
        for step in range(walk_length):
            current = paths[:, step]
            probs = adj[current]
            
            if self.use_coherence and self.coherence_calc and step > 0:
                previous = paths[:, step - 1]
                
                for walk_idx in range(num_walks):
                    curr_node = nodes[current[walk_idx].item()]
                    prev_node = nodes[previous[walk_idx].item()]
                    
                    neighbors = list(self.kg.G.neighbors(curr_node))
                    
                    for neighbor in neighbors:
                        neighbor_idx = nodes.index(neighbor) if neighbor in nodes else -1
                        if neighbor_idx >= 0:
                            coherence = self.coherence_calc.get_coherence(prev_node, neighbor)
                            probs[walk_idx, neighbor_idx] *= (1.0 + coherence)
            
            probs = torch.clamp(probs, min=1e-10)
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            next_nodes = torch.multinomial(probs, 1).squeeze(1)
            paths[:, step + 1] = next_nodes
        
        return paths
    
    def random_walks_gpu(self, start_idx, adj, num_walks, walk_length):
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
        context_nodes = ["USER"]
        if not self.kg.search_history or len(self.kg.search_history) < 10:
            return context_nodes
        
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
        
        if current_timestamp:
            hour = current_timestamp.hour
            temporal_matches = []
            for search in self.kg.search_history[-1000:]:
                search_time = search.get('timestamp')
                if search_time and abs(search_time.hour - hour) <= 2:
                    for entity in search.get('entities', []):
                        entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                        if entity_id in self.kg.G:
                            temporal_matches.append(entity_id)
            
            if temporal_matches:
                temporal_freq = Counter(temporal_matches)
                top_temporal = temporal_freq.most_common(1)
                if top_temporal and top_temporal[0][0] not in context_nodes:
                    context_nodes.append(top_temporal[0][0])
        
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
    
    def predict_next_category(self, current_timestamp=None, use_context=True, reasoning_mode='forward'):
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor(reasoning_mode)
        
        if not use_context:
            if "USER" not in node_to_idx:
                return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
            
            user_idx = node_to_idx["USER"]
            
            if self.use_coherence:
                paths = self.random_walks_gpu_with_coherence(user_idx, adj, nodes, 
                                                            self.num_walks, self.walk_length)
            else:
                paths = self.random_walks_gpu(user_idx, adj, self.num_walks, self.walk_length)
            
            category_visits = {cat: 0 for cat in self.kg.categories}
            
            paths_cpu = paths.cpu().numpy()
            for path in paths_cpu:
                for node_idx in path:
                    node = nodes[node_idx]
                    if self.kg.G.nodes.get(node, {}).get('node_type') == 'category':
                        category_visits[node] += 1
        else:
            context_nodes = self.get_context_nodes(current_timestamp)
            
            walks_per_node = self.num_walks // len(context_nodes)
            remaining_walks = self.num_walks % len(context_nodes)
            
            category_visits = {cat: 0 for cat in self.kg.categories}
            
            for i, start_node in enumerate(context_nodes):
                if start_node not in node_to_idx:
                    continue
                
                num_walks = walks_per_node + (1 if i < remaining_walks else 0)
                start_idx = node_to_idx[start_node]
                
                if self.use_coherence:
                    paths = self.random_walks_gpu_with_coherence(start_idx, adj, nodes,
                                                                num_walks, self.walk_length)
                else:
                    paths = self.random_walks_gpu(start_idx, adj, num_walks, self.walk_length)
                
                paths_cpu = paths.cpu().numpy()
                for path in paths_cpu:
                    for node_idx in path:
                        node = nodes[node_idx]
                        if self.kg.G.nodes.get(node, {}).get('node_type') == 'category':
                            category_visits[node] += 1
        
        total_visits = sum(category_visits.values())
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        return {cat: category_visits[cat] / total_visits for cat in self.kg.categories}
    
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
    
    def update_graph_weights(self, predicted_dist, actual_categories, learning_rate=LEARNING_RATE):
        """Update with competitive learning"""
        for cat in self.kg.categories:
            predicted_prob = predicted_dist.get(cat, 0)
            actual_conf = actual_categories.get(cat, 0)
            error = actual_conf - predicted_prob
            
            if self.kg.G.has_edge("USER", cat):
                current = self.kg.G["USER"][cat]['weight']
                adjustment = learning_rate * error
                new_weight = np.clip(current + adjustment, 0.0, 1.0)
                self.kg.G["USER"][cat]['weight'] = new_weight
                
                if adjustment > 0 and self.use_competition and self.competition_manager:
                    self.competition_manager.apply_competition(cat, learning_rate)
            
            if self.kg.G.has_edge(cat, "USER"):
                current = self.kg.G[cat]["USER"]['weight']
                adjustment = learning_rate * error * 0.3
                new_weight = np.clip(current + adjustment, 0.0, 1.0)
                self.kg.G[cat]["USER"]['weight'] = new_weight
    
    def predict_from_context(self, start_nodes, num_walks_per_node=10, reasoning_mode='forward'):
        """Context-based prediction"""
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor(reasoning_mode)
        
        category_visits = {cat: 0 for cat in self.kg.categories}
        
        for start_node in start_nodes:
            if start_node not in node_to_idx:
                continue
            
            start_idx = node_to_idx[start_node]
            
            if self.use_coherence:
                paths = self.random_walks_gpu_with_coherence(start_idx, adj, nodes,
                                                            num_walks_per_node, self.walk_length)
            else:
                paths = self.random_walks_gpu(start_idx, adj, num_walks_per_node, self.walk_length)
            
            paths_cpu = paths.cpu().numpy()
            for path in paths_cpu:
                for node_idx in path:
                    node = nodes[node_idx]
                    if self.kg.G.nodes.get(node, {}).get('node_type') == 'category':
                        category_visits[node] += 1
        
        total_visits = sum(category_visits.values())
        if total_visits == 0:
            return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        return {cat: category_visits[cat] / total_visits for cat in self.kg.categories}
    
    def explain_why(self, target_node, num_walks=30):
        """Backward reasoning"""
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor('backward')
        
        if target_node not in node_to_idx:
            return {}
        
        target_idx = node_to_idx[target_node]
        
        if self.use_coherence:
            paths = self.random_walks_gpu_with_coherence(target_idx, adj, nodes,
                                                        num_walks, self.walk_length)
        else:
            paths = self.random_walks_gpu(target_idx, adj, num_walks, self.walk_length)
        
        cause_visits = {}
        
        paths_cpu = paths.cpu().numpy()
        for path in paths_cpu:
            for node_idx in path:
                node = nodes[node_idx]
                if self.kg.G.nodes.get(node, {}).get('node_type') in ['category', 'entity']:
                    cause_visits[node] = cause_visits.get(node, 0) + 1
        
        total = sum(cause_visits.values())
        if total == 0:
            return {}
        
        sorted_causes = sorted(cause_visits.items(), key=lambda x: x[1], reverse=True)[:10]
        return {node: count/total for node, count in sorted_causes}
    
    def analyze_and_tag_edge_types(self):
        if self.edge_type_detector:
            self.edge_type_detector.analyze_search_history()
            return self.edge_type_detector.tag_all_edges()
        return {}