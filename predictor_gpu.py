"""
GPU Predictor - FIXED VERSION
✅ User→Category edges now decay properly
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
        
        self.coherence_calc = CoherenceCalculator(knowledge_graph) if use_coherence else None
        self.edge_type_detector = EdgeTypeDetector(knowledge_graph) if use_edge_types else None
        self.competition_manager = CompetitionManager(knowledge_graph) if use_competition else None
        
        features = [f for f, enabled in [("Coherence", use_coherence), 
                    ("Edge Types", use_edge_types), ("Competition", use_competition)] if enabled]
        print(f"GraphPredictorGPU on {self.device} ({' + '.join(features) if features else 'Basic'})")
    
    def graph_to_adjacency_tensor(self, reasoning_mode='forward'):
        nodes = list(self.kg.G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)
        
        adj = torch.zeros((n, n), device=self.device)
        
        for u, v, data in self.kg.G.edges(data=True):
            weight = data.get('weight', 0.1)
            
            if self.use_edge_types and self.edge_type_detector:
                edge_type = data.get('relationship_type', 'unknown')
                weight *= self.edge_type_detector.get_edge_type_weight_multiplier(edge_type, reasoning_mode)
            
            adj[node_to_idx[u], node_to_idx[v]] = max(weight, 0.01)
        
        row_sums = adj.sum(dim=1, keepdim=True)
        adj[row_sums.squeeze() == 0] = 1.0 / n
        adj = adj / adj.sum(dim=1, keepdim=True)
        
        return adj, nodes, node_to_idx
    
    def random_walks_gpu_with_coherence(self, start_idx, adj, nodes, num_walks, walk_length):
        n = adj.shape[0]
        paths = torch.zeros((num_walks, walk_length + 1), dtype=torch.long, device=self.device)
        paths[:, 0] = start_idx
        path_coherence_scores = torch.zeros((num_walks, walk_length), device=self.device)
        
        for step in range(walk_length):
            current = paths[:, step]
            probs = adj[current].clone()
            
            if self.use_coherence and self.coherence_calc and step > 0:
                previous = paths[:, step - 1]
                
                for walk_idx in range(num_walks):
                    curr_node = nodes[current[walk_idx].item()]
                    prev_node = nodes[previous[walk_idx].item()]
                    
                    for neighbor in self.kg.G.neighbors(curr_node):
                        neighbor_idx = nodes.index(neighbor) if neighbor in nodes else -1
                        if neighbor_idx >= 0:
                            coherence = self.coherence_calc.get_coherence(prev_node, neighbor)
                            probs[walk_idx, neighbor_idx] *= (1.0 + coherence)
                            path_coherence_scores[walk_idx, step] = coherence
            
            probs = torch.clamp(probs, min=1e-10)
            probs = probs / probs.sum(dim=1, keepdim=True)
            paths[:, step + 1] = torch.multinomial(probs, 1).squeeze(1)
        
        return paths, path_coherence_scores.mean(dim=1)
    
    def random_walks_gpu(self, start_idx, adj, num_walks, walk_length):
        n = adj.shape[0]
        paths = torch.zeros((num_walks, walk_length + 1), dtype=torch.long, device=self.device)
        paths[:, 0] = start_idx
        
        for step in range(walk_length):
            current = paths[:, step]
            probs = torch.clamp(adj[current], min=1e-10)
            probs = probs / probs.sum(dim=1, keepdim=True)
            paths[:, step + 1] = torch.multinomial(probs, 1).squeeze(1)
        
        return paths, torch.ones(num_walks, device=self.device) * 0.5
    
    def get_context_nodes(self, current_timestamp=None, num_context_nodes=3):
        context_nodes = ["USER"]
        if not self.kg.search_history or len(self.kg.search_history) < 10:
            return context_nodes
        
        recent_entities = []
        for search in self.kg.search_history[-5:]:
            for entity in search.get('entities', []):
                entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                if entity_id in self.kg.G:
                    score = 1.0 + np.log1p(self.kg.G.nodes[entity_id].get('mention_count', 1))
                    recent_entities.append((entity_id, score))
        
        recent_entities.sort(key=lambda x: x[1], reverse=True)
        context_nodes.extend([e for e, _ in recent_entities[:2] if e not in context_nodes])
        
        if current_timestamp:
            hour = current_timestamp.hour
            temporal_matches = []
            for search in self.kg.search_history[-1000:]:
                if search.get('timestamp') and abs(search['timestamp'].hour - hour) <= 2:
                    for entity in search.get('entities', []):
                        entity_id = f"entity_{entity.lower().replace(' ', '_')}"
                        if entity_id in self.kg.G:
                            temporal_matches.append(entity_id)
            
            if temporal_matches:
                top_temporal = Counter(temporal_matches).most_common(1)[0][0]
                if top_temporal not in context_nodes:
                    context_nodes.append(top_temporal)
        
        if len(self.surprise_scores) > 25:
            recent_surprise = np.mean(self.surprise_scores[-25:])
            baseline_surprise = np.mean(self.surprise_scores[:-25])
            
            if recent_surprise > baseline_surprise * 1.3 and self.kg.search_history:
                last_categories = self.kg.search_history[-1].get('categories', {})
                if last_categories:
                    top_cat = max(last_categories.items(), key=lambda x: x[1])[0]
                    if top_cat not in context_nodes:
                        context_nodes.append(top_cat)
        
        return context_nodes[:num_context_nodes + 1]
    
    def predict_next_category(self, current_timestamp=None, use_context=True, reasoning_mode='forward'):
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor(reasoning_mode)
        category_visits = {cat: 0.0 for cat in self.kg.categories}
        
        if not use_context:
            if "USER" not in node_to_idx:
                return {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
            
            paths, qualities = (self.random_walks_gpu_with_coherence if self.use_coherence 
                               else self.random_walks_gpu)(node_to_idx["USER"], adj, nodes, 
                                                          self.num_walks, self.walk_length)
            
            for path, quality in zip(paths.cpu().numpy(), qualities.cpu().numpy()):
                for node_idx in path:
                    node = nodes[node_idx]
                    if self.kg.G.nodes.get(node, {}).get('node_type') == 'category':
                        category_visits[node] += (1.0 + quality)
        else:
            context_nodes = self.get_context_nodes(current_timestamp)
            walks_per_node = self.num_walks // len(context_nodes)
            
            for i, start_node in enumerate(context_nodes):
                if start_node not in node_to_idx:
                    continue
                
                num_walks = walks_per_node + (1 if i < self.num_walks % len(context_nodes) else 0)
                paths, qualities = (self.random_walks_gpu_with_coherence if self.use_coherence 
                                   else self.random_walks_gpu)(node_to_idx[start_node], adj, nodes,
                                                              num_walks, self.walk_length)
                
                for path, quality in zip(paths.cpu().numpy(), qualities.cpu().numpy()):
                    for node_idx in path:
                        node = nodes[node_idx]
                        if self.kg.G.nodes.get(node, {}).get('node_type') == 'category':
                            category_visits[node] += (1.0 + quality)
        
        total_visits = sum(category_visits.values())
        return {cat: category_visits[cat] / total_visits for cat in self.kg.categories} if total_visits > 0 else {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
    
    def calculate_surprise(self, predicted_dist, actual_categories):
        actual_dist = {cat: 0.0 for cat in self.kg.categories}
        total_confidence = sum(actual_categories.values())
        
        if total_confidence > 0:
            for cat, conf in actual_categories.items():
                if cat in actual_dist:
                    actual_dist[cat] = conf / total_confidence
        else:
            actual_dist = {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
        
        epsilon = 1e-10
        predicted_array = np.array([predicted_dist.get(cat, epsilon) + epsilon for cat in self.kg.categories])
        actual_array = np.array([actual_dist.get(cat, epsilon) + epsilon for cat in self.kg.categories])
        
        predicted_array /= predicted_array.sum()
        actual_array /= actual_array.sum()
        
        return np.sum(actual_array * np.log(actual_array / predicted_array))
    
    def update_graph_weights(self, predicted_dist, actual_categories, learning_rate=LEARNING_RATE):
        for cat in self.kg.categories:
            error = actual_categories.get(cat, 0) - predicted_dist.get(cat, 0)
            
            if self.kg.G.has_edge("USER", cat):
                current = self.kg.G["USER"][cat]['weight']
                new_weight = np.clip(current + learning_rate * error, 0.0, 1.0)
                self.kg.G["USER"][cat]['weight'] = new_weight
                
                if error > 0 and self.use_competition and self.competition_manager:
                    self.competition_manager.apply_competition(cat, learning_rate)
            
            if self.kg.G.has_edge(cat, "USER"):
                current = self.kg.G[cat]["USER"]['weight']
                self.kg.G[cat]["USER"]['weight'] = np.clip(current + learning_rate * error * 0.3, 0.0, 1.0)
    
    def predict_from_context(self, start_nodes, num_walks_per_node=10, reasoning_mode='forward'):
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor(reasoning_mode)
        category_visits = {cat: 0.0 for cat in self.kg.categories}
        
        for start_node in start_nodes:
            if start_node not in node_to_idx:
                continue
            
            paths, qualities = (self.random_walks_gpu_with_coherence if self.use_coherence 
                               else self.random_walks_gpu)(node_to_idx[start_node], adj, nodes,
                                                          num_walks_per_node, self.walk_length)
            
            for path, quality in zip(paths.cpu().numpy(), qualities.cpu().numpy()):
                for node_idx in path:
                    node = nodes[node_idx]
                    if self.kg.G.nodes.get(node, {}).get('node_type') == 'category':
                        category_visits[node] += (1.0 + quality)
        
        total_visits = sum(category_visits.values())
        return {cat: category_visits[cat] / total_visits for cat in self.kg.categories} if total_visits > 0 else {cat: 1.0/len(self.kg.categories) for cat in self.kg.categories}
    
    def explain_why(self, target_node, num_walks=30):
        adj, nodes, node_to_idx = self.graph_to_adjacency_tensor('backward')
        
        if target_node not in node_to_idx:
            return {}
        
        paths, qualities = (self.random_walks_gpu_with_coherence if self.use_coherence 
                           else self.random_walks_gpu)(node_to_idx[target_node], adj, nodes,
                                                      num_walks, self.walk_length)
        
        cause_visits = {}
        for path, quality in zip(paths.cpu().numpy(), qualities.cpu().numpy()):
            for node_idx in path:
                node = nodes[node_idx]
                if self.kg.G.nodes.get(node, {}).get('node_type') in ['category', 'entity']:
                    cause_visits[node] = cause_visits.get(node, 0.0) + (1.0 + quality)
        
        total = sum(cause_visits.values())
        if total == 0:
            return {}
        
        return {node: count/total for node, count in sorted(cause_visits.items(), key=lambda x: x[1], reverse=True)[:10]}
    
    def rebuild_coherence_metadata(self):
        if self.coherence_calc:
            self.coherence_calc.rebuild_metadata()
    
    def rebuild_competition_matrix(self):
        if self.competition_manager:
            self.competition_manager.rebuild_competition_matrix()
    
    def apply_temporal_decay(self, current_timestamp, decay_rate=0.0005):
        """
        Apply decay to changing interests and associations
        ✅ FIXED: Now includes User→Category edges
        """
        # ✅ EXPANDED: Include User→Category edges
        DECAYABLE_EDGE_TYPES = {'interested_in', 'co_occurs', 'suggests'}
        
        edges_affected = 0
        
        for u, v, data in self.kg.G.edges(data=True):
            edge_type = data.get('edge_type')
            
            if edge_type not in DECAYABLE_EDGE_TYPES:
                continue
            
            # ✅ REMOVED: No longer skip User→Category edges
            
            last_updated = data.get('last_updated')
            if not last_updated:
                continue
            
            days_since = (current_timestamp - last_updated).days
            
            # ✅ NEW: Slower decay for User→Category (more stable)
            if edge_type == 'interested_in' and u == 'USER':
                effective_decay_rate = decay_rate * 0.5
            else:
                effective_decay_rate = decay_rate
            
            decay_factor = np.exp(-effective_decay_rate * days_since)
            new_weight = data['weight'] * decay_factor
            
            if new_weight < 0.05:
                data['weight'] = 0.01
                data['historical_peak'] = max(data.get('historical_peak', 0), data['weight'] / decay_factor)
                edges_affected += 1
            else:
                data['weight'] = max(new_weight, 0.01)
                data['historical_peak'] = max(data.get('historical_peak', 0), data['weight'])
        
        return edges_affected
    
    def analyze_and_tag_edge_types(self):
        if self.edge_type_detector:
            self.edge_type_detector.analyze_search_history()
            return self.edge_type_detector.tag_all_edges()
        return {}