"""
Knowledge Graph Infrastructure - FIXED
✅ FIX: Simplified rejoin logic without "dormant" flags
✅ KEEP: historical_peak for intelligent revival
"""
import networkx as nx
import numpy as np
from collections import Counter


class UserKnowledgeGraph:
    """
    True bidirectional knowledge graph with entity-entity co-occurrence
    """
    
    def __init__(self, categories):
        self.G = nx.DiGraph()
        self.categories = categories
        
        self.G.add_node("USER", node_type="user")
        
        for cat in categories:
            self.G.add_node(cat, node_type="category")
            self.G.add_edge("USER", cat, weight=0.0, edge_type="interested_in")
            self.G.add_edge(cat, "USER", weight=0.0, edge_type="defines")
        
        self.search_history = []
        
    def _should_create_entity_edge(self, entity_A_id, entity_B_id):
        """Decide if entity pair deserves an edge"""
        mentions_A = self.G.nodes[entity_A_id].get('mention_count', 0)
        mentions_B = self.G.nodes[entity_B_id].get('mention_count', 0)
        
        if mentions_A < 3 or mentions_B < 3:
            return False
        
        if mentions_A + mentions_B < 10:
            return False
        
        return True
    
    def _add_entity_cooccurrence_edges(self, entities, timestamp):
        """
        Add selective entity-entity co-occurrence edges with SIMPLIFIED rejoin logic
        ✅ FIX: No "dormant" flags, just check weight < 0.05
        """
        entity_ids = [f"entity_{e.lower().replace(' ', '_')}" for e in entities 
                     if f"entity_{e.lower().replace(' ', '_')}" in self.G]
        
        sorted_entities = sorted(
            entity_ids,
            key=lambda e: self.G.nodes[e].get('mention_count', 0),
            reverse=True
        )[:5]
        
        for i, entity_A_id in enumerate(sorted_entities):
            for entity_B_id in sorted_entities[i+1:]:
                if not self._should_create_entity_edge(entity_A_id, entity_B_id):
                    continue
                
                # ✅ SIMPLIFIED: Forward edge A→B
                if self.G.has_edge(entity_A_id, entity_B_id):
                    edge_data = self.G[entity_A_id][entity_B_id]
                    current = edge_data['weight']
                    
                    # If weight is very low, use historical_peak for smart revival
                    if current < 0.05:
                        historical_peak = edge_data.get('historical_peak', 0.3)
                        edge_data['weight'] = min(historical_peak * 0.5, 0.4)
                    else:
                        edge_data['weight'] = min(current + 0.1, 1.0)
                    
                    edge_data['last_updated'] = timestamp
                else:
                    self.G.add_edge(entity_A_id, entity_B_id, 
                                   weight=0.3, 
                                   edge_type='co_occurs',
                                   created=timestamp,
                                   last_updated=timestamp)
                
                # Backward edge B→A
                if self.G.has_edge(entity_B_id, entity_A_id):
                    edge_data = self.G[entity_B_id][entity_A_id]
                    current = edge_data['weight']
                    
                    # If weight is very low, use historical_peak for smart revival
                    if current < 0.05:
                        historical_peak = edge_data.get('historical_peak', 0.3)
                        edge_data['weight'] = min(historical_peak * 0.5, 0.4)
                    else:
                        edge_data['weight'] = min(current + 0.1, 1.0)
                    
                    edge_data['last_updated'] = timestamp
                else:
                    self.G.add_edge(entity_B_id, entity_A_id,
                                   weight=0.3,
                                   edge_type='co_occurs',
                                   created=timestamp,
                                   last_updated=timestamp)
    
    def add_search_event(self, timestamp, query, entities, categories, attributes):
        event = {
            'timestamp': timestamp,
            'query': query,
            'entities': entities,
            'entity_ids': [f"entity_{e.lower().replace(' ', '_')}" for e in entities],
            'categories': categories,
            'attributes': attributes
        }
        self.search_history.append(event)
        
        for entity in entities:
            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
            
            if not self.G.has_node(entity_id):
                self.G.add_node(entity_id, 
                               node_type="entity", 
                               label=entity,
                               first_seen=timestamp,
                               mention_count=0)
            
            self.G.nodes[entity_id]['mention_count'] += 1
            self.G.nodes[entity_id]['last_seen'] = timestamp
            
            # Entity→Category edges (definitional, no decay)
            for cat, confidence in categories.items():
                if confidence > 0.3:
                    if self.G.has_edge(entity_id, cat):
                        current = self.G[entity_id][cat]['weight']
                        self.G[entity_id][cat]['weight'] = min(current + confidence * 0.25, 1.0)
                    else:
                        self.G.add_edge(entity_id, cat, 
                                       weight=confidence * 0.6,
                                       edge_type="belongs_to",
                                       created=timestamp)
                    
                    # Category→Entity edges
                    if self.G.has_edge(cat, entity_id):
                        edge_data = self.G[cat][entity_id]
                        current = edge_data['weight']
                        
                        # If weight is very low, revive smartly
                        if current < 0.05:
                            historical_peak = edge_data.get('historical_peak', 0.35)
                            mention_boost = min(self.G.nodes[entity_id]['mention_count'] / 100.0, 0.5)
                            edge_data['weight'] = min(historical_peak * 0.6 + mention_boost * 0.05, 0.5)
                        else:
                            mention_boost = min(self.G.nodes[entity_id]['mention_count'] / 100.0, 0.5)
                            edge_data['weight'] = min(current + confidence * 0.15 + mention_boost * 0.05, 1.0)
                        
                        edge_data['last_updated'] = timestamp
                    else:
                        self.G.add_edge(cat, entity_id,
                                       weight=confidence * 0.35,
                                       edge_type="suggests",
                                       created=timestamp,
                                       last_updated=timestamp)
            
            # User→Entity edges
            if self.G.has_edge("USER", entity_id):
                edge_data = self.G["USER"][entity_id]
                current = edge_data['weight']
                
                # If weight is very low, revive smartly
                if current < 0.05:
                    historical_peak = edge_data.get('historical_peak', 0.35)
                    edge_data['weight'] = min(historical_peak * 0.6, 0.5)
                else:
                    edge_data['weight'] = min(current + 0.15, 1.0)
                
                edge_data['last_updated'] = timestamp
            else:
                self.G.add_edge("USER", entity_id,
                               weight=0.35,
                               edge_type="interested_in",
                               created=timestamp,
                               last_updated=timestamp)
            
            # Entity→User edges (characterization, no decay)
            if self.G.has_edge(entity_id, "USER"):
                current = self.G[entity_id]["USER"]['weight']
                uniqueness = 1.0 / (1.0 + self.G.nodes[entity_id]['mention_count'] / 50.0)
                self.G[entity_id]["USER"]['weight'] = min(current + 0.08 * uniqueness, 1.0)
            else:
                self.G.add_edge(entity_id, "USER",
                               weight=0.25,
                               edge_type="characterizes",
                               created=timestamp)
        
        self._add_entity_cooccurrence_edges(entities, timestamp)
        
        # User→Category edges (handled by competition, timestamp for tracking only)
        for cat, confidence in categories.items():
            if self.G.has_edge("USER", cat):
                current = self.G["USER"][cat]['weight']
                self.G["USER"][cat]['weight'] = min(current + confidence * 0.12, 1.0)
                self.G["USER"][cat]['last_updated'] = timestamp
            
            if self.G.has_edge(cat, "USER"):
                current = self.G[cat]["USER"]['weight']
                self.G[cat]["USER"]['weight'] = min(current + confidence * 0.06, 1.0)
    
    def get_category_distribution(self):
        dist = {}
        for cat in self.categories:
            if self.G.has_edge("USER", cat):
                dist[cat] = self.G["USER"][cat]['weight']
            else:
                dist[cat] = 0.0
        
        total = sum(dist.values())
        if total > 0:
            dist = {k: v/total for k, v in dist.items()}
        
        return dist
    
    def get_top_entities(self, category=None, top_n=10):
        entities = []
        
        for node, data in self.G.nodes(data=True):
            if data.get('node_type') == 'entity':
                user_edge_weight = self.G["USER"][node]['weight'] if self.G.has_edge("USER", node) else 0
                mention_count = data.get('mention_count', 0)
                importance = user_edge_weight * (1 + np.log1p(mention_count))
                
                if category:
                    if self.G.has_edge(node, category):
                        cat_weight = self.G[node][category]['weight']
                        importance *= cat_weight
                    else:
                        continue
                
                entities.append({
                    'entity': data['label'],
                    'importance': importance,
                    'mentions': mention_count
                })
        
        entities.sort(key=lambda x: x['importance'], reverse=True)
        return entities[:top_n]
    
    def get_salient_entities(self, category, top_n=10):
        entities = []
        
        if category in self.G:
            for neighbor in self.G.neighbors(category):
                node_data = self.G.nodes.get(neighbor, {})
                if node_data.get('node_type') == 'entity':
                    salience = self.G[category][neighbor]['weight']
                    mention_count = node_data.get('mention_count', 0)
                    
                    entities.append({
                        'entity': node_data.get('label', neighbor),
                        'salience': salience,
                        'mentions': mention_count
                    })
        
        entities.sort(key=lambda x: x['salience'], reverse=True)
        return entities[:top_n]
    
    def get_defining_traits(self, top_n=10):
        traits = []
        
        for node in self.G.predecessors("USER"):
            if self.G.has_edge(node, "USER"):
                defining_weight = self.G[node]["USER"]['weight']
                node_data = self.G.nodes.get(node, {})
                
                traits.append({
                    'trait': node_data.get('label', node),
                    'type': node_data.get('node_type', 'unknown'),
                    'defining_strength': defining_weight
                })
        
        traits.sort(key=lambda x: x['defining_strength'], reverse=True)
        return traits[:top_n]
    
    def get_cooccurring_entities(self, entity_id, top_k=10):
        """Get entities that co-occur with given entity"""
        if entity_id not in self.G:
            return []
        
        cooccurring = []
        for neighbor in self.G.neighbors(entity_id):
            if self.G.nodes.get(neighbor, {}).get('node_type') == 'entity':
                weight = self.G[entity_id][neighbor].get('weight', 0)
                label = self.G.nodes[neighbor].get('label', neighbor)
                cooccurring.append((label, weight))
        
        cooccurring.sort(key=lambda x: x[1], reverse=True)
        return cooccurring[:top_k]
    
    def __repr__(self):
        return f"UserKnowledgeGraph(nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()}, searches={len(self.search_history)})"