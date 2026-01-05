"""
Knowledge Graph Infrastructure
Probabilistic graph that accumulates weak signals from search behavior
"""
import networkx as nx
import numpy as np
from collections import Counter


class UserKnowledgeGraph:
    """
    Dynamic knowledge graph that learns user identity through search patterns
    """
    
    def __init__(self, categories):
        self.G = nx.DiGraph()
        self.categories = categories
        
        # Add user node at center
        self.G.add_node("USER", node_type="user")
        
        # Add category nodes
        for cat in categories:
            self.G.add_node(cat, node_type="category")
            self.G.add_edge("USER", cat, weight=0.0, edge_type="interest")
        
        # Track history for learning
        self.search_history = []
        
    def add_search_event(self, timestamp, query, entities, categories, attributes):
        """Process a search event and update graph"""
        event = {
            'timestamp': timestamp,
            'query': query,
            'entities': entities,
            'categories': categories,
            'attributes': attributes
        }
        self.search_history.append(event)
        
        # Add entity nodes and edges
        for entity in entities:
            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
            
            if not self.G.has_node(entity_id):
                self.G.add_node(entity_id, 
                               node_type="entity", 
                               label=entity,
                               first_seen=timestamp,
                               mention_count=0)
            
            # Update mention count
            self.G.nodes[entity_id]['mention_count'] += 1
            self.G.nodes[entity_id]['last_seen'] = timestamp
            
            # Connect entity to relevant categories
            for cat, confidence in categories.items():
                if confidence > 0.3:
                    if self.G.has_edge(entity_id, cat):
                        current_weight = self.G[entity_id][cat]['weight']
                        new_weight = min(current_weight + confidence * 0.2, 1.0)
                        self.G[entity_id][cat]['weight'] = new_weight
                    else:
                        self.G.add_edge(entity_id, cat, 
                                       weight=confidence * 0.5,
                                       edge_type="belongs_to",
                                       created=timestamp)
            
            # Connect entity to USER
            if self.G.has_edge("USER", entity_id):
                current_weight = self.G["USER"][entity_id]['weight']
                self.G["USER"][entity_id]['weight'] = min(current_weight + 0.15, 1.0)
            else:
                self.G.add_edge("USER", entity_id,
                               weight=0.3,
                               edge_type="interested_in",
                               created=timestamp)
        
        # Update category interest weights
        for cat, confidence in categories.items():
            if self.G.has_edge("USER", cat):
                current_weight = self.G["USER"][cat]['weight']
                new_weight = min(current_weight + confidence * 0.1, 1.0)
                self.G["USER"][cat]['weight'] = new_weight
                self.G["USER"][cat]['last_updated'] = timestamp
    
    def get_category_distribution(self):
        """Get current distribution of user interests across categories"""
        dist = {}
        for cat in self.categories:
            if self.G.has_edge("USER", cat):
                dist[cat] = self.G["USER"][cat]['weight']
            else:
                dist[cat] = 0.0
        
        # Normalize
        total = sum(dist.values())
        if total > 0:
            dist = {k: v/total for k, v in dist.items()}
        
        return dist
    
    def get_top_entities(self, category=None, top_n=10):
        """Get most important entities, optionally filtered by category"""
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
    
    def __repr__(self):
        return f"UserKnowledgeGraph(nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()}, searches={len(self.search_history)})"