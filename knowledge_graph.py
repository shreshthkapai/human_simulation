"""
Knowledge Graph Infrastructure - COMPLETE BIDIRECTIONAL
Every relationship creates TWO edges with INDEPENDENT weights
Captures both classification (entity→category) AND salience (category→entity)
"""
import networkx as nx
import numpy as np
from collections import Counter


class UserKnowledgeGraph:
    """
    True bidirectional knowledge graph
    - Forward weight: "How much does X belong to Y?"
    - Backward weight: "How much does Y make me think of X?"
    """
    
    def __init__(self, categories):
        self.G = nx.DiGraph()
        self.categories = categories
        
        # Add user node at center
        self.G.add_node("USER", node_type="user")
        
        # Add category nodes with BIDIRECTIONAL edges
        for cat in categories:
            self.G.add_node(cat, node_type="category")
            
            # USER → category: "How interested am I in this category?"
            self.G.add_edge("USER", cat, 
                          weight=0.0,
                          edge_type="interested_in")
            
            # category → USER: "How much does this category define me?"
            self.G.add_edge(cat, "USER",
                          weight=0.0,
                          edge_type="defines")
        
        # Track history for learning
        self.search_history = []
        
    def add_search_event(self, timestamp, query, entities, categories, attributes):
        """Process a search event and update graph with TRUE bidirectional logic"""
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
            
            # Connect entity ↔ category BIDIRECTIONALLY with INDEPENDENT weights
            for cat, confidence in categories.items():
                if confidence > 0.3:
                    
                    # FORWARD: entity → category
                    # Meaning: "How strongly does this entity belong to this category?"
                    # Example: entity_paris → Travel = 0.9 (Paris is very much a travel thing)
                    if self.G.has_edge(entity_id, cat):
                        current = self.G[entity_id][cat]['weight']
                        # Strong update for forward (classification)
                        self.G[entity_id][cat]['weight'] = min(current + confidence * 0.25, 1.0)
                    else:
                        self.G.add_edge(entity_id, cat, 
                                       weight=confidence * 0.6,  # Start higher for forward
                                       edge_type="belongs_to",
                                       created=timestamp)
                    
                    # BACKWARD: category → entity
                    # Meaning: "How much does thinking about this category evoke this entity?"
                    # Example: Travel → entity_paris = 0.7 (thinking travel often leads to Paris)
                    #          Travel → entity_hometown = 0.2 (thinking travel rarely leads to hometown)
                    if self.G.has_edge(cat, entity_id):
                        current = self.G[cat][entity_id]['weight']
                        # Context-dependent update for salience
                        # If entity mentioned a lot, it becomes more salient
                        mention_boost = min(self.G.nodes[entity_id]['mention_count'] / 100.0, 0.5)
                        self.G[cat][entity_id]['weight'] = min(current + confidence * 0.15 + mention_boost * 0.05, 1.0)
                    else:
                        self.G.add_edge(cat, entity_id,
                                       weight=confidence * 0.35,  # Start lower for backward
                                       edge_type="suggests",
                                       created=timestamp)
            
            # Connect USER ↔ entity BIDIRECTIONALLY with INDEPENDENT weights
            
            # FORWARD: USER → entity
            # Meaning: "How interested am I in this entity?"
            if self.G.has_edge("USER", entity_id):
                current = self.G["USER"][entity_id]['weight']
                self.G["USER"][entity_id]['weight'] = min(current + 0.15, 1.0)
            else:
                self.G.add_edge("USER", entity_id,
                               weight=0.35,
                               edge_type="interested_in",
                               created=timestamp)
            
            # BACKWARD: entity → USER  
            # Meaning: "How much does this entity characterize/define me uniquely?"
            # Rare entities get higher weight (more defining)
            if self.G.has_edge(entity_id, "USER"):
                current = self.G[entity_id]["USER"]['weight']
                # Boost for rare/unique entities
                uniqueness = 1.0 / (1.0 + self.G.nodes[entity_id]['mention_count'] / 50.0)
                self.G[entity_id]["USER"]['weight'] = min(current + 0.08 * uniqueness, 1.0)
            else:
                self.G.add_edge(entity_id, "USER",
                               weight=0.25,
                               edge_type="characterizes",
                               created=timestamp)
        
        # Update USER ↔ category weights BIDIRECTIONALLY
        for cat, confidence in categories.items():
            
            # FORWARD: USER → category
            # Meaning: "How interested am I in this category?"
            if self.G.has_edge("USER", cat):
                current = self.G["USER"][cat]['weight']
                self.G["USER"][cat]['weight'] = min(current + confidence * 0.12, 1.0)
                self.G["USER"][cat]['last_updated'] = timestamp
            
            # BACKWARD: category → USER
            # Meaning: "How much does this category define my identity?"
            # Strong/consistent interests define more
            if self.G.has_edge(cat, "USER"):
                current = self.G[cat]["USER"]['weight']
                # Define-ness grows slower (need consistent interest)
                self.G[cat]["USER"]['weight'] = min(current + confidence * 0.06, 1.0)
    
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
                # Use forward edge (USER → entity = interest)
                user_edge_weight = self.G["USER"][node]['weight'] if self.G.has_edge("USER", node) else 0
                mention_count = data.get('mention_count', 0)
                importance = user_edge_weight * (1 + np.log1p(mention_count))
                
                if category:
                    if self.G.has_edge(node, category):
                        # Use forward edge (entity → category = belongs to)
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
        """
        NEW: Get entities that come to mind when thinking about a category
        Uses BACKWARD edges: category → entity (salience/generation)
        
        Example: "When I think about Travel, what entities pop into my head?"
        """
        entities = []
        
        if category in self.G:
            for neighbor in self.G.neighbors(category):
                node_data = self.G.nodes.get(neighbor, {})
                if node_data.get('node_type') == 'entity':
                    # Use BACKWARD weight (category → entity = salience)
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
        """
        NEW: What defines me uniquely?
        Uses BACKWARD edges: entity/category → USER (characterization)
        """
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
    
    def __repr__(self):
        return f"UserKnowledgeGraph(nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()}, searches={len(self.search_history)})"