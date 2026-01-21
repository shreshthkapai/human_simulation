"""
Personalization Module - Phase 2 of Two-Phase System
Adjusts generic LLM scores using user's graph

Smooth sigmoid blend weight instead of step function
"""
import numpy as np


def personalize_scores(llm_scores, entities, kg, blend_weight=0.5):
    """
    Adjust generic LLM scores based on user's graph
    
    Args:
        llm_scores: Generic scores from LLM {'Travel': 0.85, ...}
        entities: Extracted entities ['Paris', 'hotels']
        kg: User's knowledge graph
        blend_weight: How much to trust graph vs LLM (0.0-1.0)
    
    Returns:
        Personalized scores {'Travel': 0.92, ...}
    """
    
    # Handle cold start (graph too small)
    if len(kg.search_history) < 100:
        return llm_scores
    
    personalized = {}
    
    for category in kg.categories:
        llm_score = llm_scores.get(category, 0.0)
        
        if llm_score < 0.1:
            personalized[category] = llm_score
            continue
        
        adjustment = 0.0
        
        # 1. User affinity (does this user care about this category?)
        if kg.G.has_edge('USER', category):
            user_weight = kg.G['USER'][category]['weight']
            adjustment += user_weight * 0.4
        
        # 2. Entity evidence (do entities belong to this category for this user?)
        entity_contributions = []
        for entity in entities:
            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
            if kg.G.has_node(entity_id) and kg.G.has_edge(entity_id, category):
                entity_weight = kg.G[entity_id][category]['weight']
                entity_contributions.append(entity_weight)
        
        if entity_contributions:
            adjustment += np.mean(entity_contributions) * 0.4
        
        # 3. Recent context (has user searched this category recently?)
        recent_searches = kg.search_history[-10:]
        recent_category_count = sum(
            1 for search in recent_searches 
            if category in search.get('categories', {})
        )
        
        if recent_category_count > 0:
            context_boost = min(recent_category_count / 10.0, 0.3)
            adjustment += context_boost
        
        # Blend LLM score with graph adjustment
        # Early training: trust LLM more (blend_weight low)
        # Late training: trust graph more (blend_weight high)
        adaptive_blend = min(len(kg.search_history) / 2000, blend_weight)
        
        personalized[category] = (
            llm_score * (1 - adaptive_blend) + 
            adjustment * adaptive_blend
        )
    
    # Normalize to ensure scores are reasonable
    total = sum(personalized.values())
    if total > 0:
        personalized = {k: v / total for k, v in personalized.items()}
    
    return personalized


def calculate_blend_weight(graph_size):
    """
    Smooth sigmoid blend weight instead of step function
    
    Calculate how much to trust graph vs LLM based on training progress
    Uses sigmoid for smooth transition:
    
    Early training (< 500): ~0.2 (trust LLM 80%)
    Mid training (500-2000): smooth ramp
    Late training (> 2000): ~0.6 (trust graph 60%)
    
    Formula: blend_weight = 0.2 + 0.4 * sigmoid((size - 1250) / 400)
    """
    # Sigmoid parameters
    midpoint = 1250  # Inflection point (halfway between 500 and 2000)
    steepness = 400  # Controls smoothness (higher = smoother)
    
    # Sigmoid function: 1 / (1 + e^(-x))
    x = (graph_size - midpoint) / steepness
    sigmoid_value = 1.0 / (1.0 + np.exp(-x))
    
    # Map sigmoid [0, 1] to blend_weight [0.2, 0.6]
    min_blend = 0.2  # Early training
    max_blend = 0.6  # Late training
    blend_weight = min_blend + (max_blend - min_blend) * sigmoid_value
    
    return blend_weight
