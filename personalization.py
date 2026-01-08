"""
Personalization Module - Phase 2 of Two-Phase System
Adjusts generic LLM scores using user's graph
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
    Calculate how much to trust graph vs LLM based on training progress
    
    Early training (< 500 searches): Trust LLM more (0.2)
    Mid training (500-2000): Gradually increase (0.2 → 0.5)
    Late training (> 2000): Trust graph more (0.5)
    """
    if graph_size < 500:
        return 0.2
    elif graph_size < 2000:
        return 0.2 + (graph_size - 500) / 1500 * 0.3
    else:
        return 0.5