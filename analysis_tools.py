"""
Analysis Tools for Post-Processing
Transition clustering, sub-persona detection, and narrative generation
"""
import numpy as np
import networkx as nx
from datetime import timedelta
from collections import Counter, defaultdict
from networkx.algorithms import community
from config import MAX_GAP_DAYS, MIN_CLUSTER_SIZE, TOP_N_ENTITIES


def cluster_transitions(transitions, max_gap_days=MAX_GAP_DAYS):
    """
    Group transitions that are close in time into periods
    """
    if not transitions:
        return []
    
    sorted_trans = sorted(transitions, key=lambda x: x['timestamp'])
    
    clusters = []
    current_cluster = {
        'transitions': [sorted_trans[0]],
        'start_time': sorted_trans[0]['timestamp'],
        'end_time': sorted_trans[0]['timestamp'],
        'start_index': sorted_trans[0]['search_index'],
        'end_index': sorted_trans[0]['search_index']
    }
    
    for trans in sorted_trans[1:]:
        time_gap = (trans['timestamp'] - current_cluster['end_time']).days
        
        if time_gap <= max_gap_days:
            current_cluster['transitions'].append(trans)
            current_cluster['end_time'] = trans['timestamp']
            current_cluster['end_index'] = trans['search_index']
        else:
            clusters.append(current_cluster)
            current_cluster = {
                'transitions': [trans],
                'start_time': trans['timestamp'],
                'end_time': trans['timestamp'],
                'start_index': trans['search_index'],
                'end_index': trans['search_index']
            }
    
    clusters.append(current_cluster)
    
    return clusters


def detect_sub_personas(kg):
    """
    Find communities/clusters in the knowledge graph
    Returns community information
    """
    # Remove USER and category nodes
    G_entities = kg.G.copy()
    G_entities.remove_node("USER")
    
    category_nodes = [node for node, data in G_entities.nodes(data=True) 
                      if data.get('node_type') == 'category']
    G_entities.remove_nodes_from(category_nodes)
    
    # Convert to undirected
    G_undirected = G_entities.to_undirected()
    
    # Find communities
    communities = community.greedy_modularity_communities(G_undirected, weight='weight')
    
    community_info = []
    
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:10], 1):
        entities = list(comm)
        
        if len(entities) < MIN_CLUSTER_SIZE:
            continue
        
        total_mentions = sum(kg.G.nodes[e].get('mention_count', 0) for e in entities if e in kg.G)
        
        # Find connected categories
        connected_cats = Counter()
        for entity in entities:
            if entity in kg.G:
                for neighbor in kg.G.neighbors(entity):
                    if kg.G.nodes[neighbor].get('node_type') == 'category':
                        connected_cats[neighbor] += 1
        
        # Get top entities
        entity_mentions = [(e, kg.G.nodes[e].get('mention_count', 0)) 
                           for e in entities if e in kg.G]
        top_entities = sorted(entity_mentions, key=lambda x: x[1], reverse=True)[:TOP_N_ENTITIES]
        
        community_info.append({
            'id': i,
            'size': len(entities),
            'mentions': total_mentions,
            'categories': connected_cats,
            'top_entities': top_entities
        })
    
    return community_info, communities


def get_category_connections(kg):
    """Get overall category distribution across all entities"""
    cat_connections = Counter()
    for node in kg.G.nodes():
        if kg.G.nodes[node].get('node_type') == 'entity':
            for neighbor in kg.G.neighbors(node):
                if kg.G.nodes[neighbor].get('node_type') == 'category':
                    cat_connections[neighbor] += 1
    return cat_connections


def generate_narrative(kg, predictor, detector, transition_clusters, community_info, cat_connections):
    """
    Generate user identity narrative from analysis
    """
    timespan_years = (kg.search_history[-1]['timestamp'] - kg.search_history[0]['timestamp']).days / 365
    first_1k_surprise = np.mean(predictor.surprise_scores[:1000])
    last_1k_surprise = np.mean(predictor.surprise_scores[-1000:])
    improvement_pct = (first_1k_surprise - last_1k_surprise) / first_1k_surprise * 100
    
    narrative = []
    narrative.append("\n" + "="*70)
    narrative.append("USER IDENTITY NARRATIVE")
    narrative.append(f"{timespan_years:.1f}-Year Behavioral Analysis")
    narrative.append("="*70 + "\n")
    
    narrative.append("OVERVIEW:")
    narrative.append(f"Analyzed {len(kg.search_history):,} searches over {timespan_years:.1f} years")
    narrative.append(f"Detected {len(detector.transitions):,} behavioral transitions")
    narrative.append(f"Identified {len(transition_clusters)} distinct life events")
    narrative.append(f"Found {len(community_info)} behavioral sub-personas")
    narrative.append(f"Model learning: {improvement_pct:.1f}% surprise reduction over time")
    narrative.append("")
    
    narrative.append("LIFE TRAJECTORY:")
    narrative.append("")
    
    sorted_clusters = sorted(transition_clusters, key=lambda x: len(x['transitions']), reverse=True)
    major_events = sorted_clusters[:5]
    
    narrative.append("The search history reveals major transitional periods:")
    narrative.append("")
    
    for i, cluster in enumerate(major_events, 1):
        duration = (cluster['end_time'] - cluster['start_time']).days
        intensity = len(cluster['transitions']) / max(duration, 1)
        
        categories_in_period = {}
        for trans in cluster['transitions']:
            for cat, conf in trans['new_categories'].items():
                categories_in_period[cat] = categories_in_period.get(cat, 0) + conf
        
        top_cat = sorted(categories_in_period.items(), key=lambda x: x[1], reverse=True)[0][0] if categories_in_period else "Unknown"
        
        narrative.append(f"{i}. {cluster['start_time'].strftime('%B %Y')} ({duration} days, {intensity:.1f} transitions/day)")
        narrative.append(f"   Dominant theme: {top_cat}")
        
        if top_cat in ['Daily_Life', 'Location'] and intensity > 2.5:
            narrative.append(f"   → High-intensity daily life changes suggest major relocation or life restructuring")
        elif top_cat in ['Technology', 'Work_Career']:
            narrative.append(f"   → Career-focused period, possibly skill development or job transition")
        elif top_cat == 'Travel':
            narrative.append(f"   → Extended travel or exploration phase")
        elif top_cat == 'Life_Transitions' and intensity > 5:
            narrative.append(f"   → Very high-intensity transition (>5/day) indicates significant life upheaval")
        
        narrative.append("")
    
    narrative.append("BEHAVIORAL PATTERNS:")
    narrative.append("")
    
    total_connections = sum(cat_connections.values())
    dominant_interests = cat_connections.most_common(3)
    
    narrative.append("Interest composition:")
    for cat, count in dominant_interests:
        pct = (count / total_connections) * 100
        narrative.append(f"  • {cat}: {pct:.1f}% of entity connections")
    
    narrative.append(f"\nThe knowledge graph shows a relatively balanced profile with")
    narrative.append(f"primary focus on {dominant_interests[0][0]} and {dominant_interests[1][0]}.")
    
    if len(community_info) >= 3:
        narrative.append(f"\nBehavioral clustering identified {len(community_info)} distinct sub-personas,")
        narrative.append(f"suggesting a multi-faceted identity spanning professional, personal, and")
        narrative.append(f"location-based contexts.")
    
    narrative.append("\n" + "-"*70)
    narrative.append("MODEL PERFORMANCE:")
    narrative.append("-"*70 + "\n")
    
    narrative.append(f"Graph structure: {kg.G.number_of_nodes():,} nodes, {kg.G.number_of_edges():,} edges")
    narrative.append(f"Edge density: {kg.G.number_of_edges()/kg.G.number_of_nodes():.2f} edges/node")
    narrative.append(f"Prediction accuracy improved {improvement_pct:.1f}% from early to late searches")
    narrative.append(f"Transition detection rate: {len(detector.transitions)/len(kg.search_history)*100:.1f}% of searches")
    narrative.append("="*70 + "\n")
    
    return "\n".join(narrative)