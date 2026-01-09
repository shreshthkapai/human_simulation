"""
Standalone Analysis Script - WITH EDGE TYPE ANALYSIS
Load saved checkpoints and perform detailed analysis including edge typing
"""
import warnings
warnings.filterwarnings('ignore')

from training_pipeline import load_checkpoint
from analysis_tools import (cluster_transitions, detect_sub_personas, 
                            get_category_connections, generate_narrative)
from visualization import plot_learning_trajectory, plot_comprehensive_analysis
from config import FINAL_CHECKPOINT, USE_EDGE_TYPES
import numpy as np


def analyze_checkpoint(checkpoint_file=FINAL_CHECKPOINT):
    """
    Load a checkpoint and perform comprehensive analysis
    NOW WITH EDGE TYPE DETECTION
    """
    
    print("="*70)
    print("ANALYZING SAVED CHECKPOINT (WITH EDGE TYPES)")
    print("="*70)
    print()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_file}")
    kg, predictor, detector, last_idx = load_checkpoint(checkpoint_file)
    
    if kg is None:
        print(f"Error: Could not load checkpoint {checkpoint_file}")
        return
    
    print(f"Loaded successfully!")
    print()
    
    # Basic statistics
    print("BASIC STATISTICS:")
    print(f"  Total searches processed: {len(kg.search_history):,}")
    print(f"  Graph nodes: {kg.G.number_of_nodes():,}")
    print(f"  Graph edges: {kg.G.number_of_edges():,}")
    print(f"  Surprise scores collected: {len(predictor.surprise_scores):,}")
    print(f"  Transitions detected: {len(detector.transitions):,}")
    print()
    
    # EDGE TYPE ANALYSIS (NEW!)
    if USE_EDGE_TYPES and hasattr(predictor, 'edge_type_detector'):
        print("ANALYZING EDGE TYPES...")
        print("This analyzes temporal patterns to infer relationship types")
        print("(This is a one-time analysis, ~2-5 minutes)")
        print()
        
        type_distribution = predictor.analyze_and_tag_edge_types()
        
        print("\nEDGE TYPE DISTRIBUTION:")
        total_edges = sum(type_distribution.values())
        for edge_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_edges * 100) if total_edges > 0 else 0
            print(f"  {edge_type}: {count:,} edges ({pct:.1f}%)")
        print()
        
        # Show examples of each type
        print("EXAMPLE EDGES BY TYPE:")
        examples_found = {t: 0 for t in ['causal', 'associative', 'temporal_correlation']}
        
        for u, v, data in kg.G.edges(data=True):
            edge_type = data.get('relationship_type', 'unknown')
            
            if edge_type in examples_found and examples_found[edge_type] < 2:
                u_label = kg.G.nodes[u].get('label', u) if u != 'USER' else u
                v_label = kg.G.nodes[v].get('label', v) if 'entity' in v else v
                strength = data.get('type_strength', 0)
                
                print(f"  {edge_type}: {u_label} → {v_label} (strength: {strength:.2f})")
                examples_found[edge_type] += 1
            
            if all(count >= 2 for count in examples_found.values()):
                break
        print()
    
    # Transition clustering
    print("CLUSTERING TRANSITIONS...")
    transition_clusters = cluster_transitions(detector.transitions)
    
    print(f"  Raw transitions: {len(detector.transitions):,}")
    print(f"  Clustered into: {len(transition_clusters):,} life events")
    print()
    
    # Show top events
    sorted_clusters = sorted(transition_clusters, key=lambda x: len(x['transitions']), reverse=True)
    print("TOP 5 MAJOR LIFE EVENTS:")
    for i, cluster in enumerate(sorted_clusters[:5], 1):
        duration = (cluster['end_time'] - cluster['start_time']).days
        print(f"  {i}. {cluster['start_time'].date()} → {cluster['end_time'].date()}")
        print(f"     Duration: {duration} days, Transitions: {len(cluster['transitions'])}")
    print()
    
    # Sub-persona detection
    print("DETECTING SUB-PERSONAS...")
    community_info, communities = detect_sub_personas(kg)
    
    print(f"  Detected {len(community_info)} sub-persona communities")
    print()
    
    print("TOP SUB-PERSONAS:")
    for i, comm in enumerate(community_info[:5], 1):
        print(f"  {i}. Sub-Persona {comm['id']}")
        print(f"     Size: {comm['size']} entities")
        print(f"     Total mentions: {comm['mentions']}")
        
        if comm['categories']:
            top_cats = comm['categories'].most_common(3)
            print(f"     Top categories: {', '.join([f'{cat}({count})' for cat, count in top_cats])}")
        
        print(f"     Key entities:")
        for entity, mentions in comm['top_entities'][:3]:
            entity_label = kg.G.nodes[entity].get('label', entity)
            print(f"       - {entity_label} ({mentions} mentions)")
        print()
    
    # Category distribution
    print("CATEGORY DISTRIBUTION:")
    cat_connections = get_category_connections(kg)
    total_connections = sum(cat_connections.values())
    
    for cat, count in cat_connections.most_common():
        pct = (count / total_connections) * 100
        print(f"  {cat}: {count:,} connections ({pct:.1f}%)")
    print()
    
    # Learning metrics
    print("LEARNING METRICS:")
    if len(predictor.surprise_scores) >= 1000:
        first_1k = np.mean(predictor.surprise_scores[:1000])
        last_1k = np.mean(predictor.surprise_scores[-1000:])
        improvement = (first_1k - last_1k) / first_1k * 100
        print(f"  First 1000 searches avg surprise: {first_1k:.3f}")
        print(f"  Last 1000 searches avg surprise: {last_1k:.3f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    print(f"  Overall avg surprise: {np.mean(predictor.surprise_scores):.3f}")
    print(f"  Transition rate: {len(detector.transitions)/len(kg.search_history)*100:.2f}%")
    print()
    
    # Generate narrative
    print("GENERATING NARRATIVE...")
    narrative = generate_narrative(kg, predictor, detector, transition_clusters, 
                                   community_info, cat_connections)
    
    with open('user_narrative.txt', 'w', encoding='utf-8') as f:
        f.write(narrative)
    print("  Saved to user_narrative.txt")
    print()
    
    # Create visualizations
    print("CREATING VISUALIZATIONS...")
    plot_learning_trajectory(predictor, detector)
    plot_comprehensive_analysis(kg, predictor, detector, transition_clusters, 
                                community_info, cat_connections)
    print()
    
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return kg, predictor, detector, transition_clusters, community_info


if __name__ == "__main__":
    kg, predictor, detector, clusters, personas = analyze_checkpoint()