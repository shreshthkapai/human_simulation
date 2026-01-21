"""
Comprehensive Text-Based Analysis Script
Outputs ALL essential information to text files for detailed analysis
"""
import numpy as np
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json


def load_checkpoint(filename='checkpoint_FINAL.pkl'):
    """Load the final checkpoint"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['kg'], data['predictor'], data['detector']


def analyze_surprise_distribution(predictor):
    """Detailed surprise score analysis"""
    scores = np.array(predictor.surprise_scores)
    
    analysis = []
    analysis.append("="*80)
    analysis.append("SURPRISE SCORE DISTRIBUTION ANALYSIS")
    analysis.append("="*80)
    analysis.append("")
    
    # Overall statistics
    analysis.append("OVERALL STATISTICS:")
    analysis.append(f"  Total scores: {len(scores):,}")
    analysis.append(f"  Mean: {np.mean(scores):.4f}")
    analysis.append(f"  Median: {np.median(scores):.4f}")
    analysis.append(f"  Std Dev: {np.std(scores):.4f}")
    analysis.append(f"  Min: {np.min(scores):.4f}")
    analysis.append(f"  Max: {np.max(scores):.4f}")
    analysis.append("")
    
    # Percentiles
    analysis.append("PERCENTILE DISTRIBUTION:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(scores, p)
        analysis.append(f"  {p}th percentile: {val:.4f}")
    analysis.append("")
    
    # Learning trajectory (by quintiles)
    analysis.append("LEARNING TRAJECTORY (by quintile):")
    quintile_size = len(scores) // 5
    for i in range(5):
        start = i * quintile_size
        end = (i + 1) * quintile_size if i < 4 else len(scores)
        quintile_scores = scores[start:end]
        analysis.append(f"  Quintile {i+1} ({start:,}-{end:,}):")
        analysis.append(f"    Mean: {np.mean(quintile_scores):.4f}")
        analysis.append(f"    Std: {np.std(quintile_scores):.4f}")
    analysis.append("")
    
    # Top surprise spikes (potential transitions)
    analysis.append("TOP 50 SURPRISE SPIKES (Potential Transitions):")
    top_indices = np.argsort(scores)[-50:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        analysis.append(f"  {rank}. Index {idx:,}: {scores[idx]:.4f}")
    analysis.append("")
    
    # Adaptive threshold calculation
    mean_surprise = np.mean(scores)
    std_surprise = np.std(scores)
    analysis.append("ADAPTIVE THRESHOLD ANALYSIS:")
    for sigma in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        threshold = mean_surprise + sigma * std_surprise
        above_threshold = np.sum(scores > threshold)
        percentage = (above_threshold / len(scores)) * 100
        analysis.append(f"  σ={sigma}: threshold={threshold:.4f}, {above_threshold:,} scores above ({percentage:.2f}%)")
    analysis.append("")
    
    return "\n".join(analysis)


def analyze_graph_structure(kg):
    """Detailed graph structure analysis"""
    analysis = []
    analysis.append("="*80)
    analysis.append("KNOWLEDGE GRAPH STRUCTURE")
    analysis.append("="*80)
    analysis.append("")
    
    # Basic stats
    analysis.append("BASIC STATISTICS:")
    analysis.append(f"  Total nodes: {kg.G.number_of_nodes():,}")
    analysis.append(f"  Total edges: {kg.G.number_of_edges():,}")
    analysis.append(f"  Avg degree: {kg.G.number_of_edges() / kg.G.number_of_nodes():.2f}")
    analysis.append("")
    
    # Node types
    analysis.append("NODE TYPE BREAKDOWN:")
    node_types = Counter(data.get('node_type', 'unknown') for _, data in kg.G.nodes(data=True))
    for ntype, count in node_types.most_common():
        analysis.append(f"  {ntype}: {count:,}")
    analysis.append("")
    
    # Edge types
    analysis.append("EDGE TYPE BREAKDOWN:")
    edge_types = Counter(data.get('edge_type', 'unknown') for _, _, data in kg.G.edges(data=True))
    for etype, count in edge_types.most_common():
        analysis.append(f"  {etype}: {count:,}")
    analysis.append("")
    
    # Relationship types (if edge typing was used)
    analysis.append("RELATIONSHIP TYPE BREAKDOWN:")
    rel_types = Counter(data.get('relationship_type', 'unknown') for _, _, data in kg.G.edges(data=True))
    for rtype, count in rel_types.most_common():
        percentage = (count / kg.G.number_of_edges()) * 100
        analysis.append(f"  {rtype}: {count:,} ({percentage:.1f}%)")
    analysis.append("")
    
    return "\n".join(analysis)


def analyze_data_schema(kg):
    """Document what data each node and edge type carries"""
    analysis = []
    analysis.append("="*80)
    analysis.append("DATA SCHEMA - WHAT EACH NODE/EDGE CARRIES")
    analysis.append("="*80)
    analysis.append("")
    
    analysis.append("NODE TYPES AND THEIR DATA:")
    analysis.append("")
    
    # Analyze USER node
    if "USER" in kg.G:
        user_data = kg.G.nodes["USER"]
        analysis.append("1. USER NODE:")
        analysis.append("   Type: user (singleton)")
        analysis.append("   Attributes:")
        for key, value in user_data.items():
            analysis.append(f"     - {key}: {type(value).__name__}")
        analysis.append("")
    
    # Analyze category nodes
    analysis.append("2. CATEGORY NODES:")
    analysis.append(f"   Type: category (n={len(kg.categories)})")
    analysis.append("   Categories: " + ", ".join(kg.categories))
    # Sample one category
    sample_cat = kg.categories[0]
    if sample_cat in kg.G:
        cat_data = kg.G.nodes[sample_cat]
        analysis.append("   Attributes (example from 'Fashion'):")
        for key, value in cat_data.items():
            analysis.append(f"     - {key}: {type(value).__name__}")
    analysis.append("")
    
    # Analyze entity nodes
    analysis.append("3. ENTITY NODES:")
    entity_nodes = [n for n, d in kg.G.nodes(data=True) if d.get('node_type') == 'entity']
    analysis.append(f"   Type: entity (n={len(entity_nodes):,})")
    analysis.append("   ID format: 'entity_lowercase_name_with_underscores'")
    if entity_nodes:
        sample_entity = entity_nodes[0]
        entity_data = kg.G.nodes[sample_entity]
        analysis.append(f"   Attributes (example from '{entity_data.get('label', sample_entity)}'):")
        for key, value in entity_data.items():
            val_type = type(value).__name__
            if key == 'first_seen' or key == 'last_seen':
                val_type = f"{val_type} (datetime)"
            analysis.append(f"     - {key}: {val_type}")
            if key == 'label':
                analysis.append(f"         (human-readable name)")
            elif key == 'mention_count':
                analysis.append(f"         (how many searches mentioned this)")
            elif key == 'first_seen':
                analysis.append(f"         (timestamp of first appearance)")
            elif key == 'last_seen':
                analysis.append(f"         (timestamp of most recent appearance)")
    analysis.append("")
    
    analysis.append("-"*80)
    analysis.append("EDGE TYPES AND THEIR DATA:")
    analysis.append("")
    
    # Analyze each edge type
    edge_type_examples = defaultdict(list)
    for u, v, data in kg.G.edges(data=True):
        etype = data.get('edge_type', 'unknown')
        if len(edge_type_examples[etype]) < 3:
            edge_type_examples[etype].append((u, v, data))
    
    for i, (etype, examples) in enumerate(sorted(edge_type_examples.items()), 1):
        u, v, data = examples[0]
        
        # Get node types
        u_type = kg.G.nodes[u].get('node_type', 'unknown')
        v_type = kg.G.nodes[v].get('node_type', 'unknown')
        
        analysis.append(f"{i}. EDGE TYPE: '{etype}'")
        analysis.append(f"   Pattern: {u_type} → {v_type}")
        analysis.append(f"   Count: {sum(1 for _, _, d in kg.G.edges(data=True) if d.get('edge_type') == etype):,}")
        analysis.append("   Attributes:")
        for key, value in data.items():
            val_type = type(value).__name__
            if key == 'created' or key == 'last_updated':
                val_type = f"{val_type} (datetime)"
            analysis.append(f"     - {key}: {val_type}")
            
            # Add explanations
            if key == 'weight':
                analysis.append(f"         (strength of connection, 0.0-1.0)")
            elif key == 'edge_type':
                analysis.append(f"         (semantic meaning of edge)")
            elif key == 'relationship_type':
                analysis.append(f"         (temporal: causal/associative/temporal_correlation)")
            elif key == 'type_strength':
                analysis.append(f"         (confidence in relationship_type)")
            elif key == 'created':
                analysis.append(f"         (when edge first appeared)")
            elif key == 'last_updated':
                analysis.append(f"         (when edge last strengthened)")
            elif key == 'historical_peak':
                analysis.append(f"         (highest weight ever achieved)")
        
        # Show example
        u_label = kg.G.nodes[u].get('label', u) if u != 'USER' else 'USER'
        v_label = kg.G.nodes[v].get('label', v) if v_type == 'entity' else v
        analysis.append(f"   Example: {u_label} → {v_label}")
        analysis.append("")
    
    analysis.append("-"*80)
    analysis.append("EDGE TYPE SEMANTICS:")
    analysis.append("")
    
    semantics = {
        'interested_in': 'USER → category: User shows interest in this category',
        'defines': 'category → USER: Category helps define user identity',
        'belongs_to': 'entity → category: Entity is classified under category',
        'suggests': 'category → entity: Category suggests this entity is relevant',
        'co_occurs': 'entity → entity: Entities appear together in searches',
        'characterizes': 'entity → USER: Entity reveals something about user',
    }
    
    for etype, description in semantics.items():
        count = sum(1 for _, _, d in kg.G.edges(data=True) if d.get('edge_type') == etype)
        if count > 0:
            analysis.append(f"  • {etype}: {description}")
            analysis.append(f"    ({count:,} edges)")
            analysis.append("")
    
    analysis.append("-"*80)
    analysis.append("RELATIONSHIP TYPE SEMANTICS (Temporal Patterns):")
    analysis.append("")
    
    rel_semantics = {
        'causal': 'A consistently appears BEFORE B (sequential dependency)',
        'associative': 'A and B appear simultaneously (co-occurrence)',
        'temporal_correlation': 'A and B appear at similar times but not together',
        'unknown': 'Insufficient data to determine relationship type',
    }
    
    for rtype, description in rel_semantics.items():
        count = sum(1 for _, _, d in kg.G.edges(data=True) if d.get('relationship_type') == rtype)
        if count > 0:
            percentage = (count / kg.G.number_of_edges()) * 100
            analysis.append(f"  • {rtype}: {description}")
            analysis.append(f"    ({count:,} edges, {percentage:.1f}%)")
            analysis.append("")
    
    return "\n".join(analysis)


def analyze_categories(kg):
    """Detailed category analysis"""
    analysis = []
    analysis.append("="*80)
    analysis.append("CATEGORY ANALYSIS")
    analysis.append("="*80)
    analysis.append("")
    
    # User-Category weights
    analysis.append("USER → CATEGORY WEIGHTS (Interest Strength):")
    user_cats = []
    for cat in kg.categories:
        if kg.G.has_edge("USER", cat):
            weight = kg.G["USER"][cat]['weight']
            user_cats.append((cat, weight))
    
    user_cats.sort(key=lambda x: x[1], reverse=True)
    for cat, weight in user_cats:
        analysis.append(f"  {cat:20s}: {weight:.4f}")
    analysis.append("")
    
    # Category-User weights
    analysis.append("CATEGORY → USER WEIGHTS (Defining Strength):")
    cat_users = []
    for cat in kg.categories:
        if kg.G.has_edge(cat, "USER"):
            weight = kg.G[cat]["USER"]['weight']
            cat_users.append((cat, weight))
    
    cat_users.sort(key=lambda x: x[1], reverse=True)
    for cat, weight in cat_users:
        analysis.append(f"  {cat:20s}: {weight:.4f}")
    analysis.append("")
    
    # Category occurrence in searches
    analysis.append("CATEGORY OCCURRENCE IN SEARCHES:")
    cat_occurrences = Counter()
    for search in kg.search_history:
        for cat in search.get('categories', {}).keys():
            cat_occurrences[cat] += 1
    
    for cat, count in cat_occurrences.most_common():
        percentage = (count / len(kg.search_history)) * 100
        analysis.append(f"  {cat:20s}: {count:,} searches ({percentage:.1f}%)")
    analysis.append("")
    
    return "\n".join(analysis)


def analyze_entities(kg, top_n=100):
    """Detailed entity analysis"""
    analysis = []
    analysis.append("="*80)
    analysis.append("ENTITY ANALYSIS")
    analysis.append("="*80)
    analysis.append("")
    
    # Collect all entities
    entities = []
    for node, data in kg.G.nodes(data=True):
        if data.get('node_type') == 'entity':
            label = data.get('label', node)
            mentions = data.get('mention_count', 0)
            
            # Get user connection weight
            user_weight = 0
            if kg.G.has_edge("USER", node):
                user_weight = kg.G["USER"][node]['weight']
            
            # Get categories
            connected_cats = []
            for neighbor in kg.G.neighbors(node):
                if kg.G.nodes.get(neighbor, {}).get('node_type') == 'category':
                    cat_weight = kg.G[node][neighbor]['weight']
                    connected_cats.append((neighbor, cat_weight))
            
            entities.append({
                'id': node,
                'label': label,
                'mentions': mentions,
                'user_weight': user_weight,
                'categories': sorted(connected_cats, key=lambda x: x[1], reverse=True)
            })
    
    # Sort by mentions
    entities.sort(key=lambda x: x['mentions'], reverse=True)
    
    analysis.append(f"TOP {top_n} ENTITIES BY MENTION COUNT:")
    analysis.append("")
    
    for i, entity in enumerate(entities[:top_n], 1):
        analysis.append(f"{i}. {entity['label']}")
        analysis.append(f"   Mentions: {entity['mentions']:,}")
        analysis.append(f"   User weight: {entity['user_weight']:.4f}")
        if entity['categories']:
            top_cats = entity['categories'][:3]
            cat_str = ", ".join([f"{cat}({w:.2f})" for cat, w in top_cats])
            analysis.append(f"   Top categories: {cat_str}")
        analysis.append("")
    
    return "\n".join(analysis)


def analyze_temporal_patterns(kg):
    """Temporal pattern analysis"""
    analysis = []
    analysis.append("="*80)
    analysis.append("TEMPORAL PATTERNS")
    analysis.append("="*80)
    analysis.append("")
    
    # Extract timestamps
    timestamps = [s['timestamp'] for s in kg.search_history]
    
    # Overall timespan
    first = min(timestamps)
    last = max(timestamps)
    timespan_days = (last - first).days
    
    analysis.append("TIMESPAN:")
    analysis.append(f"  First search: {first.strftime('%Y-%m-%d %H:%M:%S')}")
    analysis.append(f"  Last search: {last.strftime('%Y-%m-%d %H:%M:%S')}")
    analysis.append(f"  Total days: {timespan_days:,}")
    analysis.append(f"  Total years: {timespan_days / 365:.2f}")
    analysis.append("")
    
    # Searches per year
    analysis.append("SEARCHES PER YEAR:")
    year_counts = Counter(ts.year for ts in timestamps)
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        analysis.append(f"  {year}: {count:,}")
    analysis.append("")
    
    # Searches per month
    analysis.append("SEARCHES PER MONTH (last 12 months):")
    month_counts = Counter(ts.strftime('%Y-%m') for ts in timestamps)
    for month in sorted(month_counts.keys())[-12:]:
        count = month_counts[month]
        analysis.append(f"  {month}: {count:,}")
    analysis.append("")
    
    # Hour distribution
    analysis.append("SEARCHES BY HOUR OF DAY:")
    hour_counts = Counter(ts.hour for ts in timestamps)
    for hour in range(24):
        count = hour_counts.get(hour, 0)
        bar = '█' * (count // 100)
        analysis.append(f"  {hour:02d}:00 | {count:5,} {bar}")
    analysis.append("")
    
    # Day of week
    analysis.append("SEARCHES BY DAY OF WEEK:")
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = Counter(ts.weekday() for ts in timestamps)
    for dow in range(7):
        count = dow_counts.get(dow, 0)
        percentage = (count / len(timestamps)) * 100
        analysis.append(f"  {dow_names[dow]:9s}: {count:,} ({percentage:.1f}%)")
    analysis.append("")
    
    return "\n".join(analysis)


def analyze_sub_personas(kg):
    """Sub-persona community analysis"""
    from networkx.algorithms import community
    
    analysis = []
    analysis.append("="*80)
    analysis.append("SUB-PERSONA COMMUNITIES")
    analysis.append("="*80)
    analysis.append("")
    
    # Create entity-only graph
    G_entities = kg.G.copy()
    G_entities.remove_node("USER")
    category_nodes = [node for node, data in G_entities.nodes(data=True) 
                      if data.get('node_type') == 'category']
    G_entities.remove_nodes_from(category_nodes)
    
    # Find communities
    G_undirected = G_entities.to_undirected()
    communities = community.greedy_modularity_communities(G_undirected, weight='weight')
    
    analysis.append(f"DETECTED {len(communities)} COMMUNITIES:")
    analysis.append("")
    
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:20], 1):
        entities = list(comm)
        
        # Get total mentions
        total_mentions = sum(kg.G.nodes[e].get('mention_count', 0) for e in entities if e in kg.G)
        
        # Find connected categories
        connected_cats = Counter()
        for entity in entities:
            if entity in kg.G:
                for neighbor in kg.G.neighbors(entity):
                    if kg.G.nodes[neighbor].get('node_type') == 'category':
                        connected_cats[neighbor] += 1
        
        # Get top entities
        entity_data = []
        for e in entities:
            if e in kg.G:
                label = kg.G.nodes[e].get('label', e)
                mentions = kg.G.nodes[e].get('mention_count', 0)
                entity_data.append((label, mentions))
        
        entity_data.sort(key=lambda x: x[1], reverse=True)
        
        analysis.append(f"COMMUNITY {i}:")
        analysis.append(f"  Size: {len(entities)} entities")
        analysis.append(f"  Total mentions: {total_mentions:,}")
        analysis.append(f"  Top categories: {', '.join([f'{cat}({count})' for cat, count in connected_cats.most_common(5)])}")
        analysis.append(f"  Top entities:")
        for entity, mentions in entity_data[:10]:
            analysis.append(f"    - {entity} ({mentions} mentions)")
        analysis.append("")
    
    return "\n".join(analysis)


def find_potential_transitions(predictor, kg, threshold_sigma=1.0):
    """Find potential transitions with custom threshold"""
    scores = np.array(predictor.surprise_scores)
    mean_surprise = np.mean(scores)
    std_surprise = np.std(scores)
    threshold = mean_surprise + threshold_sigma * std_surprise
    
    analysis = []
    analysis.append("="*80)
    analysis.append(f"POTENTIAL TRANSITIONS (σ={threshold_sigma})")
    analysis.append("="*80)
    analysis.append("")
    
    analysis.append(f"Threshold: {threshold:.4f} (mean + {threshold_sigma}σ)")
    analysis.append("")
    
    # Find windows above threshold
    window_size = 25
    transitions = []
    
    for i in range(window_size, len(scores)):
        window = scores[i-window_size:i]
        avg_surprise = np.mean(window)
        
        if avg_surprise > threshold:
            # Get search info
            search = kg.search_history[i]
            transitions.append({
                'index': i,
                'timestamp': search['timestamp'],
                'surprise': avg_surprise,
                'query': search['query'],
                'categories': search['categories']
            })
    
    analysis.append(f"FOUND {len(transitions)} POTENTIAL TRANSITION POINTS:")
    analysis.append("")
    
    # Group nearby transitions
    if transitions:
        grouped = []
        current_group = [transitions[0]]
        
        for trans in transitions[1:]:
            if trans['index'] - current_group[-1]['index'] < 100:
                current_group.append(trans)
            else:
                grouped.append(current_group)
                current_group = [trans]
        grouped.append(current_group)
        
        analysis.append(f"GROUPED INTO {len(grouped)} DISTINCT EVENTS:")
        analysis.append("")
        
        for i, group in enumerate(grouped, 1):
            start = group[0]
            end = group[-1]
            avg_surprise = np.mean([t['surprise'] for t in group])
            
            analysis.append(f"EVENT {i}:")
            analysis.append(f"  Period: {start['timestamp'].strftime('%Y-%m-%d')} to {end['timestamp'].strftime('%Y-%m-%d')}")
            analysis.append(f"  Searches: {start['index']:,} to {end['index']:,}")
            analysis.append(f"  Avg surprise: {avg_surprise:.4f}")
            analysis.append(f"  Peak surprise: {max(t['surprise'] for t in group):.4f}")
            
            # Get dominant categories
            all_cats = Counter()
            for t in group:
                for cat, conf in t['categories'].items():
                    all_cats[cat] += conf
            
            analysis.append(f"  Dominant categories: {', '.join([f'{cat}({count:.1f})' for cat, count in all_cats.most_common(3)])}")
            
            # Sample queries
            analysis.append(f"  Sample queries:")
            for t in group[:5]:
                analysis.append(f"    - {t['query'][:80]}")
            analysis.append("")
    
    return "\n".join(analysis)


def main():
    """Generate comprehensive text analysis"""
    print("Loading checkpoint...")
    kg, predictor, detector = load_checkpoint('checkpoint_FINAL.pkl')
    
    print("Generating comprehensive analysis...")
    print()
    
    # Generate all analyses
    reports = []
    
    print("  1/7 Analyzing surprise distribution...")
    reports.append(analyze_surprise_distribution(predictor))
    
    print("  2/7 Analyzing graph structure...")
    reports.append(analyze_graph_structure(kg))
    
    print("  3/7 Documenting data schema...")
    reports.append(analyze_data_schema(kg))
    
    print("  4/7 Analyzing categories...")
    reports.append(analyze_categories(kg))
    
    print("  5/7 Analyzing entities...")
    reports.append(analyze_entities(kg, top_n=100))
    
    print("  6/7 Analyzing temporal patterns...")
    reports.append(analyze_temporal_patterns(kg))
    
    print("  7/7 Analyzing sub-personas...")
    reports.append(analyze_sub_personas(kg))
    
    print("  8/8 Finding potential transitions...")
    reports.append(find_potential_transitions(predictor, kg, threshold_sigma=1.0))
    reports.append(find_potential_transitions(predictor, kg, threshold_sigma=0.8))
    
    # Combine all reports
    full_report = "\n\n".join(reports)
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'comprehensive_analysis_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print()
    print("="*80)
    print(f"ANALYSIS COMPLETE")
    print("="*80)
    print(f"Saved to: {filename}")
    print(f"Total size: {len(full_report):,} characters")
    print()
    
    # Also create a summary JSON for programmatic access
    summary = {
        'graph_stats': {
            'nodes': kg.G.number_of_nodes(),
            'edges': kg.G.number_of_edges(),
            'searches': len(kg.search_history),
        },
        'surprise_stats': {
            'mean': float(np.mean(predictor.surprise_scores)),
            'std': float(np.std(predictor.surprise_scores)),
            'min': float(np.min(predictor.surprise_scores)),
            'max': float(np.max(predictor.surprise_scores)),
        },
        'timespan': {
            'first': kg.search_history[0]['timestamp'].isoformat(),
            'last': kg.search_history[-1]['timestamp'].isoformat(),
            'days': (kg.search_history[-1]['timestamp'] - kg.search_history[0]['timestamp']).days
        }
    }
    
    json_filename = f'summary_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary JSON: {json_filename}")
    print()


if __name__ == "__main__":
    main()