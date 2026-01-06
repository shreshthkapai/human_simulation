"""
Visualization Tools
Creates comprehensive analysis plots
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from scipy.ndimage import uniform_filter1d
from config import SMOOTHING_WINDOW, FIGURE_DPI, CHECKPOINT_EVERY


def plot_learning_trajectory(predictor, detector, output_file='learning_trajectory.png'):
    """Plot learning trajectory analysis"""
    surprise_scores = np.array(predictor.surprise_scores)
    surprise_smooth = uniform_filter1d(surprise_scores, size=SMOOTHING_WINDOW, mode='nearest')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Raw surprise over time
    axes[0, 0].scatter(range(len(surprise_scores)), surprise_scores, alpha=0.1, s=1, color='blue')
    axes[0, 0].plot(range(len(surprise_smooth)), surprise_smooth, color='red', linewidth=2, label='Rolling avg')
    axes[0, 0].axhline(y=4, color='orange', linestyle='--', label='Transition threshold')
    axes[0, 0].set_xlabel('Search Index')
    axes[0, 0].set_ylabel('Surprise (KL Divergence)')
    axes[0, 0].set_title('Learning Trajectory: Surprise Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Surprise distribution by period
    periods = 5
    period_size = len(surprise_scores) // periods
    period_surprises = [surprise_scores[i*period_size:(i+1)*period_size] for i in range(periods)]
    period_labels = [f'{i*20}%-{(i+1)*20}%' for i in range(periods)]
    
    axes[0, 1].violinplot(period_surprises, positions=range(periods), showmeans=True)
    axes[0, 1].set_xticks(range(periods))
    axes[0, 1].set_xticklabels(period_labels)
    axes[0, 1].set_xlabel('Progress Through Dataset')
    axes[0, 1].set_ylabel('Surprise Distribution')
    axes[0, 1].set_title('Surprise Distribution by Period')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative average
    cumulative_avg = np.cumsum(surprise_scores) / np.arange(1, len(surprise_scores) + 1)
    axes[1, 0].plot(cumulative_avg, color='green', linewidth=2)
    axes[1, 0].set_xlabel('Search Index')
    axes[1, 0].set_ylabel('Cumulative Average Surprise')
    axes[1, 0].set_title('Model Convergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Transition density
    transition_indices = [t['search_index'] for t in detector.transitions]
    bins = 50
    hist, bin_edges = np.histogram(transition_indices, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    axes[1, 1].bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0])*0.8, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Search Index')
    axes[1, 1].set_ylabel('Transition Count')
    axes[1, 1].set_title('Transition Density Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Learning trajectory saved to {output_file}")


def plot_comprehensive_analysis(kg, predictor, detector, transition_clusters, community_info, 
                                cat_connections, output_file='complete_analysis.png'):
    """Create comprehensive visualization with all analysis"""
    
    surprise_scores = np.array(predictor.surprise_scores)
    surprise_smooth = uniform_filter1d(surprise_scores, size=SMOOTHING_WINDOW, mode='nearest')
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Learning Trajectory
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(surprise_smooth, color='#2E86AB', linewidth=2, label='Surprise (rolling avg)')
    ax1.axhline(y=4.0, color='#A23B72', linestyle='--', linewidth=1.5, label='Transition threshold')
    ax1.fill_between(range(len(surprise_smooth)), surprise_smooth, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Search Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Surprise (KL Divergence)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Learning Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Transition Timeline
    ax2 = fig.add_subplot(gs[1, :])
    transition_indices = [t['search_index'] for t in detector.transitions]
    sorted_clusters = sorted(transition_clusters, key=lambda x: len(x['transitions']), reverse=True)
    
    for i, cluster in enumerate(sorted_clusters[:5], 1):
        start_idx = cluster['start_index']
        end_idx = cluster['end_index']
        ax2.axvspan(start_idx, end_idx, alpha=0.3, color=f'C{i-1}', 
                   label=f'Event {i}: {cluster["start_time"].strftime("%b %Y")}')
    
    ax2.scatter(transition_indices, [1]*len(transition_indices), alpha=0.3, s=10, color='red')
    ax2.set_xlabel('Search Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Transitions', fontsize=12, fontweight='bold')
    ax2.set_title('Major Life Events Timeline', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_ylim(0.5, 1.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Category Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    cats = [c for c, _ in cat_connections.most_common()]
    counts = [c for _, c in cat_connections.most_common()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    ax3.barh(cats, counts, color=colors)
    ax3.set_xlabel('Entity Connections', fontsize=10, fontweight='bold')
    ax3.set_title('Interest Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Sub-Persona Sizes
    ax4 = fig.add_subplot(gs[2, 1])
    persona_labels = [f'P{i+1}' for i in range(len(community_info))]
    persona_sizes = [c['size'] for c in community_info]
    ax4.bar(persona_labels, persona_sizes, color='#F18F01')
    ax4.set_xlabel('Sub-Persona', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Entity Count', fontsize=10, fontweight='bold')
    ax4.set_title('Sub-Persona Community Sizes', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Graph Growth from Checkpoints
    ax5 = fig.add_subplot(gs[2, 2])
    
    checkpoint_files = []
    for i in range(100):
        filename = f'checkpoint_{i}.pkl'
        if os.path.exists(filename):
            checkpoint_files.append((i, filename))
    
    if os.path.exists('checkpoint_FINAL.pkl'):
        max_checkpoint = max([c[0] for c in checkpoint_files]) if checkpoint_files else 0
        checkpoint_files.append((max_checkpoint + 1, 'checkpoint_FINAL.pkl'))
    
    checkpoint_files.sort()
    
    searches_processed = []
    nodes_count = []
    edges_count = []
    
    for checkpoint_num, filename in checkpoint_files:
        try:
            with open(filename, 'rb') as f:
                checkpoint = pickle.load(f)
            
            kg_checkpoint = checkpoint['kg']
            
            if 'FINAL' in filename:
                search_idx = len(kg_checkpoint.search_history)
            else:
                search_idx = (checkpoint_num + 1) * CHECKPOINT_EVERY
            
            searches_processed.append(search_idx)
            nodes_count.append(kg_checkpoint.G.number_of_nodes())
            edges_count.append(kg_checkpoint.G.number_of_edges())
        
        except Exception:
            continue
    
    if len(searches_processed) > 0:
        ax5.plot(searches_processed, nodes_count, 'o-', label='Nodes', linewidth=2, markersize=4, color='#06A77D')
        ax5.plot(searches_processed, edges_count, 's-', label='Edges', linewidth=2, markersize=4, color='#D4AA00')
        ax5.set_xlabel('Searches Processed', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax5.set_title('Knowledge Graph Growth', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
    
    timespan = (kg.search_history[-1]['timestamp'] - kg.search_history[0]['timestamp']).days / 365
    plt.suptitle(f'Longitudinal Behavioral Modeling via Predictive Knowledge Graphs\n{timespan:.1f}-Year Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comprehensive analysis saved to {output_file}")