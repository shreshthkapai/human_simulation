"""
Main Script for Graph-Based Persona Generation
Orchestrates the entire pipeline from data loading to visualization
"""
import warnings
warnings.filterwarnings('ignore')

from utils import load_search_data, prepare_dataframe, print_data_summary
from training_pipeline import full_production_run, load_checkpoint
from analysis_tools import (cluster_transitions, detect_sub_personas, 
                            get_category_connections, generate_narrative)
from visualization import plot_learning_trajectory, plot_comprehensive_analysis
from config import FINAL_CHECKPOINT


def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("GRAPH-BASED PERSONA GENERATION FROM SEARCH HISTORY")
    print("="*70)
    print()
    
    # Step 1: Load and prepare data
    print("[1/5] Loading search data...")
    search_data = load_search_data()
    df = prepare_dataframe(search_data)
    print_data_summary(search_data, df)
    print()
    
    # Step 2: Run training (or resume from checkpoint)
    print("[2/5] Starting training pipeline...")
    print("This will process all searches and build the knowledge graph.")
    print("Progress is automatically saved. Press Ctrl+C to pause safely.")
    print()
    
    # Uncomment to resume from checkpoint:
    # kg, predictor, detector = full_production_run(df, resume_from='checkpoint_X.pkl')
    
    kg, predictor, detector = full_production_run(df)
    print()
    
    # Step 3: Load final results and analyze
    print("[3/5] Analyzing results...")
    
    # If you want to load a saved checkpoint instead of training:
    # kg, predictor, detector, _ = load_checkpoint(FINAL_CHECKPOINT)
    
    # Cluster transitions into life events
    transition_clusters = cluster_transitions(detector.transitions)
    print(f"Clustered {len(detector.transitions)} transitions into {len(transition_clusters)} life events")
    
    # Detect sub-personas
    community_info, communities = detect_sub_personas(kg)
    print(f"Detected {len(community_info)} sub-persona communities")
    
    # Get category connections
    cat_connections = get_category_connections(kg)
    print()
    
    # Step 4: Generate narrative
    print("[4/5] Generating user narrative...")
    narrative = generate_narrative(kg, predictor, detector, transition_clusters, 
                                   community_info, cat_connections)
    print(narrative)
    
    # Save narrative to file
    with open('user_narrative.txt', 'w') as f:
        f.write(narrative)
    print("Narrative saved to user_narrative.txt")
    print()
    
    # Step 5: Create visualizations
    print("[5/5] Creating visualizations...")
    plot_learning_trajectory(predictor, detector)
    plot_comprehensive_analysis(kg, predictor, detector, transition_clusters, 
                                community_info, cat_connections)
    print()
    
    print("="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print()
    print("Generated files:")
    print("  - checkpoint_FINAL.pkl (trained model)")
    print("  - user_narrative.txt (analysis report)")
    print("  - learning_trajectory.png (learning plots)")
    print("  - complete_analysis.png (comprehensive visualization)")
    print()
    print("To load saved results for further analysis:")
    print("  from training_pipeline import load_checkpoint")
    print("  kg, predictor, detector, _ = load_checkpoint('checkpoint_FINAL.pkl')")


if __name__ == "__main__":
    main()