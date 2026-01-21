"""
Main Script for Graph-Based Persona Generation
Orchestrates the entire pipeline from data loading to visualization
"""
import warnings
import os
import glob
import re
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
    print("Progress is automatically saved. Press Ctrl+C to pause safely.")
    
    # Auto-resume logic: find the latest checkpoint file
    checkpoint_files = glob.glob("checkpoint_*.pkl")
    # Exclude 'FINAL' or 'interrupt' strings to find numeric ones
    latest_checkpoint = None
    latest_checkpoint_num = -1
    
    if checkpoint_files:
        # Extract numbers and pick highest
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+).pkl', f)
            if match:
                num = int(match.group(1))
                if num > latest_checkpoint_num:
                    latest_checkpoint_num = num
                    latest_checkpoint = f
            
    if latest_checkpoint:
        print(f"  [AUTO-RESUME] Found existing checkpoint: {latest_checkpoint}")
        try:
            # Capture all 5 return values including checkpoint_num
            kg, predictor, detector, last_idx, checkpoint_num = load_checkpoint(latest_checkpoint)
            
            # Pass checkpoint_num to full_production_run
            kg, predictor, detector = full_production_run(
                df, 
                resume_from=latest_checkpoint,
                start_checkpoint_num=checkpoint_num  # Pass it through!
            )
        except Exception as e:
            print(f"  [ERROR] Failed to load checkpoint {latest_checkpoint}: {e}")
            print("  Starting fresh run instead...")
            kg, predictor, detector = full_production_run(df)
    else:
        kg, predictor, detector = full_production_run(df)
    
    print()
    
    # Step 3: Load final results and analyze
    print("[3/5] Analyzing results...")
    
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
    with open('user_narrative.txt', 'w', encoding='utf-8') as f:
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
    print("  kg, predictor, detector, start_idx, num = load_checkpoint('checkpoint_FINAL.pkl')")


if __name__ == "__main__":
    main()