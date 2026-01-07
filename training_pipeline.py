"""
Production Training Pipeline - WITH CONTEXT-SENSITIVE PREDICTION
Now passes timestamp to predictor for context-aware predictions
"""
import pickle
import os
import time
from datetime import datetime
from knowledge_graph import UserKnowledgeGraph
from predictor_hybrid import GraphPredictorHybrid
from transition_detector import TransitionDetector
from llm_client import query_llm_batch, parse_batch_response
from config import CATEGORIES, CHECKPOINT_EVERY, BATCH_SIZE, LEARNING_RATE


def save_checkpoint(kg, predictor, detector, checkpoint_num, last_processed_idx):
    """Save current state to disk"""
    checkpoint_data = {
        'kg': kg,
        'predictor': predictor,
        'detector': detector,
        'last_processed_idx': last_processed_idx,
        'timestamp': datetime.now()
    }
    
    filename = f"checkpoint_{checkpoint_num}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"    [SAVED] {filename} ({file_size:.1f}MB) at index {last_processed_idx}")
    
    return filename


def load_checkpoint(filename):
    """Load checkpoint from disk"""
    if not os.path.exists(filename):
        return None, None, None, 0
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"[RESUME] Loaded from {filename}")
    print(f"         Previously processed up to index {data['last_processed_idx']}")
    
    return data['kg'], data['predictor'], data['detector'], data['last_processed_idx']


def full_production_run(df_clean, resume_from=None, checkpoint_every=CHECKPOINT_EVERY, 
                       batch_size=BATCH_SIZE, use_context=True):
    """
    Process all searches with CONTEXT-SENSITIVE prediction
    NEW: use_context parameter enables context-aware activation
    """
    
    if resume_from and os.path.exists(resume_from):
        kg, predictor, detector, start_idx = load_checkpoint(resume_from)
        print(f"[RESUME] Continuing from search index {start_idx}")
    else:
        kg = UserKnowledgeGraph(CATEGORIES)
        predictor = GraphPredictorHybrid(kg)
        detector = TransitionDetector()
        start_idx = 0
        print("[START] Fresh run initialized")
    
    total_searches = len(df_clean)
    
    print("\n" + "="*70)
    print("FULL PRODUCTION RUN - CONTEXT-SENSITIVE MODE")
    print("="*70)
    print(f"Total searches: {total_searches:,}")
    print(f"Starting from: {start_idx:,}")
    print(f"Remaining: {total_searches - start_idx:,}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint every: {checkpoint_every} searches")
    print(f"Context-aware: {use_context}")
    print(f"Estimated time: {(total_searches - start_idx) * 2.5 / 3600:.1f} hours")
    print("="*70 + "\n")
    
    processed = 0
    skipped = 0
    checkpoint_num = 0
    batch_buffer = []
    batch_data_buffer = []
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    try:
        for idx in range(start_idx, total_searches):
            row = df_clean.iloc[idx]
            query = row['query']
            timestamp = row['timestamp']
            
            # CONTEXT-SENSITIVE PREDICTION
            # Pass timestamp so predictor can use temporal context
            predicted_dist = predictor.predict_next_category(
                current_timestamp=timestamp if use_context else None,
                use_context=use_context
            )
            
            batch_buffer.append(query)
            batch_data_buffer.append({
                'index': idx,
                'query': query,
                'timestamp': timestamp,
                'predicted_dist': predicted_dist
            })
            
            if len(batch_buffer) >= batch_size:
                response = query_llm_batch(batch_buffer)
                
                if response:
                    parsed_batch = parse_batch_response(response, len(batch_buffer))
                    
                    if parsed_batch and len(parsed_batch) == len(batch_buffer):
                        for data, parsed in zip(batch_data_buffer, parsed_batch):
                            entities = parsed.get('entities', [])
                            categories = parsed.get('categories', {})
                            attributes = parsed.get('attributes', {})
                            
                            if not categories:
                                skipped += 1
                                continue
                            
                            kg.add_search_event(data['timestamp'], data['query'], 
                                              entities, categories, attributes)
                            
                            surprise = predictor.calculate_surprise(data['predicted_dist'], categories)
                            predictor.surprise_scores.append(surprise)
                            predictor.update_graph_weights(data['predicted_dist'], categories, learning_rate=LEARNING_RATE)
                            
                            is_transition, trans_score = detector.detect_transition(predictor.surprise_scores)
                            if is_transition:
                                detector.log_transition(data['timestamp'], data['index'], 
                                                       categories, trans_score)
                            
                            processed += 1
                    else:
                        skipped += len(batch_buffer)
                else:
                    skipped += len(batch_buffer)
                
                batch_buffer = []
                batch_data_buffer = []
                
                if processed % 500 == 0 and processed > 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total_searches - start_idx - processed) / rate if rate > 0 else 0
                    
                    import numpy as np
                    recent_surprise = np.mean(predictor.surprise_scores[-500:]) if len(predictor.surprise_scores) >= 500 else np.mean(predictor.surprise_scores)
                    
                    print(f"[{processed:>6}/{total_searches - start_idx}] "
                          f"Graph: {kg.G.number_of_nodes():>5} nodes, {kg.G.number_of_edges():>6} edges | "
                          f"Surprise: {recent_surprise:.3f} | "
                          f"Transitions: {len(detector.transitions):>3} | "
                          f"ETA: {remaining/3600:.1f}h")
                
                if processed % checkpoint_every == 0 and processed > 0:
                    save_checkpoint(kg, predictor, detector, checkpoint_num, idx)
                    checkpoint_num += 1
                    
                    checkpoint_elapsed = time.time() - last_checkpoint_time
                    checkpoint_rate = checkpoint_every / checkpoint_elapsed
                    print(f"    [STATS] Last {checkpoint_every} searches: {checkpoint_elapsed/60:.1f} min ({checkpoint_rate:.2f} searches/sec)")
                    last_checkpoint_time = time.time()
        
        # Process final batch
        if batch_buffer:
            response = query_llm_batch(batch_buffer)
            if response:
                parsed_batch = parse_batch_response(response, len(batch_buffer))
                if parsed_batch:
                    for data, parsed in zip(batch_data_buffer, parsed_batch):
                        entities = parsed.get('entities', [])
                        categories = parsed.get('categories', {})
                        attributes = parsed.get('attributes', {})
                        
                        if categories:
                            kg.add_search_event(data['timestamp'], data['query'], 
                                              entities, categories, attributes)
                            surprise = predictor.calculate_surprise(data['predicted_dist'], categories)
                            predictor.surprise_scores.append(surprise)
                            predictor.update_graph_weights(data['predicted_dist'], categories)
                            
                            is_transition, trans_score = detector.detect_transition(predictor.surprise_scores)
                            if is_transition:
                                detector.log_transition(data['timestamp'], data['index'], 
                                                       categories, trans_score)
                            
                            processed += 1
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("[INTERRUPTED] Saving progress...")
        save_checkpoint(kg, predictor, detector, f"interrupt_{checkpoint_num}", idx)
        print("Progress saved. Resume with the saved checkpoint file.")
        print("="*70)
        return kg, predictor, detector
    
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        print("Saving emergency checkpoint...")
        save_checkpoint(kg, predictor, detector, f"error_{checkpoint_num}", idx)
        raise
    
    final_filename = save_checkpoint(kg, predictor, detector, 'FINAL', total_searches - 1)
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("FULL RUN COMPLETE")
    print("="*70)
    print(f"Total processed: {processed:,} searches")
    print(f"Skipped: {skipped:,}")
    print(f"Success rate: {processed/(processed+skipped)*100:.1f}%")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Average speed: {processed/total_elapsed:.2f} searches/sec")
    print(f"")
    print(f"Final graph:")
    print(f"  Nodes: {kg.G.number_of_nodes():,}")
    print(f"  Edges: {kg.G.number_of_edges():,}")
    print(f"  Transitions detected: {len(detector.transitions)}")
    import numpy as np
    print(f"  Average surprise: {np.mean(predictor.surprise_scores):.3f}")
    print(f"")
    print(f"Saved to: {final_filename}")
    print("="*70)
    
    return kg, predictor, detector