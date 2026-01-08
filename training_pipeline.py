"""
Production Training Pipeline - Two-Phase System + Temporal Decay
Phase 1: LLM generic scoring
Phase 2: Graph personalization
"""
import pickle
import os
import time
from datetime import datetime
from knowledge_graph import UserKnowledgeGraph
from predictor_hybrid import GraphPredictorHybrid
from transition_detector import TransitionDetector
from llm_client import query_llm_batch, parse_batch_response
from personalization import personalize_scores, calculate_blend_weight
from config import (CATEGORIES, CHECKPOINT_EVERY, BATCH_SIZE, LEARNING_RATE,
                   ENTITY_EDGE_DECAY_RATE, APPLY_DECAY_EVERY)


def save_checkpoint(kg, predictor, detector, checkpoint_num, last_processed_idx):
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
    if not os.path.exists(filename):
        return None, None, None, 0
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"[RESUME] Loaded from {filename}")
    print(f"         Previously processed up to index {data['last_processed_idx']}")
    
    return data['kg'], data['predictor'], data['detector'], data['last_processed_idx']


def full_production_run(df_clean, resume_from=None, checkpoint_every=CHECKPOINT_EVERY, 
                       batch_size=BATCH_SIZE):
    
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
    print("TWO-PHASE TRAINING: LLM + GRAPH PERSONALIZATION")
    print("="*70)
    print(f"Total searches: {total_searches:,}")
    print(f"Starting from: {start_idx:,}")
    print(f"Remaining: {total_searches - start_idx:,}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint every: {checkpoint_every}")
    print(f"Phase 1: LLM generic scoring (formula-guided)")
    print(f"Phase 2: Graph personalization (adaptive blending)")
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
            
            batch_buffer.append(query)
            batch_data_buffer.append({
                'index': idx,
                'query': query,
                'timestamp': timestamp
            })
            
            if len(batch_buffer) >= batch_size:
                # PHASE 1: LLM - Generic semantic scoring
                response = query_llm_batch(batch_buffer)
                
                if response:
                    parsed_batch = parse_batch_response(response, len(batch_buffer))
                    
                    if parsed_batch and len(parsed_batch) == len(batch_buffer):
                        for data, parsed in zip(batch_data_buffer, parsed_batch):
                            entities = parsed.get('entities', [])
                            llm_categories = parsed.get('categories', {})
                            attributes = parsed.get('attributes', {})
                            
                            if not llm_categories:
                                skipped += 1
                                continue
                            
                            # PREDICT FIRST 
                            predicted_dist = predictor.predict_next_category(
                                current_timestamp=data['timestamp'],
                                use_context=True
                            )
                            
                            # PHASE 2: Graph - Personalization
                            blend_weight = calculate_blend_weight(len(kg.search_history))
                            final_categories = personalize_scores(
                                llm_categories, 
                                entities, 
                                kg, 
                                blend_weight
                            )
                            
                            # Show Phase 1 vs Phase 2 for first few searches
                            if processed < 10 or (processed % 1000 == 0 and processed < 5000):
                                print(f"\n  [EXAMPLE] Query: \"{data['query'][:50]}...\"")
                                print(f"    Phase 1 (LLM):   {dict(list(llm_categories.items())[:2])}")
                                print(f"    Phase 2 (Final): {dict(list(final_categories.items())[:2])}")
                                print(f"    Entities: {entities[:3]}")
                            
                            # Add search event to graph
                            kg.add_search_event(
                                data['timestamp'], 
                                data['query'], 
                                entities, 
                                final_categories, 
                                attributes
                            )
                            
                            # Calculate surprise and update
                            surprise = predictor.calculate_surprise(predicted_dist, final_categories)
                            predictor.surprise_scores.append(surprise)
                            predictor.update_graph_weights(predicted_dist, final_categories, learning_rate=LEARNING_RATE)
                            
                            # Detect transitions
                            is_transition, trans_score = detector.detect_transition(predictor.surprise_scores)
                            if is_transition:
                                detector.log_transition(
                                    data['timestamp'], 
                                    data['index'], 
                                    final_categories, 
                                    trans_score
                                )
                            
                            processed += 1
                    else:
                        skipped += len(batch_buffer)
                else:
                    skipped += len(batch_buffer)
                
                batch_buffer = []
                batch_data_buffer = []
                
                # Apply temporal decay periodically
                if processed % APPLY_DECAY_EVERY == 0 and processed > 0:
                    edges_before = kg.G.number_of_edges()
                    predictor.apply_temporal_decay(timestamp, decay_rate=ENTITY_EDGE_DECAY_RATE)
                    edges_after = kg.G.number_of_edges()
                    edges_removed = edges_before - edges_after
                    if edges_removed > 0:
                        print(f"  [DECAY] Removed {edges_removed} weak entity↔entity edges")
                
                # Progress reporting every 100 searches
                if processed % 100 == 0 and processed > 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total_searches - start_idx - processed) / rate if rate > 0 else 0
                    
                    import numpy as np
                    recent_surprise = np.mean(predictor.surprise_scores[-100:]) if len(predictor.surprise_scores) >= 100 else np.mean(predictor.surprise_scores)
                    current_blend = calculate_blend_weight(len(kg.search_history))
                    
                    # Count entity-entity edges
                    entity_entity_edges = sum(1 for u, v, d in kg.G.edges(data=True) 
                                             if d.get('edge_type') == 'co_occurs')
                    
                    print(f"\n[{processed:>6}/{total_searches - start_idx}] Progress Report")
                    print(f"  Graph: {kg.G.number_of_nodes():>5} nodes, {kg.G.number_of_edges():>6} edges ({entity_entity_edges} entity↔entity)")
                    print(f"  Learning: Surprise={recent_surprise:.3f}, Blend={current_blend:.2f} (LLM={1-current_blend:.0%}, Graph={current_blend:.0%})")
                    print(f"  Transitions: {len(detector.transitions):>3} detected")
                    print(f"  Speed: {rate:.2f} searches/sec, ETA: {remaining/3600:.1f}h")
                
                # Detailed reporting every 500 searches
                if processed % 500 == 0 and processed > 0:
                    import numpy as np
                    
                    # Top categories by user affinity
                    top_cats = sorted(
                        [(cat, kg.G['USER'][cat]['weight']) for cat in kg.categories if kg.G.has_edge('USER', cat)],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                    
                    # Recent transition info
                    recent_transitions = detector.transitions[-5:] if len(detector.transitions) >= 5 else detector.transitions
                    
                    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
                    print(f"  ║ DETAILED STATS @ {processed:,} searches")
                    print(f"  ╠══════════════════════════════════════════════════════════════╣")
                    print(f"  ║ Top User Interests:")
                    for i, (cat, weight) in enumerate(top_cats, 1):
                        print(f"  ║   {i}. {cat:20s} {weight:.3f}")
                    print(f"  ║")
                    print(f"  ║ Recent Transitions:")
                    if recent_transitions:
                        for trans in recent_transitions[-3:]:
                            date_str = trans['timestamp'].strftime('%Y-%m-%d')
                            top_cat = max(trans['new_categories'].items(), key=lambda x: x[1])[0]
                            print(f"  ║   {date_str}: {top_cat} (score: {trans['transition_score']:.2f})")
                    else:
                        print(f"  ║   No transitions detected yet")
                    print(f"  ╚══════════════════════════════════════════════════════════════╝")
                
                # Checkpointing
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
                        llm_categories = parsed.get('categories', {})
                        attributes = parsed.get('attributes', {})
                        
                        if llm_categories:
                            # Predict first
                            predicted_dist = predictor.predict_next_category(
                                current_timestamp=data['timestamp'], use_context=True
                            )
                            
                            # Personalize
                            blend_weight = calculate_blend_weight(len(kg.search_history))
                            final_categories = personalize_scores(llm_categories, entities, kg, blend_weight)
                            
                            # Update graph
                            kg.add_search_event(data['timestamp'], data['query'], 
                                              entities, final_categories, attributes)
                            
                            # Calculate surprise and update
                            surprise = predictor.calculate_surprise(predicted_dist, final_categories)
                            predictor.surprise_scores.append(surprise)
                            predictor.update_graph_weights(predicted_dist, final_categories)
                            
                            is_transition, trans_score = detector.detect_transition(predictor.surprise_scores)
                            if is_transition:
                                detector.log_transition(data['timestamp'], data['index'], 
                                                       final_categories, trans_score)
                            
                            processed += 1
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("[INTERRUPTED] Saving progress...")
        save_checkpoint(kg, predictor, detector, f"interrupt_{checkpoint_num}", idx)
        print("Progress saved. Resume with the saved checkpoint file.")
        print("="*70)
        return kg, predictor, detector
    
    final_filename = save_checkpoint(kg, predictor, detector, 'FINAL', total_searches - 1)
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
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
    print(f"  Transitions: {len(detector.transitions)}")
    import numpy as np
    print(f"  Avg surprise: {np.mean(predictor.surprise_scores):.3f}")
    print(f"")
    print(f"Saved to: {final_filename}")
    print("="*70)
    
    return kg, predictor, detector