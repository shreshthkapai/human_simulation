"""
Production Training Pipeline - FIXED VERSION
✅ FIX 1: Checkpoint duplication loop fixed
✅ FIX 2: Edge type analysis added every 3000 searches
✅ FIX 3: Coherence cache cleared every 1000 searches
✅ FIX 4: Accurate decay logging
"""
import pickle
import os
import time
import re
from datetime import datetime
from knowledge_graph import UserKnowledgeGraph
from predictor_hybrid import GraphPredictorHybrid
from transition_detector import TransitionDetector
from llm_client import query_llm_batch, parse_batch_response, validate_and_clean_item
from personalization import personalize_scores, calculate_blend_weight
from config import (CATEGORIES, CHECKPOINT_EVERY, BATCH_SIZE, LEARNING_RATE,
                   ENTITY_EDGE_DECAY_RATE, APPLY_DECAY_EVERY)


def save_checkpoint(kg, predictor, detector, checkpoint_num, last_processed_idx):
    """
    Save checkpoint with EXPLICIT checkpoint number
    """
    checkpoint_data = {
        'kg': kg,
        'predictor': predictor,
        'detector': detector,
        'last_processed_idx': last_processed_idx,
        'checkpoint_num': checkpoint_num,
        'timestamp': datetime.now()
    }
    
    filename = f"checkpoint_{checkpoint_num}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"    [SAVED] {filename} ({file_size:.1f}MB) at index {last_processed_idx}")
    
    return filename


def load_checkpoint(filename):
    """
    Load checkpoint and restore counter
    """
    if not os.path.exists(filename):
        return None, None, None, 0, 0
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    if 'checkpoint_num' in data:
        checkpoint_num = data['checkpoint_num'] + 1
    else:
        checkpoint_num = 0
        match = re.search(r'checkpoint_(\d+).pkl', filename)
        if match:
            checkpoint_num = int(match.group(1)) + 1
    
    print(f"[RESUME] Loaded from {filename}")
    print(f"         Previously processed up to index {data['last_processed_idx']}")
    print(f"         Next checkpoint will be #{checkpoint_num}")
    
    return data['kg'], data['predictor'], data['detector'], data['last_processed_idx'], checkpoint_num


def should_trigger(counter_name, processed, processed_before, interval):
    """
    Robust trigger detection that handles any batch size
    Returns list of trigger points that were crossed
    """
    before_count = processed_before // interval
    after_count = processed // interval
    
    if after_count > before_count:
        triggers_crossed = after_count - before_count
        return True, triggers_crossed
    return False, 0


def full_production_run(df_clean, resume_from=None, start_checkpoint_num=0,
                       checkpoint_every=CHECKPOINT_EVERY, batch_size=BATCH_SIZE):
    """
    Full training pipeline with all fixes applied
    """
    
    if resume_from and os.path.exists(resume_from):
        kg, predictor, detector, start_idx, checkpoint_num = load_checkpoint(resume_from)
        print(f"[RESUME] Continuing from search index {start_idx} (Checkpoint #{checkpoint_num})")
    else:
        kg = UserKnowledgeGraph(CATEGORIES)
        predictor = GraphPredictorHybrid(kg)
        detector = TransitionDetector()
        start_idx = 0
        checkpoint_num = start_checkpoint_num
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
    print(f"Estimated time: {(total_searches - start_idx) * 2.5 / 3600:.1f} hours")
    print("="*70 + "\n")
    
    processed = 0
    skipped = 0
    batch_buffer = []
    batch_data_buffer = []
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    # Track last trigger counts for robust boundary detection
    last_decay_trigger = 0
    last_progress_trigger = 0
    last_stats_trigger = 0
    last_checkpoint_trigger = 0
    last_coherence_rebuild = 0
    last_edge_type_analysis = 0  # ✅ NEW: Track edge type analysis
    
    try:
        for idx in range(start_idx, total_searches):
            row = df_clean.iloc[idx]
            query = row['query']
            timestamp = row['timestamp']
            
            batch_buffer.append(query)
            batch_data_buffer.append({'index': idx, 'query': query, 'timestamp': timestamp})
            
            if len(batch_buffer) >= batch_size:
                processed_before_batch = processed
                
                if processed == 0 and skipped == 0:
                    print(f"  [STATUS] Sending first batch of {len(batch_buffer)} queries to LLM...")

                response = query_llm_batch(batch_buffer)
                
                if response:
                    parsed_batch = parse_batch_response(response, len(batch_buffer))
                    
                    if parsed_batch and len(parsed_batch) == len(batch_buffer):
                        for data, raw_item in zip(batch_data_buffer, parsed_batch):
                            parsed = validate_and_clean_item(raw_item)
                            if parsed is None:
                                skipped += 1
                                print(f"  [SKIP] Item #{data['index']}: \"{data['query'][:40]}...\"")
                                continue
                            
                            entities = parsed['entities']
                            llm_categories = parsed['categories']
                            attributes = parsed['attributes']
                            
                            if not llm_categories:
                                skipped += 1
                                continue
                            
                            # Prediction step
                            predicted_dist = predictor.predict_next_category(
                                current_timestamp=data['timestamp'], use_context=True
                            )
                            
                            # Personalization step
                            blend_weight = calculate_blend_weight(len(kg.search_history))
                            final_categories = personalize_scores(llm_categories, entities, kg, blend_weight)
                            
                            # Verbose debugging for first few or periodic items
                            if processed < 10 or (processed % 100 == 0 and processed < 5000):
                                print(f"\n  [EXAMPLE] Query: \"{data['query'][:50]}...\"")
                                print(f"    Phase 1 (LLM):   {dict(list(llm_categories.items())[:2])}")
                                print(f"    Phase 2 (Final): {dict(list(final_categories.items())[:2])}")
                                print(f"    Entities: {entities[:3]}")
                            
                            # Update Knowledge Graph
                            kg.add_search_event(data['timestamp'], data['query'], entities, final_categories, attributes)
                            
                            # Update Predictor
                            surprise = predictor.calculate_surprise(predicted_dist, final_categories)
                            predictor.surprise_scores.append(surprise)
                            predictor.update_graph_weights(predicted_dist, final_categories, learning_rate=LEARNING_RATE)
                            
                            # Transition Detection
                            is_transition, trans_score = detector.detect_transition(predictor.surprise_scores)
                            if is_transition:
                                detector.log_transition(data['timestamp'], data['index'], final_categories, trans_score)
                            
                            processed += 1
                    else:
                        skipped += len(batch_buffer)
                else:
                    skipped += len(batch_buffer)
                    if processed == 0:
                        print(f"  [WARNING] LLM request failed. Skipped {len(batch_buffer)} items.")
                
                batch_buffer = []
                batch_data_buffer = []
                
                # ========================================================================
                # PERIODIC TRIGGERS - ROBUST BOUNDARY DETECTION
                # ========================================================================
                
                # 1. Temporal Decay (every APPLY_DECAY_EVERY)
                should_decay, decay_count = should_trigger('decay', processed, last_decay_trigger, APPLY_DECAY_EVERY)
                if should_decay and processed > 0:
                    edges_before = kg.G.number_of_edges()
                    predictor.apply_temporal_decay(timestamp, decay_rate=ENTITY_EDGE_DECAY_RATE)
                    edges_changed = edges_before - kg.G.number_of_edges()
                    if edges_changed > 0:
                        print(f"  [DECAY] Set {edges_changed} edges to low weight (decay)")  # ✅ FIXED MESSAGE
                    last_decay_trigger = processed
                
                # 2. Progress Reporting (every 100 items)
                should_progress, progress_count = should_trigger('progress', processed, last_progress_trigger, 100)
                if should_progress and processed > 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total_searches - start_idx - processed) / rate if rate > 0 else 0
                    
                    import numpy as np
                    recent_surprise = np.mean(predictor.surprise_scores[-100:]) if len(predictor.surprise_scores) >= 100 else np.mean(predictor.surprise_scores or [0])
                    current_blend = calculate_blend_weight(len(kg.search_history))
                    ee_edges = sum(1 for _, _, d in kg.G.edges(data=True) if d.get('edge_type') == 'co_occurs')
                    
                    print(f"\n[{processed:>6}/{total_searches - start_idx}] Progress Report")
                    print(f"  Graph: {kg.G.number_of_nodes():>5} nodes, {kg.G.number_of_edges():>6} edges ({ee_edges} entities)")
                    print(f"  Learning: Surprise={recent_surprise:.3f}, Blend={current_blend:.2f}")
                    print(f"  Speed: {rate:.2f} s/sec, ETA: {remaining/3600:.1f}h (Skipped: {skipped})")
                    last_progress_trigger = processed
                
                # 3. Detailed Stats (every 500 items)
                should_stats, stats_count = should_trigger('stats', processed, last_stats_trigger, 500)
                if should_stats and processed > 0:
                    top_cats = sorted(
                        [(cat, kg.G['USER'][cat]['weight']) for cat in kg.categories if kg.G.has_edge('USER', cat)],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                    print(f"\n  ╔══════════════════════════════════╗")
                    print(f"  ║ STATS @ {processed:,} searches")
                    print(f"  ╠══════════════════════════════════╣")
                    for i, (cat, weight) in enumerate(top_cats, 1):
                        print(f"  ║ {i}. {cat:15s} {weight:.3f}")
                    print(f"  ╚══════════════════════════════════╝")
                    last_stats_trigger = processed
                
                # 4. ✅ FIX: Coherence Metadata Rebuild + Cache Clear (every 1000 items)
                should_rebuild, rebuild_count = should_trigger('coherence', processed, last_coherence_rebuild, 1000)
                if should_rebuild and processed > 0:
                    print(f"  [COHERENCE] Updating metadata and clearing cache...")
                    if predictor.coherence_calc:
                        predictor.coherence_calc.incremental_update()
                        predictor.coherence_calc.clear_cache()  # ✅ NEW: Clear cache
                    last_coherence_rebuild = processed
                
                # 5. ✅ NEW: Edge Type Analysis (every 3000 items)
                should_analyze_edges, edge_analysis_count = should_trigger('edge_type', processed, last_edge_type_analysis, 3000)
                if should_analyze_edges and processed > 0:
                    print(f"  [EDGE TYPES] Analyzing temporal patterns...")
                    if predictor.edge_type_detector:
                        predictor.edge_type_detector.analyze_search_history()
                        type_dist = predictor.edge_type_detector.tag_all_edges()
                        print(f"    Tagged: {sum(type_dist.values())} edges")
                    last_edge_type_analysis = processed
                
                # 6. ✅ FIX: Checkpoints (every CHECKPOINT_EVERY) - NO LOOP
                should_checkpoint, checkpoint_count = should_trigger('checkpoint', processed, last_checkpoint_trigger, checkpoint_every)
                if should_checkpoint and processed > 0:
                    save_checkpoint(kg, predictor, detector, checkpoint_num, idx)
                    checkpoint_num += checkpoint_count  # ✅ FIXED: Increment by count, not in loop
                    last_checkpoint_trigger = processed
                    last_checkpoint_time = time.time()
        
        # Final batch processing
        if batch_buffer:
            response = query_llm_batch(batch_buffer)
            if response:
                parsed_batch = parse_batch_response(response, len(batch_buffer))
                if parsed_batch:
                    for data, raw_item in zip(batch_data_buffer, parsed_batch):
                        parsed = validate_and_clean_item(raw_item)
                        if parsed:
                            kg.add_search_event(data['timestamp'], data['query'], parsed['entities'], parsed['categories'], parsed['attributes'])
                            processed += 1
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving progress...")
        save_checkpoint(kg, predictor, detector, f"interrupt_{checkpoint_num}", idx)
        return kg, predictor, detector
    
    final_filename = save_checkpoint(kg, predictor, detector, 'FINAL', total_searches - 1)
    print(f"\nTRAINING COMPLETE. Saved to: {final_filename}")
    return kg, predictor, detector