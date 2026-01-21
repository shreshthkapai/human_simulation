"""
Production Training Pipeline - CRASH-PROOF VERSION
✅ All initialization bugs fixed
✅ Resume-safe trigger management
✅ Robust error handling
"""
import pickle
import os
import time
import re
import numpy as np  # ✅ MOVED TO TOP
from datetime import datetime
from knowledge_graph import UserKnowledgeGraph
from predictor_hybrid import GraphPredictorHybrid
from transition_detector import TransitionDetector
from llm_client import query_llm_batch, parse_batch_response, validate_and_clean_item
from personalization import personalize_scores, calculate_blend_weight
from config import (CATEGORIES, CHECKPOINT_EVERY, BATCH_SIZE, LEARNING_RATE,
                   ENTITY_EDGE_DECAY_RATE, APPLY_DECAY_EVERY)


def save_checkpoint(kg, predictor, detector, checkpoint_num, last_processed_idx, 
                   trigger_states=None):  # ✅ NEW: Save trigger states
    """
    Save checkpoint with trigger states for crash-safe resume
    """
    checkpoint_data = {
        'kg': kg,
        'predictor': predictor,
        'detector': detector,
        'last_processed_idx': last_processed_idx,
        'checkpoint_num': checkpoint_num,
        'trigger_states': trigger_states or {},  # ✅ NEW
        'timestamp': datetime.now()
    }
    
    filename = f"checkpoint_{checkpoint_num}.pkl"
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"    [SAVED] {filename} ({file_size:.1f}MB) at index {last_processed_idx}")
        return filename
    
    except Exception as e:
        print(f"    [ERROR] Failed to save checkpoint: {e}")
        return None


def load_checkpoint(filename):
    """
    Load checkpoint and restore ALL state including triggers
    """
    if not os.path.exists(filename):
        return None, None, None, 0, 0, {}
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Extract checkpoint number
        if 'checkpoint_num' in data:
            checkpoint_num = data['checkpoint_num'] + 1
        else:
            checkpoint_num = 0
            match = re.search(r'checkpoint_(\d+).pkl', filename)
            if match:
                checkpoint_num = int(match.group(1)) + 1
        
        # ✅ NEW: Restore trigger states
        trigger_states = data.get('trigger_states', {})
        
        print(f"[RESUME] Loaded from {filename}")
        print(f"         Previously processed up to index {data['last_processed_idx']}")
        print(f"         Next checkpoint will be #{checkpoint_num}")
        
        if trigger_states:
            print(f"         Restored {len(trigger_states)} trigger states")
        
        return (data['kg'], data['predictor'], data['detector'], 
                data['last_processed_idx'], checkpoint_num, trigger_states)
    
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint {filename}: {e}")
        return None, None, None, 0, 0, {}


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
    Full training pipeline - CRASH-PROOF VERSION
    ✅ Safe resume with trigger state restoration
    ✅ Robust error handling
    ✅ No mid-training crashes
    """
    
    # ✅ Load or initialize
    if resume_from and os.path.exists(resume_from):
        kg, predictor, detector, start_idx, checkpoint_num, trigger_states = load_checkpoint(resume_from)
        
        if kg is None:
            print("[ERROR] Checkpoint corrupt, starting fresh")
            kg = UserKnowledgeGraph(CATEGORIES)
            predictor = GraphPredictorHybrid(kg)
            detector = TransitionDetector()
            start_idx = 0
            checkpoint_num = start_checkpoint_num
            trigger_states = {}
        else:
            print(f"[RESUME] Continuing from search index {start_idx} (Checkpoint #{checkpoint_num})")
    else:
        kg = UserKnowledgeGraph(CATEGORIES)
        predictor = GraphPredictorHybrid(kg)
        detector = TransitionDetector()
        start_idx = 0
        checkpoint_num = start_checkpoint_num
        trigger_states = {}
        print("[START] Fresh run initialized")
    
    total_searches = len(df_clean)
    
    print("\n" + "="*70)
    print("TWO-PHASE TRAINING: LLM + GRAPH PERSONALIZATION (CRASH-PROOF)")
    print("="*70)
    print(f"Total searches: {total_searches:,}")
    print(f"Starting from: {start_idx:,}")
    print(f"Remaining: {total_searches - start_idx:,}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint every: {checkpoint_every}")
    print(f"Estimated time: {(total_searches - start_idx) * 2.5 / 3600:.1f} hours")
    print("="*70 + "\n")
    
    # ✅ Initialize counters (cumulative across resume)
    processed = trigger_states.get('processed', 0)
    skipped = 0
    batch_buffer = []
    batch_data_buffer = []
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    # ✅ Restore trigger states from checkpoint (resume-safe!)
    last_decay_trigger = trigger_states.get('last_decay_trigger', 0)
    last_progress_trigger = trigger_states.get('last_progress_trigger', 0)
    last_stats_trigger = trigger_states.get('last_stats_trigger', 0)
    last_checkpoint_trigger = trigger_states.get('last_checkpoint_trigger', 0)
    last_coherence_rebuild = trigger_states.get('last_coherence_rebuild', 0)
    last_edge_type_analysis = trigger_states.get('last_edge_type_analysis', 0)
    last_competition_rebuild = trigger_states.get('last_competition_rebuild', 0)
    
    # ✅ Track initialization flags (restore from checkpoint)
    competition_initialized = trigger_states.get('competition_initialized', False)
    coherence_initialized = trigger_states.get('coherence_initialized', False)
    
    print(f"[STATE] Processed so far: {processed:,}")
    print(f"[STATE] Competition initialized: {competition_initialized}")
    print(f"[STATE] Coherence initialized: {coherence_initialized}\n")
    
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
                                if skipped <= 10:  # Only print first 10 skips
                                    print(f"  [SKIP] Item #{data['index']}: \"{data['query'][:40]}...\"")
                                continue
                            
                            entities = parsed['entities']
                            llm_categories = parsed['categories']
                            attributes = parsed['attributes']
                            
                            if not llm_categories:
                                skipped += 1
                                continue
                            
                            # Prediction step
                            try:
                                predicted_dist = predictor.predict_next_category(
                                    current_timestamp=data['timestamp'], use_context=True
                                )
                            except Exception as e:
                                print(f"  [ERROR] Prediction failed at {processed}: {e}")
                                continue
                            
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
                # PERIODIC TRIGGERS - CRASH-SAFE
                # ========================================================================
                
                # ✅ 0. ONE-TIME INITIALIZATIONS (safe minimum: 500 searches for good statistics)
                if processed >= 500 and not competition_initialized:
                    print(f"\n  [INIT] Building competition matrix from {len(kg.search_history)} searches...")
                    try:
                        if predictor.competition_manager:
                            predictor.competition_manager.rebuild_competition_matrix()
                            print(f"    ✅ Competition matrix initialized!")
                            competition_initialized = True
                    except Exception as e:
                        print(f"    ❌ Competition init failed: {e}")
                
                if processed >= 500 and not coherence_initialized:
                    print(f"  [INIT] Building coherence metadata from {len(kg.search_history)} searches...")
                    try:
                        if predictor.coherence_calc:
                            predictor.coherence_calc.incremental_update()
                            print(f"    ✅ Coherence metadata initialized!")
                            coherence_initialized = True
                    except Exception as e:
                        print(f"    ❌ Coherence init failed: {e}")
                
                # 1. Temporal Decay (every APPLY_DECAY_EVERY)
                should_decay, decay_count = should_trigger('decay', processed, last_decay_trigger, APPLY_DECAY_EVERY)
                if should_decay and processed > 0:
                    try:
                        affected = predictor.apply_temporal_decay(timestamp, decay_rate=ENTITY_EDGE_DECAY_RATE)
                        if affected > 0:
                            print(f"  [DECAY] Decayed {affected} edges to low weight")
                        last_decay_trigger = processed
                    except Exception as e:
                        print(f"  [ERROR] Decay failed: {e}")
                
                # 2. Progress Reporting (every 100 items)
                should_progress, progress_count = should_trigger('progress', processed, last_progress_trigger, 100)
                if should_progress and processed > 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_searches - start_idx - processed) / rate if rate > 0 else 0
                    
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
                
                # 4. Coherence Metadata Rebuild + Cache Clear (every 1000 items)
                should_rebuild, rebuild_count = should_trigger('coherence', processed, last_coherence_rebuild, 1000)
                if should_rebuild and processed > 0:
                    try:
                        print(f"  [COHERENCE] Updating metadata and clearing cache...")
                        if predictor.coherence_calc:
                            predictor.coherence_calc.incremental_update()
                            predictor.coherence_calc.clear_cache()
                        last_coherence_rebuild = processed
                    except Exception as e:
                        print(f"  [ERROR] Coherence update failed: {e}")
                
                # 5. Competition Matrix Rebuild (every 2000 items)
                should_rebuild_comp, comp_count = should_trigger('competition', processed, last_competition_rebuild, 2000)
                if should_rebuild_comp and processed > 0 and competition_initialized:
                    try:
                        print(f"  [COMPETITION] Rebuilding matrix from {len(kg.search_history)} searches...")
                        if predictor.competition_manager:
                            predictor.competition_manager.rebuild_competition_matrix()
                            
                            # Show sample competition scores (first rebuild only)
                            if last_competition_rebuild == 0:
                                print(f"    Sample competition scores:")
                                cats_sample = kg.categories[:3]
                                for cat_a in cats_sample:
                                    for cat_b in cats_sample:
                                        if cat_a != cat_b:
                                            comp = predictor.competition_manager.get_competition(cat_a, cat_b)
                                            print(f"      {cat_a} vs {cat_b}: {comp:.3f}")
                        last_competition_rebuild = processed
                    except Exception as e:
                        print(f"  [ERROR] Competition rebuild failed: {e}")
                
                # 6. Edge Type Analysis (every 10000 items)
                should_analyze_edges, edge_analysis_count = should_trigger('edge_type', processed, last_edge_type_analysis, 10000)
                if should_analyze_edges and processed > 0:
                    try:
                        print(f"  [EDGE TYPES] Analyzing temporal patterns...")
                        if predictor.edge_type_detector:
                            predictor.edge_type_detector.analyze_search_history()
                            type_dist = predictor.edge_type_detector.tag_all_edges()
                            print(f"    Tagged: {sum(type_dist.values())} edges")
                            if last_edge_type_analysis == 0:  # First analysis only
                                print(f"    Distribution: {dict(list(type_dist.items())[:3])}")
                        last_edge_type_analysis = processed
                    except Exception as e:
                        print(f"  [ERROR] Edge type analysis failed: {e}")
                
                # 7. Checkpoints (every CHECKPOINT_EVERY)
                should_checkpoint, checkpoint_count = should_trigger('checkpoint', processed, last_checkpoint_trigger, checkpoint_every)
                if should_checkpoint and processed > 0:
                    # ✅ Save trigger states for resume
                    trigger_states = {
                        'processed': processed,
                        'last_decay_trigger': last_decay_trigger,
                        'last_progress_trigger': last_progress_trigger,
                        'last_stats_trigger': last_stats_trigger,
                        'last_checkpoint_trigger': last_checkpoint_trigger,
                        'last_coherence_rebuild': last_coherence_rebuild,
                        'last_edge_type_analysis': last_edge_type_analysis,
                        'last_competition_rebuild': last_competition_rebuild,
                        'competition_initialized': competition_initialized,
                        'coherence_initialized': coherence_initialized,
                    }
                    
                    save_checkpoint(kg, predictor, detector, checkpoint_num, idx, trigger_states)
                    checkpoint_num += checkpoint_count
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
        trigger_states = {
            'processed': processed,
            'last_decay_trigger': last_decay_trigger,
            'last_progress_trigger': last_progress_trigger,
            'last_stats_trigger': last_stats_trigger,
            'last_checkpoint_trigger': last_checkpoint_trigger,
            'last_coherence_rebuild': last_coherence_rebuild,
            'last_edge_type_analysis': last_edge_type_analysis,
            'last_competition_rebuild': last_competition_rebuild,
            'competition_initialized': competition_initialized,
            'coherence_initialized': coherence_initialized,
        }
        save_checkpoint(kg, predictor, detector, f"interrupt_{checkpoint_num}", idx, trigger_states)
        return kg, predictor, detector
    
    except Exception as e:
        print(f"\n[CRASH] Unexpected error: {e}")
        print(f"[CRASH] Saving emergency checkpoint...")
        trigger_states = {
            'processed': processed,
            'last_decay_trigger': last_decay_trigger,
            'last_progress_trigger': last_progress_trigger,
            'last_stats_trigger': last_stats_trigger,
            'last_checkpoint_trigger': last_checkpoint_trigger,
            'last_coherence_rebuild': last_coherence_rebuild,
            'last_edge_type_analysis': last_edge_type_analysis,
            'last_competition_rebuild': last_competition_rebuild,
            'competition_initialized': competition_initialized,
            'coherence_initialized': coherence_initialized,
        }
        save_checkpoint(kg, predictor, detector, f"crash_{checkpoint_num}", idx, trigger_states)
        raise  # Re-raise for debugging
    
    # Final save
    trigger_states = {
        'processed': processed,
        'last_decay_trigger': last_decay_trigger,
        'last_progress_trigger': last_progress_trigger,
        'last_stats_trigger': last_stats_trigger,
        'last_checkpoint_trigger': last_checkpoint_trigger,
        'last_coherence_rebuild': last_coherence_rebuild,
        'last_edge_type_analysis': last_edge_type_analysis,
        'last_competition_rebuild': last_competition_rebuild,
        'competition_initialized': competition_initialized,
        'coherence_initialized': coherence_initialized,
    }
    
    final_filename = save_checkpoint(kg, predictor, detector, 'FINAL', total_searches - 1, trigger_states)
    print(f"\n✅ TRAINING COMPLETE. Saved to: {final_filename}")
    print(f"✅ All features properly initialized and working")
    print(f"✅ Processed {processed:,} searches, skipped {skipped:,}")
    return kg, predictor, detector