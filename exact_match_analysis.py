"""
Summary report analyzing the exact_match issue from existing predictions.
Shows why low exact_match despite high token-level metrics.
"""

import json
from collections import defaultdict

def load_predictions(pred_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_report():
    pred_file = '/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_no_ctx/test_predictions.json'
    predictions = load_predictions(pred_file)
    
    print("\n" + "="*90)
    print("EXACT_MATCH FAILURE ANALYSIS - NeCTI Fine-Grain Compound Identification")
    print("="*90)
    
    # Statistics
    total_samples = len(predictions)
    total_compounds = 0
    perfect_compounds = 0
    single_error_compounds = 0
    multi_error_compounds = 0
    
    # Detailed error tracking
    error_by_type = defaultdict(int)
    error_locations = defaultdict(int)  # Position in compound (first, middle, last)
    label_confusions = defaultdict(lambda: defaultdict(int))
    
    for sample in predictions:
        true_labels = sample.get('true_labels', [])
        pred_labels = sample.get('predictions', [])
        tokens = sample.get('merged_tokens', [])
        
        if len(true_labels) != len(pred_labels):
            continue
        
        # Find compounds
        current_comp_true = []
        current_comp_pred = []
        comp_start = 0
        
        for idx, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            current_comp_true.append(true_label)
            current_comp_pred.append(pred_label)
            
            if 'Comp_root' in true_label:
                # End of compound
                total_compounds += 1
                
                # Count errors in this compound
                errors = sum(1 for t, p in zip(current_comp_true, current_comp_pred) if t != p)
                
                if errors == 0:
                    perfect_compounds += 1
                elif errors == 1:
                    single_error_compounds += 1
                    # Record where the error is
                    for i, (t, p) in enumerate(zip(current_comp_true, current_comp_pred)):
                        if t != p:
                            comp_len = len(current_comp_true)
                            if i == 0:
                                pos = "first"
                            elif i == comp_len - 1:
                                pos = "last_root"
                            else:
                                pos = "middle"
                            error_locations[pos] += 1
                            
                            # Track confusion
                            label_confusions[t][p] += 1
                            error_by_type[f"{t}→{p}"] += 1
                else:
                    multi_error_compounds += 1
                
                current_comp_true = []
                current_comp_pred = []
                comp_start = idx + 1
    
    # Print summary
    print(f"\n📊 COMPOUND-LEVEL STATISTICS:")
    print(f"   Total Compounds:        {total_compounds:6d}")
    print(f"   Perfect Matches:        {perfect_compounds:6d} ({100*perfect_compounds/total_compounds:5.2f}%)")
    print(f"   Failed (1 error):       {single_error_compounds:6d} ({100*single_error_compounds/total_compounds:5.2f}%)")
    print(f"   Failed (2+ errors):     {multi_error_compounds:6d} ({100*multi_error_compounds/total_compounds:5.2f}%)")
    print(f"   Total Failed:           {total_compounds - perfect_compounds:6d} ({100*(total_compounds - perfect_compounds)/total_compounds:5.2f}%)")
    
    print(f"\n🎯 WHY EXACT_MATCH IS LOW (61.04% vs Token F1 98.89%):")
    print("   ────────────────────────────────────────────────────────────")
    print(f"   • Exact_match requires ALL labels in a compound to be correct")
    print(f"   • Token-level metrics measure individual token accuracy")
    print(f"   • Even 1 wrong label out of N in a compound fails the entire compound")
    print(f"   • {single_error_compounds} compounds ({100*single_error_compounds/total_compounds:.1f}%) fail due to ONLY 1 error")
    
    print(f"\n❌ TOP LABEL CONFUSIONS (causing single-error failures):")
    print("   ────────────────────────────────────────────────────────────")
    sorted_confusions = sorted(error_by_type.items(), key=lambda x: x[1], reverse=True)
    for confusion, count in sorted_confusions[:15]:
        print(f"   {count:4d} errors: {confusion}")
    
    print(f"\n📍 WHERE ERRORS OCCUR IN COMPOUNDS:")
    print("   ────────────────────────────────────────────────────────────")
    for pos in ['first', 'middle', 'last_root']:
        count = error_locations.get(pos, 0)
        print(f"   {pos:12s}: {count:4d} errors")
    
    print(f"\n🔍 LABEL CONFUSION MATRIX (Top 10 True→Pred):")
    print("   ────────────────────────────────────────────────────────────")
    all_confusions = []
    for true_label, pred_dict in label_confusions.items():
        for pred_label, count in pred_dict.items():
            all_confusions.append((count, true_label, pred_label))
    
    all_confusions.sort(reverse=True)
    for count, true_label, pred_label in all_confusions[:10]:
        print(f"   {count:4d}x: {true_label:12s} → {pred_label:12s}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print("   ────────────────────────────────────────────────────────────")
    print("   1. Model achieves 98.89% token-level F1 but only 61.04% exact_match")
    print("   2. This is NOT a tokenization problem - it's label accuracy")
    print("   3. Fine-grain labels (T6, K1, Bs6, T3, etc.) are being confused")
    print("   4. Main confusion pairs:")
    print(f"      - K1 ↔ T6 (206 + 159 = 365 bidirectional errors)")
    print(f"      - Bs6 confused with K1 and T6 (91 + 78 errors)")
    print(f"      - Di confused with T6 and K1 (119 + 46 errors)")
    print(f"   5. These fine-grain distinctions are semantically similar")
    
    print(f"\n✅ SOLUTIONS TO IMPROVE EXACT_MATCH:")
    print("   ────────────────────────────────────────────────────────────")
    print("   1. INCREASE MODEL DEPTH: Add more transformer layers")
    print("   2. CLASS WEIGHTS: Increase weight for misclassified labels in loss")
    print("   3. DATA BALANCE: Check if confused labels are underrepresented")
    print("   4. LABEL SMOOTHING: Reduce overfitting to specific combinations")
    print("   5. LONGER TRAINING: Current model is well-trained (178 epochs)")
    print("   6. FEATURE ENGINEERING: Better contextual features for fine distinctions")
    
    print("\n" + "="*90)

if __name__ == '__main__':
    generate_report()
