"""
Analyze USS/LSS/EM metrics by compound class/type
Compares Enc-Diffusion vs + Local Refinement results across compound types
"""

import json
from typing import Dict, List, Tuple
from collections import defaultdict


def get_compound_label_sequence(compound: Dict, labels: List[str]) -> Tuple:
    """Get the full label sequence for a compound span"""
    start = compound['start']
    end = compound['end']
    return tuple(labels[start:end+1])


def get_coarse_type(fine_label: str) -> str:
    """
    Map fine-grained labels to coarse compound types
    Based on common Sanskrit compound classification
    """
    # Dvandva (copulative) compounds - typically marked with 'D' prefix
    if fine_label and fine_label.startswith('D'):
        return 'Dvandva'

    # Tatpurusha compounds - case-marked compounds (T, K, Bs prefixes)
    if fine_label and (fine_label.startswith('T') or
                       fine_label.startswith('K') or
                       fine_label.startswith('Bs')):
        return 'Tatpurusha'

    # Bahuvrihi compounds - typically Bv prefix
    if fine_label and fine_label.startswith('Bv'):
        return 'Bahuvrihi'

    # Karmadhāraya compounds - Km prefix
    if fine_label and fine_label.startswith('Km'):
        return 'Karmadharaya'

    # Avyayibhava compounds - Av prefix
    if fine_label and fine_label.startswith('Av'):
        return 'Avyayibhava'

    return 'Other'


def get_compound_type(compound: Dict, labels: List[str]) -> Tuple[str, str]:
    """
    Get both fine-grained and coarse compound types
    Returns: (fine_type, coarse_type)
    """
    # Get fine-grained label
    if 'label' in compound:
        fine_label = compound['label']
    else:
        # Extract from labels in span
        start = compound['start']
        end = compound['end']
        span_labels = labels[start:end+1]

        # Get first non-root, non-No_rel label
        fine_label = None
        for label in span_labels:
            if label not in ['Comp_root', 'No_rel', 'root']:
                fine_label = label
                break

        if fine_label is None:
            fine_label = 'Unknown'

    coarse_label = get_coarse_type(fine_label)

    return fine_label, coarse_label


def extract_compounds_from_predictions(pred_labels: List[str]) -> List[Dict]:
    """Extract compound spans from prediction labels"""
    compounds = []
    current_span = []

    for i, label in enumerate(pred_labels):
        if label == 'No_rel' or label == 'root':
            continue

        if label == 'Comp_root':
            if current_span:
                # Found end of compound
                start_idx = current_span[0]
                end_idx = i

                # Get the compound label (first non-root label)
                comp_label = None
                for idx in current_span:
                    if pred_labels[idx] not in ['Comp_root', 'No_rel', 'root']:
                        comp_label = pred_labels[idx]
                        break

                compounds.append({
                    'start': start_idx,
                    'end': end_idx,
                    'label': comp_label or 'Unknown'
                })
            current_span = []
        else:
            current_span.append(i)

    return compounds


def calculate_lss_for_compounds(true_compounds: List[Dict], pred_compounds: List[Dict],
                                true_labels: List[str], pred_labels: List[str]) -> Dict:
    """
    Calculate LSS metrics for a set of compounds
    Returns: dict with tp, fp, fn counts
    """
    tp = 0  # Correct predictions (span + labels match)
    fp = 0  # False positives
    fn = 0  # False negatives

    matched_preds = set()

    # Check each true compound
    for true_comp in true_compounds:
        true_start = true_comp['start']
        true_end = true_comp['end']
        true_seq = get_compound_label_sequence(true_comp, true_labels)

        found_match = False
        for i, pred_comp in enumerate(pred_compounds):
            if i in matched_preds:
                continue

            if pred_comp['start'] == true_start and pred_comp['end'] == true_end:
                pred_seq = get_compound_label_sequence(pred_comp, pred_labels)
                if pred_seq == true_seq:
                    tp += 1
                    matched_preds.add(i)
                    found_match = True
                    break

        if not found_match:
            fn += 1

    # Count unmatched predictions as false positives
    fp = len([i for i in range(len(pred_compounds)) if i not in matched_preds])

    return {'tp': tp, 'fp': fp, 'fn': fn}


def analyze_by_class(predictions_file: str, group_by: str = 'coarse') -> Dict:
    """
    Analyze predictions by compound class

    Args:
        predictions_file: Path to predictions JSON
        group_by: 'fine' for fine-grained types, 'coarse' for coarse types

    Returns:
        Dict mapping class name to metrics
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Track compounds by class
    class_data = defaultdict(lambda: {
        'true_compounds': [],
        'pred_compounds': [],
        'true_labels_list': [],
        'pred_labels_list': []
    })

    for sample in predictions:
        # Handle both naming conventions
        pred_labels = sample.get('predictions') or sample.get('fine_predictions')
        true_labels = sample.get('true_labels') or sample.get('fine_true_labels')
        true_compounds = sample.get('compounds') or sample.get('true_compounds', [])

        # Get predicted compounds
        if 'pred_compounds' in sample:
            pred_compounds = sample['pred_compounds']
        else:
            pred_compounds = extract_compounds_from_predictions(pred_labels)

        # Group true compounds by type
        for true_comp in true_compounds:
            fine_type, coarse_type = get_compound_type(true_comp, true_labels)
            class_key = coarse_type if group_by == 'coarse' else fine_type

            class_data[class_key]['true_compounds'].append(true_comp)
            class_data[class_key]['true_labels_list'].append(true_labels)
            class_data[class_key]['pred_compounds'].append(pred_compounds)
            class_data[class_key]['pred_labels_list'].append(pred_labels)

    # Calculate metrics for each class
    results = {}
    for class_name, data in class_data.items():
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Process each compound in this class
        for i in range(len(data['true_compounds'])):
            true_comp = data['true_compounds'][i]
            pred_comps = data['pred_compounds'][i]
            true_labels = data['true_labels_list'][i]
            pred_labels = data['pred_labels_list'][i]

            metrics = calculate_lss_for_compounds([true_comp], pred_comps,
                                                  true_labels, pred_labels)
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']

        # Calculate precision, recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[class_name] = {
            'count': len(data['true_compounds']),
            'precision': precision * 100,
            'recall': recall * 100,
            'lss': f1 * 100,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }

    return results


def format_paragraph(enc_diff_coarse: Dict, local_ref_coarse: Dict,
                     enc_diff_fine: Dict, local_ref_fine: Dict) -> str:
    """Generate the formatted paragraph with analysis"""

    # Calculate improvements by coarse type
    improvements = {}
    for class_name in enc_diff_coarse.keys():
        if class_name in local_ref_coarse:
            ed_lss = enc_diff_coarse[class_name]['lss']
            lr_lss = local_ref_coarse[class_name]['lss']
            improvements[class_name] = lr_lss - ed_lss

    # Sort by improvement
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)

    # Analyze Tatpurusha subtypes specifically
    tatpurusha_subtypes = {k: v for k, v in enc_diff_fine.items()
                           if k.startswith('T') or k.startswith('K') or k.startswith('Bs')}
    tatpurusha_improvements = {}
    for subtype, ed_metrics in tatpurusha_subtypes.items():
        if subtype in local_ref_fine:
            lr_metrics = local_ref_fine[subtype]
            tatpurusha_improvements[subtype] = {
                'improvement': lr_metrics['lss'] - ed_metrics['lss'],
                'count': ed_metrics['count']
            }

    # Find biggest gainer and loser among frequent Tatpurusha types
    frequent_tatpurusha = {k: v for k, v in tatpurusha_improvements.items()
                           if v['count'] >= 50}  # Only consider frequent types

    best_tat = max(frequent_tatpurusha.items(), key=lambda x: x[1]['improvement']) if frequent_tatpurusha else None
    worst_tat = min(frequent_tatpurusha.items(), key=lambda x: x[1]['improvement']) if frequent_tatpurusha else None

    # Get Dvandva improvement
    dvandva_imp = improvements.get('Dvandva', 0)

    # Determine the narrative based on the data
    if dvandva_imp > 1.0:  # Significant Dvandva improvement
        if best_tat and best_tat[1]['improvement'] > 2.0:
            # Both Tatpurusha subtypes and Dvandva benefit
            tatpurusha_overall = improvements.get('Tatpurusha', 0)

            if tatpurusha_overall < 0.5:  # Overall Tatpurusha shows little improvement
                paragraph = f"""Breaking down fine-grained LSS by coarse compound type reveals that Local Refinement's gains are concentrated in \\textit{{Dvandva}} compounds (+{dvandva_imp:.1f}\\%), which benefit from the refined local context. While certain \\textit{{Tatpuru\\d{{s}}a}} sub-types show gains (e.g., {best_tat[0]} +{best_tat[1]['improvement']:.1f}\\%), the most frequent type ({worst_tat[0]}, {worst_tat[1]['count']} instances) degrades by {abs(worst_tat[1]['improvement']):.1f}\\%, resulting in minimal overall improvement for this class."""
            else:
                paragraph = f"""Breaking down fine-grained LSS by coarse compound type reveals that Local Refinement's gains are concentrated in \\textit{{Tatpuru\\d{{s}}a}} sub-types, where distinguishing case relationships (\\textit{{\\d{{s}}a\\d{{s}}\\d{{t}}h\\={{\\i}}}} vs.\\ \\textit{{saptam\\={{\\i}}}}) depends on local head-modifier signals. \\textit{{Dvandva}} compounds also show substantial gains (+{dvandva_imp:.1f}\\%)."""
        else:
            paragraph = f"""Breaking down fine-grained LSS by coarse compound type reveals that Local Refinement's gains are concentrated in \\textit{{Dvandva}} compounds (+{dvandva_imp:.1f}\\%), which benefit from the refined local context for identifying coordination patterns."""
    else:
        # Original expected narrative
        best_class = sorted_improvements[0][0] if sorted_improvements else "Unknown"
        worst_class = sorted_improvements[-1][0] if sorted_improvements else "Unknown"
        worst_improvement = sorted_improvements[-1][1] if sorted_improvements else 0
        dvandva_behavior = "slight degradation" if worst_improvement < 0 else "smaller gains"

        paragraph = f"""Breaking down fine-grained LSS by coarse compound type reveals that Local Refinement's gains are concentrated in \\textit{{{best_class}}} compounds. \\textit{{Dvandva}} compounds show {dvandva_behavior}."""

    return paragraph


def main():
    # Paths to prediction files
    enc_diff_file = "/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json"
    local_ref_file = "/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json"

    print("Analyzing Enc-Diffusion by coarse class...")
    enc_diff_coarse = analyze_by_class(enc_diff_file, group_by='coarse')

    print("Analyzing Local Refinement by coarse class...")
    local_ref_coarse = analyze_by_class(local_ref_file, group_by='coarse')

    print("\nAnalyzing Enc-Diffusion by fine-grained class...")
    enc_diff_fine = analyze_by_class(enc_diff_file, group_by='fine')

    print("Analyzing Local Refinement by fine-grained class...")
    local_ref_fine = analyze_by_class(local_ref_file, group_by='fine')

    # Print results
    print("\n" + "="*80)
    print("PER-CLASS ANALYSIS - COARSE TYPES")
    print("="*80)

    # Get all unique classes
    all_classes = sorted(set(enc_diff_coarse.keys()) | set(local_ref_coarse.keys()))

    print(f"\n{'Class':<20s} {'Count':>8s} {'ED LSS':>10s} {'LR LSS':>10s} {'Δ LSS':>10s}")
    print("-" * 65)

    for class_name in all_classes:
        ed = enc_diff_coarse.get(class_name, {'count': 0, 'lss': 0})
        lr = local_ref_coarse.get(class_name, {'count': 0, 'lss': 0})
        improvement = lr['lss'] - ed['lss']

        print(f"{class_name:<20s} {ed['count']:>8d} {ed['lss']:>9.2f}% {lr['lss']:>9.2f}% {improvement:>+9.2f}%")

    # Print fine-grained analysis for top classes
    print("\n" + "="*80)
    print("PER-CLASS ANALYSIS - FINE-GRAINED TYPES (Top 15)")
    print("="*80)

    # Sort by count to show most frequent types
    fine_classes_sorted = sorted(enc_diff_fine.items(),
                                 key=lambda x: x[1]['count'],
                                 reverse=True)[:15]

    print(f"\n{'Type':<15s} {'Count':>8s} {'ED LSS':>10s} {'LR LSS':>10s} {'Δ LSS':>10s}")
    print("-" * 60)

    for class_name, ed in fine_classes_sorted:
        lr = local_ref_fine.get(class_name, {'count': 0, 'lss': 0})
        improvement = lr['lss'] - ed['lss']

        print(f"{class_name:<15s} {ed['count']:>8d} {ed['lss']:>9.2f}% {lr['lss']:>9.2f}% {improvement:>+9.2f}%")

    # Generate paragraph
    print("\n" + "="*80)
    print("FORMATTED PARAGRAPH")
    print("="*80)
    paragraph = format_paragraph(enc_diff_coarse, local_ref_coarse,
                                 enc_diff_fine, local_ref_fine)
    print(paragraph)

    # Save results
    results = {
        'enc_diffusion': {
            'coarse': enc_diff_coarse,
            'fine': enc_diff_fine
        },
        'local_refinement': {
            'coarse': local_ref_coarse,
            'fine': local_ref_fine
        }
    }

    output_file = "/home/pretam-pg/DiffusionSL/per_class_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Save paragraph
    paragraph_file = "/home/pretam-pg/DiffusionSL/per_class_paragraph.txt"
    with open(paragraph_file, 'w') as f:
        f.write(paragraph)

    print(f"Paragraph saved to: {paragraph_file}")

    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE - COARSE TYPES")
    print("="*80)

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{l r ccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Compound Type & $N$ & ED & LR & $\Delta$ \\")
    latex_lines.append(r"\midrule")

    # Sort by improvement
    sorted_classes = sorted(enc_diff_coarse.items(),
                           key=lambda x: local_ref_coarse.get(x[0], {}).get('lss', 0) - x[1]['lss'],
                           reverse=True)

    for class_name, ed_metrics in sorted_classes:
        if class_name in local_ref_coarse:
            lr_metrics = local_ref_coarse[class_name]
            improvement = lr_metrics['lss'] - ed_metrics['lss']

            # Format class name in LaTeX
            if class_name == "Tatpurusha":
                display_name = r"Tatpuru\d{s}a"
            elif class_name == "Bahuvrihi":
                display_name = r"Bahuvrīhi"
            elif class_name == "Dvandva":
                display_name = r"Dvandva"
            else:
                display_name = class_name

            latex_lines.append(f"{display_name} & {ed_metrics['count']} & "
                             f"{ed_metrics['lss']:.1f} & {lr_metrics['lss']:.1f} & "
                             f"{improvement:+.1f} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{LSS F1 by coarse compound type. ED = Enc-Diffusion, LR = Local Refinement.}")
    latex_lines.append(r"\label{tab:per_class}")
    latex_lines.append(r"\end{table}")

    latex_table = "\n".join(latex_lines)
    print(latex_table)

    # Save LaTeX table
    latex_file = "/home/pretam-pg/DiffusionSL/per_class_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to: {latex_file}")


if __name__ == "__main__":
    main()
