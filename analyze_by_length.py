"""
Analyze USS/LSS/EM metrics by compound length
Compares Enc-Diffusion vs + Local Refinement results
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_relations_from_labels(labels: List[str], start_idx: int = 0) -> List[List]:
    """Convert label sequence to relations format"""
    relations = []
    for i, label in enumerate(labels):
        if label != 'No_rel' and label != 'root':
            token_idx = start_idx + i + 1  # 1-indexed
            relations.append([token_idx, 0, label])
    return relations


def get_compound_length(compound: Dict) -> int:
    """Get the number of tokens in a compound"""
    if 'tokens' in compound:
        return len(compound['tokens'])
    else:
        # Calculate from start and end indices
        return compound['end'] - compound['start'] + 1


def calculate_uss(true_relations: List[List], pred_relations: List[List]) -> float:
    """Calculate Unlabeled Span Score (USS)"""
    correct = 0
    true_count = 0
    pred_count = 0

    # Filter out No_rel and root
    true_spans = [(r[0], r[1]) for r in true_relations if r[2] != 'No_rel' and r[2] != 'root']
    pred_spans = [(r[0], r[1]) for r in pred_relations if r[2] != 'No_rel' and r[2] != 'root']

    for span in pred_spans:
        if span in true_spans:
            correct += 1

    true_count = len(true_spans)
    pred_count = len(pred_spans)

    p = correct / pred_count if pred_count != 0 else 0
    r = correct / true_count if true_count != 0 else 0
    f1 = 2 * p * r / (p + r) if p != 0 and r != 0 else 0

    return f1


def calculate_lss(true_relations: List[List], pred_relations: List[List]) -> Dict[str, float]:
    """Calculate Labeled Span Score (LSS) and Exact Match"""
    correct = 0
    predict_count = 0
    true_count = 0

    # Filter out No_rel and root
    true_rels_filtered = [r for r in true_relations if r[2] != 'No_rel' and r[2] != 'root']
    pred_rels_filtered = [r for r in pred_relations if r[2] != 'No_rel' and r[2] != 'root']

    # Count matching relations
    for rel in pred_rels_filtered:
        if rel in true_rels_filtered:
            correct += 1

    predict_count = len(pred_rels_filtered)
    true_count = len(true_rels_filtered)

    p = correct / predict_count if predict_count != 0 else 0
    r = correct / true_count if true_count != 0 else 0
    f1 = 2 * p * r / (p + r) if p != 0 and r != 0 else 0

    return {'precision': p, 'recall': r, 'f1': f1}


def comps_from_relations(relations: List[str]) -> List[List[str]]:
    """Extract compounds from relations list"""
    lst = []
    nested_comp = []
    for rel in relations:
        if 'Comp_root' in rel:
            lst.append(rel)
            nested_comp.append(lst)
            lst = []
        else:
            lst.append(rel)
    return nested_comp


def calculate_exact_match(true_relations: List[List], pred_relations: List[List]) -> Tuple[int, int]:
    """Calculate exact match count and total compounds"""
    # Filter out No_rel and root
    true_rels_filtered = [r for r in true_relations if r[2] != 'No_rel' and r[2] != 'root']
    pred_rels_filtered = [r for r in pred_relations if r[2] != 'No_rel' and r[2] != 'root']

    # Convert to string for comparison
    tr_copy = [','.join(map(str, lst)) for lst in true_rels_filtered]
    pr_copy = [','.join(map(str, lst)) for lst in pred_rels_filtered]

    # Extract compounds (groups ending with Comp_root)
    tr_comps = comps_from_relations(tr_copy)
    pr_comps = comps_from_relations(pr_copy)

    # Exact match count
    em = 0
    for comp in pr_comps:
        if comp in tr_comps:
            em += 1

    return em, len(tr_comps)


def analyze_by_length(predictions_file: str, length_bins: Dict[str, Tuple[int, int]]) -> Dict:
    """
    Analyze predictions by compound length bins

    Args:
        predictions_file: Path to test_predictions.json
        length_bins: Dict mapping bin name to (min_len, max_len) tuple
                     max_len=None means no upper bound

    Returns:
        Dict with metrics for each length bin
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Initialize bins
    bin_data = {bin_name: {
        'true_relations': [],
        'pred_relations': [],
        'true_count': 0,
        'pred_count': 0,
        'correct_uss': 0,
        'correct_lss': 0,
        'em_count': 0,
        'total_compounds': 0
    } for bin_name in length_bins}

    # Process each sample
    for sample in predictions:
        # Handle both naming conventions
        pred_labels = sample.get('predictions') or sample.get('fine_predictions')
        true_labels = sample.get('true_labels') or sample.get('fine_true_labels')
        compounds = sample.get('compounds') or sample.get('true_compounds', [])

        # Group compounds by length bin
        for compound in compounds:
            comp_length = get_compound_length(compound)

            # Find which bin this compound belongs to
            for bin_name, (min_len, max_len) in length_bins.items():
                if max_len is None:  # Open-ended bin
                    if comp_length >= min_len:
                        target_bin = bin_name
                        break
                elif min_len <= comp_length <= max_len:
                    target_bin = bin_name
                    break
            else:
                continue  # Skip compounds outside all bins

            # Extract relations for this compound
            start = compound['start']
            end = compound['end']

            comp_pred_labels = pred_labels[start:end+1]
            comp_true_labels = true_labels[start:end+1]

            pred_rels = extract_relations_from_labels(comp_pred_labels, start)
            true_rels = extract_relations_from_labels(comp_true_labels, start)

            # Update bin statistics
            bin_data[target_bin]['true_relations'].append(true_rels)
            bin_data[target_bin]['pred_relations'].append(pred_rels)

    # Calculate metrics for each bin
    results = {}
    for bin_name, data in bin_data.items():
        true_rels_all = data['true_relations']
        pred_rels_all = data['pred_relations']

        if len(true_rels_all) == 0:
            results[bin_name] = {
                'uss': 0.0,
                'lss': 0.0,
                'em': 0.0,
                'num_compounds': 0
            }
            continue

        # Calculate USS
        total_correct_uss = 0
        total_true_spans = 0
        total_pred_spans = 0

        for true_rels, pred_rels in zip(true_rels_all, pred_rels_all):
            true_spans = [(r[0], r[1]) for r in true_rels]
            pred_spans = [(r[0], r[1]) for r in pred_rels]

            for span in pred_spans:
                if span in true_spans:
                    total_correct_uss += 1

            total_true_spans += len(true_spans)
            total_pred_spans += len(pred_spans)

        p_uss = total_correct_uss / total_pred_spans if total_pred_spans > 0 else 0
        r_uss = total_correct_uss / total_true_spans if total_true_spans > 0 else 0
        uss = 2 * p_uss * r_uss / (p_uss + r_uss) if (p_uss + r_uss) > 0 else 0

        # Calculate LSS
        total_correct_lss = 0
        total_true_rels = 0
        total_pred_rels = 0

        for true_rels, pred_rels in zip(true_rels_all, pred_rels_all):
            for rel in pred_rels:
                if rel in true_rels:
                    total_correct_lss += 1

            total_true_rels += len(true_rels)
            total_pred_rels += len(pred_rels)

        p_lss = total_correct_lss / total_pred_rels if total_pred_rels > 0 else 0
        r_lss = total_correct_lss / total_true_rels if total_true_rels > 0 else 0
        lss = 2 * p_lss * r_lss / (p_lss + r_lss) if (p_lss + r_lss) > 0 else 0

        # Calculate Exact Match
        total_em = 0
        total_compounds = 0

        for true_rels, pred_rels in zip(true_rels_all, pred_rels_all):
            em, tc = calculate_exact_match(true_rels, pred_rels)
            total_em += em
            total_compounds += tc

        em_score = total_em / total_compounds if total_compounds > 0 else 0

        results[bin_name] = {
            'uss': uss * 100,  # Convert to percentage
            'lss': lss * 100,
            'em': em_score * 100,
            'num_compounds': len(true_rels_all)
        }

    return results


def format_latex_table(enc_diff_results: Dict, local_ref_results: Dict, length_bins: Dict) -> str:
    """Generate LaTeX table string"""

    table_lines = []
    table_lines.append(r"\begin{table}[t]")
    table_lines.append(r"\centering")
    table_lines.append(r"\small")
    table_lines.append(r"\begin{tabular}{l ccc ccc}")
    table_lines.append(r"\toprule")
    table_lines.append(r"& \multicolumn{3}{c}{\textbf{Enc-Diffusion}} & \multicolumn{3}{c}{\textbf{+ Local Refine.}} \\")
    table_lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    table_lines.append(r"$N$ & USS & LSS & EM & USS & LSS & EM \\")
    table_lines.append(r"\midrule")

    for bin_name in length_bins.keys():
        ed_metrics = enc_diff_results[bin_name]
        lr_metrics = local_ref_results[bin_name]

        # Format row
        row = f"{bin_name} & "
        row += f"{ed_metrics['uss']:.1f} & {ed_metrics['lss']:.1f} & {ed_metrics['em']:.1f} & "
        row += f"{lr_metrics['uss']:.1f} & {lr_metrics['lss']:.1f} & {lr_metrics['em']:.1f} \\\\"

        table_lines.append(row)

    table_lines.append(r"\bottomrule")
    table_lines.append(r"\end{tabular}")
    table_lines.append(r"\caption{USS/LSS/EM by compound length for Enc-Diffusion (ED) and + Local Refinement (LR). Fine-grained, w/ context.}")
    table_lines.append(r"\label{tab:by_length}")
    table_lines.append(r"\end{table}")

    return "\n".join(table_lines)


def main():
    # Define length bins
    length_bins = {
        "2--3": (2, 3),
        "4--6": (4, 6),
        "7--9": (7, 9),
        "10--12": (10, 12),
        "12+": (13, None)  # 13 and above
    }

    # Paths to prediction files
    enc_diff_file = "/home/pretam-pg/DiffusionSL/inference_results/necti_finegrain_with_ctx/test_predictions.json"
    local_ref_file = "/home/pretam-pg/DiffusionSL/inference_results/hierarchial_window_7/test_predictions.json"

    print("Analyzing Enc-Diffusion results...")
    enc_diff_results = analyze_by_length(enc_diff_file, length_bins)

    print("Analyzing + Local Refinement results...")
    local_ref_results = analyze_by_length(local_ref_file, length_bins)

    # Print detailed results
    print("\n" + "="*80)
    print("RESULTS BY COMPOUND LENGTH")
    print("="*80)

    for bin_name in length_bins.keys():
        ed = enc_diff_results[bin_name]
        lr = local_ref_results[bin_name]

        print(f"\n{bin_name} tokens:")
        print(f"  # Compounds: {ed['num_compounds']}")
        print(f"  Enc-Diffusion:     USS={ed['uss']:.2f}%  LSS={ed['lss']:.2f}%  EM={ed['em']:.2f}%")
        print(f"  + Local Refine:    USS={lr['uss']:.2f}%  LSS={lr['lss']:.2f}%  EM={lr['em']:.2f}%")
        print(f"  Improvement:       USS={lr['uss']-ed['uss']:+.2f}%  LSS={lr['lss']-ed['lss']:+.2f}%  EM={lr['em']-ed['em']:+.2f}%")

    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print()
    latex_table = format_latex_table(enc_diff_results, local_ref_results, length_bins)
    print(latex_table)

    # Save to file
    output_file = "/home/pretam-pg/DiffusionSL/table_by_length.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\n\nLaTeX table saved to: {output_file}")

    # Also save detailed results as JSON
    results_json = {
        'enc_diffusion': enc_diff_results,
        'local_refinement': local_ref_results
    }

    json_file = "/home/pretam-pg/DiffusionSL/results_by_length.json"
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Detailed results saved to: {json_file}")


if __name__ == "__main__":
    main()
