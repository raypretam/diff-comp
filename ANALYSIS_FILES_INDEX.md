# Analysis Results - File Index

All scripts and results for Enc-Diffusion vs. Local Refinement comparison.

## 📊 Analysis Scripts

### 1. `analyze_by_length.py`
Analyzes USS/LSS/EM by compound length bins (2-3, 4-6, 7-9, 10-12, 12+ tokens).

**Usage:**
```bash
python analyze_by_length.py
```

### 2. `analyze_error_types.py`
Categorizes predictions into error types: correct, label error, span error, missed.

**Usage:**
```bash
python analyze_error_types.py
```

### 3. `analyze_per_class.py`
Analyzes performance by compound class (Tatpuruṣa, Dvandva, Bahuvrīhi, etc.).

**Usage:**
```bash
python analyze_per_class.py
```

## 📄 Generated Results

### For Paper Section 1: By Length
- `table_by_length.tex` - LaTeX table ready to paste
- `results_by_length.json` - Detailed numerical results

### For Paper Section 2: Error Types
- `error_types_paragraph.txt` - Formatted paragraph
- `error_type_analysis.json` - Detailed numerical results

### For Paper Section 3: Per-Class
- `per_class_paragraph.txt` - Formatted paragraph
- `per_class_table.tex` - LaTeX table ready to paste
- `per_class_analysis.json` - Detailed numerical results

### Summary Document
- `ANALYSIS_SUMMARY.txt` - Comprehensive overview of all findings with interpretation

## 📈 Key Findings at a Glance

1. **By Length:** Local Refinement improves across all lengths, with biggest EM gains on longer compounds (7+ tokens)

2. **Error Types:** Local Refinement reduces ALL error types (no trade-offs):
   - Label errors: 28.7% → 28.1%
   - Span errors: 6.6% → 6.2%
   - Missed: 1.6% → 1.0%

3. **Per-Class:** Surprising finding - Dvandva benefits most (+3.8%), not Tatpuruṣa (+0.1%)

## 🔄 Rerunning Analyses

All scripts work with the prediction files in:
- `inference_results/necti_finegrain_with_ctx/` (Enc-Diffusion)
- `inference_results/hierarchial_window_7/` (Local Refinement)

To rerun with different data, update the file paths in each script's `main()` function.

## 📝 LaTeX Integration

Copy the following files directly into your LaTeX paper:

```latex
% Section: Performance by Compound Length
\input{table_by_length.tex}

% Section: Per-Class Analysis
\input{per_class_table.tex}
```

For the paragraphs, use the content from:
- `error_types_paragraph.txt`
- `per_class_paragraph.txt`
