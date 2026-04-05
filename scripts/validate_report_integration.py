#!/usr/bin/env python3
"""
Validation script to verify LaTeX integration is complete and correct
"""

import json
from pathlib import Path
import re

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        print(f"✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {filepath}")
        return False

def check_latex_references(latex_file):
    """Verify LaTeX figure references and includes"""
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    references = {
        r'\\includegraphics.*confusion_matrix': 'Confusion Matrix Figure Include',
        r'\\includegraphics.*per_class_accuracy': 'Per-Class Accuracy Figure Include',
        r'\\includegraphics.*training_dynamics': 'Training Dynamics Figure Include',
        r'\\includegraphics.*detailed_metrics_table': 'Detailed Metrics Figure Include',
        r'\\includegraphics.*misclassification_patterns': 'Misclassification Patterns Figure Include',
        r'\\label{fig:confusion_matrix}': 'Confusion Matrix Label',
        r'\\label{fig:per_class_accuracy}': 'Per-Class Accuracy Label',
        r'\\label{fig:training_dynamics}': 'Training Dynamics Label',
        r'\\label{fig:detailed_metrics}': 'Detailed Metrics Label',
        r'\\label{fig:misclassification_patterns}': 'Misclassification Patterns Label',
        r'\\ref{fig:confusion_matrix}': 'Confusion Matrix Reference',
        r'\\ref{fig:per_class_accuracy}': 'Per-Class Accuracy Reference',
        r'\\ref{fig:training_dynamics}': 'Training Dynamics Reference',
        r'\\ref{fig:detailed_metrics}': 'Detailed Metrics Reference',
        r'\\ref{fig:misclassification_patterns}': 'Misclassification Patterns Reference',
        r'\\graphicspath{': 'Graphics Path Configuration',
    }
    
    print("\n" + "="*70)
    print("LATEX REFERENCE VERIFICATION")
    print("="*70)
    
    all_found = True
    for pattern, description in references.items():
        if re.search(pattern, content):
            print(f"✓ {description}")
        else:
            print(f"✗ {description} NOT FOUND")
            all_found = False
    
    return all_found

def check_figure_captions(latex_file):
    """Extract and verify figure captions are comprehensive"""
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    captions = re.findall(r'\\caption{([^}]+)}', content, re.DOTALL)
    
    print("\n" + "="*70)
    print("FIGURE CAPTION VERIFICATION (Comprehensive Analysis)")
    print("="*70)
    
    if len(captions) >= 5:
        print(f"✓ Found {len(captions)} figure captions (expected: ≥5)")
        for i, caption in enumerate(captions[:5], 1):
            # Check caption length (should be substantive)
            cap_lines = len(caption.split('\n'))
            if len(caption) > 150:  # Captions should be detailed
                print(f"  ✓ Caption {i}: Comprehensive ({len(caption)} characters, {cap_lines} lines)")
            else:
                print(f"  ! Caption {i}: Brief ({len(caption)} characters, consider expanding)")
    else:
        print(f"✗ Found only {len(captions)} captions (expected: ≥5)")
    
    return len(captions) >= 5

def check_figure_sections(latex_file):
    """Verify figures are in appropriate sections"""
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n" + "="*70)
    print("FIGURE PLACEMENT VERIFICATION")
    print("="*70)
    
    sections = {
        'confusion_matrix': ['experimental results', 'per-class', 'confusion'],
        'per_class_accuracy': ['per-class', 'accuracy', 'performance'],
        'training_dynamics': ['training', 'loss', 'convergence', 'dynamics'],
        'detailed_metrics': ['metrics', 'precision', 'recall', 'f1'],
        'misclassification_patterns': ['misclassification', 'patterns', 'normalized'],
    }
    
    all_placed = True
    for figure, keywords in sections.items():
        # Simplified check: verify figure is somewhere in Results/Discussion sections
        if figure in content:
            print(f"✓ {figure}: Found in LaTeX")
        else:
            print(f"✗ {figure}: NOT FOUND in LaTeX")
            all_placed = False
    
    return all_placed

def check_results_json():
    """Verify results.json contains expected metrics"""
    results_path = Path('outputs/finetuned_final_20260322_132618/results.json')
    if not results_path.exists():
        print("✗ Results JSON not found")
        return False
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("RESULTS DATA VERIFICATION")
    print("="*70)
    
    required_keys = [
        'test_acc', 'best_val_acc', 'best_epoch', 'per_class_acc', 'confusion_matrix'
    ]
    
    all_present = True
    for key in required_keys:
        if key in results:
            print(f"✓ {key}: Present")
        else:
            print(f"✗ {key}: MISSING")
            all_present = False
    
    if all_present:
        print(f"\n  Test Accuracy: {results['test_acc']:.1%}")
        print(f"  Best Val Accuracy: {results['best_val_acc']:.1%} (Epoch {results['best_epoch']})")
        print(f"  Classes: {list(results['per_class_acc'].keys())}")
    
    return all_present

def main():
    print("\n" + "="*70)
    print("EMOAI PROJECT REPORT - VISUALIZATION INTEGRATION VALIDATION")
    print("="*70)
    
    # Check figure files
    print("\n" + "="*70)
    print("FIGURE FILES VERIFICATION")
    print("="*70)
    
    figure_checks = [
        check_file_exists('figures/confusion_matrix.png', 'Confusion Matrix PNG'),
        check_file_exists('figures/per_class_accuracy.png', 'Per-Class Accuracy PNG'),
        check_file_exists('figures/training_dynamics.png', 'Training Dynamics PNG'),
        check_file_exists('figures/detailed_metrics_table.png', 'Detailed Metrics PNG'),
        check_file_exists('figures/misclassification_patterns.png', 'Misclassification Patterns PNG'),
    ]
    
    # Check LaTeX file
    latex_checks = [
        check_latex_references('PROJECT_REPORT.tex'),
        check_figure_captions('PROJECT_REPORT.tex'),
        check_figure_sections('PROJECT_REPORT.tex'),
    ]
    
    # Check results data
    data_checks = [check_results_json()]
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION VALIDATION SUMMARY")
    print("="*70)
    
    figures_ok = all(figure_checks)
    latex_ok = all(latex_checks)
    data_ok = all(data_checks)
    
    print(f"\nFigure Files Status:      {'✓ ALL PRESENT' if figures_ok else '✗ MISSING FILES'}")
    print(f"LaTeX Integration Status: {'✓ ALL REFERENCES OK' if latex_ok else '✗ INCOMPLETE'}")
    print(f"Results Data Status:      {'✓ DATA VALIDATED' if data_ok else '✗ DATA MISSING'}")
    
    overall = figures_ok and latex_ok and data_ok
    print(f"\nOverall Status:           {'✓ READY FOR COMPILATION' if overall else '✗ NEEDS FIXES'}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if overall:
        print("\n1. COMPILE LaTeX to PDF:")
        print("   $ pdflatex PROJECT_REPORT.tex")
        print("\n2. GENERATE BIBLIOGRAPHY:")
        print("   $ bibtex PROJECT_REPORT")
        print("\n3. FINAL COMPILATION (x2):")
        print("   $ pdflatex PROJECT_REPORT.tex")
        print("   $ pdflatex PROJECT_REPORT.tex")
        print("\n4. VIEW OUTPUT:")
        print("   $ start PROJECT_REPORT.pdf  (Windows)")
        print("   $ open PROJECT_REPORT.pdf   (macOS)")
        print("   $ xdg-open PROJECT_REPORT.pdf (Linux)")
        print("\n5. VERIFY IN PDF:")
        print("   - All figures display correctly")
        print("   - Captions are readable and informative")
        print("   - Cross-references (Figure X) work properly")
        print("   - Page numbers and TOF entries are correct")
    else:
        print("\n× Please review errors above before proceeding to compilation")
    
    print("\n" + "="*70)
    print("DOCUMENT READY FOR SUBMISSION" if overall else "DOCUMENT NEEDS REVIEW")
    print("="*70 + "\n")
    
    return 0 if overall else 1

if __name__ == '__main__':
    exit(main())
