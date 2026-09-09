"""
generate_executive_summary_pdf.py
---------------------------------
Generates a one-page executive summary PDF for LinkedIn publication.
Designed for portfolio managers and risk professionals.

Design principles:
- First-principles thinking: Question → Test → Evidence → Conclusion
- Executive-summary tone: Concise, scannable, decision-focused
- No overclaiming: Objective, data-driven, appropriately cautious
- One page maximum, readable font sizes
- Professional Times New Roman font
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================================================================
# REGISTER TIMES NEW ROMAN FONTS
# ============================================================================

try:
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', 'Times New Roman Bold.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-Italic', 'Times New Roman Italic.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-BoldItalic', 'Times New Roman Bold Italic.ttf'))
    FONT_AVAILABLE = True
except:
    FONT_AVAILABLE = False
    print("Note: Times New Roman not found. Using fallback fonts.")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PAPER_DIR = OUTPUT_DIR / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = PAPER_DIR / "Factor_Exposure_Sentinel_Executive_Summary.pdf"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "all_runs.csv"
VISUALIZATION_PATH = PAPER_DIR / "custom_visualization.png"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_latest_results():
    """Load the most recent XGBoost run results from all_runs.csv."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")
    
    df = pd.read_csv(RESULTS_CSV)
    xgb_runs = df[df['model'].str.contains('xgboost', case=False)].sort_values('timestamp')
    
    if len(xgb_runs) == 0:
        raise ValueError("No XGBoost runs found in results")
    
    latest = xgb_runs.iloc[-1]
    return latest


def get_target_stats():
    """Target statistics from the project (matching pipeline output)."""
    return {
        'total_days': 3753,
        'total_events': 338,
        'event_rate': 9.01,
        'crisis_rate': 34.9,
        'normal_rate': 8.4,
        'crisis_ratio': 4.15,
        'total_clusters': 33,
        'train_samples': 1497,
        'val_samples': 1499,
        'test_samples': 477,
        'test_events': 27,
    }

# ============================================================================
# CUSTOM VISUALIZATION
# ============================================================================

def create_custom_visualization(results, target_stats, output_path):
    """
    Create a custom, single visualization for the LinkedIn executive summary.
    Shows model performance vs random baseline with false alarm annotation.
    
    Uses CORRECT test set results.
    """
    import matplotlib
    matplotlib.use('Agg')  # Backend for non-interactive use
    import matplotlib.pyplot as plt
    
    # ============================================================
    # Data Setup (CORRECT Test Set Results)
    # ============================================================
    
    # Heuristic rule is deterministic (FCI > 90th percentile)
    heuristic_auc = 0.5115
    heuristic_precision = 0.0598
    
    # XGBoost - from latest run in all_runs.csv
    xgb_auc = results['auc_roc']
    xgb_ci_lower = results['ci_lower']
    xgb_ci_upper = results['ci_upper']
    xgb_precision = results['precision']
    
    # Random Forest - deterministic (same seed, same data)
    # From run 20260909_015018
    rf_auc = 0.4226
    rf_ci_lower = 0.2887
    rf_ci_upper = 0.5460
    rf_precision = 0.2143
    
    # Logistic Regression - deterministic (same seed, same data)
    # From run 20260909_015018
    lr_auc = 0.3004
    lr_ci_lower = 0.2073
    lr_ci_upper = 0.4051
    lr_precision = 0.0
    
    # Build models list
    models = [
        ('Heuristic Rule (FCI > 90%)', heuristic_auc, None, None),
        ('XGBoost', xgb_auc, xgb_ci_lower, xgb_ci_upper),
        ('Random Forest', rf_auc, rf_ci_lower, rf_ci_upper),
        ('Logistic Regression', lr_auc, lr_ci_lower, lr_ci_upper),
    ]
    
    # Sort by AUC (ascending for readability)
    models.sort(key=lambda x: x[1])
    
    # Extract data
    names = [m[0] for m in models]
    aucs = [m[1] for m in models]
    ci_lowers = [m[2] if m[2] is not None else m[1] for m in models]
    ci_uppers = [m[3] if m[3] is not None else m[1] for m in models]
    
    # ============================================================
    # Create Figure
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    
    # Color: Blue for all models
    bar_colors = ['#1f77b4'] * len(models)
    
    # Plot bars
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, aucs, height=0.6, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add error bars for models with CIs
    for i, (auc, ci_lo, ci_up) in enumerate(zip(aucs, ci_lowers, ci_uppers)):
        if ci_lo != auc or ci_up != auc:
            ax.errorbar(auc, i, xerr=[[auc - ci_lo], [ci_up - auc]], 
                       fmt='none', color='black', capsize=3, capthick=1, linewidth=1.2)
    
    # Random baseline
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Random (AUC = 0.5)')
    
    # Add values on bars
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Find XGBoost index for annotation
    xgb_idx = None
    for i, name in enumerate(names):
        if 'XGBoost' in name:
            xgb_idx = i
            break
    
    # Add false alarm annotation for XGBoost
    if xgb_idx is not None and xgb_precision > 0:
        false_alarms = int(1 / xgb_precision)
        
        ax.annotate(f'1 in {false_alarms} alerts\nis correct', 
                    xy=(aucs[xgb_idx], xgb_idx),
                    xytext=(0.55, xgb_idx + 0.3),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                    fontsize=8, color='red', fontweight='bold')
    
    # Labels and Title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('AUC-ROC (Higher is Better)', fontsize=9)
    ax.set_xlim(0.1, 0.75)
    ax.set_title('Model Performance vs Random Baseline', fontsize=11, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=7)
    
    # Context annotations
    ax.text(0.02, 0.95, 
            f"Test: {target_stats['test_samples']} samples\n"
            f"Events: {target_stats['test_events']}\n"
            f"Event Rate: {target_stats['event_rate']}%",
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', edgecolor='black', linewidth=0.5))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"   ✅ Custom visualization saved: {output_path}")
    return output_path

# ============================================================================
# STYLES
# ============================================================================

def get_styles():
    styles = getSampleStyleSheet()
    
    if FONT_AVAILABLE:
        regular = 'TimesNewRoman'
        bold = 'TimesNewRoman-Bold'
        italic = 'TimesNewRoman-Italic'
        bold_italic = 'TimesNewRoman-BoldItalic'
    else:
        regular = 'Times-Roman'
        bold = 'Times-Bold'
        italic = 'Times-Italic'
        bold_italic = 'Times-BoldItalic'
    
    # Main title - bold, centered
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'],
        fontName=bold,
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    
    # Subtitle
    subtitle_style = ParagraphStyle(
        'SubtitleStyle', parent=styles['Normal'],
        fontName=bold,
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    
    # Author
    author_style = ParagraphStyle(
        'AuthorStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    
    # Section headers
    section_style = ParagraphStyle(
        'SectionStyle', parent=styles['Heading2'],
        fontName=bold,
        fontSize=9.5,
        leading=12,
        spaceBefore=3,
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    
    # Body text
    body_style = ParagraphStyle(
        'BodyStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8,
        leading=10,
        alignment=TA_JUSTIFY,
        spaceAfter=2,
    )
    
    # Bullet text
    bullet_style = ParagraphStyle(
        'BulletStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        spaceAfter=1,
        leftIndent=8,
        bulletIndent=0,
    )
    
    # Caption
    caption_style = ParagraphStyle(
        'CaptionStyle', parent=styles['Normal'],
        fontName=italic,
        fontSize=7,
        leading=8,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    
    # Footer
    footer_style = ParagraphStyle(
        'FooterStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=6.5,
        leading=8,
        alignment=TA_CENTER,
        spaceAfter=1,
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'author': author_style,
        'section': section_style,
        'body': body_style,
        'bullet': bullet_style,
        'caption': caption_style,
        'footer': footer_style,
        'bold': bold,
        'regular': regular,
        'italic': italic,
    }

# ============================================================================
# PDF GENERATION
# ============================================================================

def format_ci(lower, upper, decimals=3):
    if pd.isna(lower) or pd.isna(upper):
        return "N/A"
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def generate_pdf():
    """Generate the one-page executive summary PDF."""
    
    # Load data
    try:
        results = load_latest_results()
        print(f"Loaded results from: {RESULTS_CSV}")
        print(f"  XGBoost AUC: {results['auc_roc']:.4f}, CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}")
        print("Using fallback values from run 20260909_015018.")
        results = {
            'auc_roc': 0.4228,
            'ci_lower': 0.3398,
            'ci_upper': 0.5068,
            'precision': 0.0244,
            'recall': 0.0370,
            'f1': 0.0294,
            'ece': 0.0195,
            'is_significant': False,
            'verdict': 'WARNING - Not significant (CI includes 0.5)',
            'timestamp': datetime.now().isoformat(),
        }
    
    target_stats = get_target_stats()
    styles = get_styles()
    
    # Calculate key metrics for the summary
    precision = results['precision']
    false_alarms_per_correct = int(1 / precision) if precision > 0 else 0
    
    # Create custom visualization
    create_custom_visualization(results, target_stats, VISUALIZATION_PATH)
    
    story = []
    
    # ========================================================================
    # HEADER: The Hook
    # ========================================================================
    story.append(Paragraph("Factor Exposure Sentinel", styles['title']))
    story.append(Paragraph("Testing Whether Public Data Can Predict Factor Crowding", styles['subtitle']))
    story.append(Paragraph(
        f"Ken Ira Lacson Talingting | {datetime.now().strftime('%B %Y')}",
        styles['author']
    ))
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 1: THE QUESTION
    # ========================================================================
    story.append(Paragraph("The Question", styles['section']))
    story.append(Paragraph(
        "Can public data predict hidden factor concentration in portfolios 21 days before it causes a drawdown?",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 2: WHY IT MATTERS (Problem Statement)
    # ========================================================================
    story.append(Paragraph("Why It Matters", styles['section']))
    story.append(Paragraph(
        "Risk systems monitor single stocks and sectors. They miss when a portfolio secretly bets everything "
        "on a single factor — Momentum, Value, or Low-Beta.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "This contributed to the 2007 quant crisis, the 2018 volatility shock, and the 2020 COVID crash.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 3: WHAT WE DID (Method)
    # ========================================================================
    story.append(Paragraph("What We Did", styles['section']))
    story.append(Paragraph(
        "Built a reproducible pipeline using first-principles reasoning: "
        "threshold rules → logistic regression → Random Forest → XGBoost.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "Applied statistical rigor: bootstrap confidence intervals, calibration testing, time-aware splitting.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 4: THE EVIDENCE (Custom Visualization)
    # ========================================================================
    if VISUALIZATION_PATH.exists():
        img = Image(str(VISUALIZATION_PATH), width=4.5 * inch, height=3.0 * inch)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Spacer(1, 1))
        story.append(Paragraph(
            "Figure: Model performance vs random baseline. All models perform at or below random.",
            styles['caption']
        ))
        story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 5: THE NUMBERS (Key Results)
    # ========================================================================
    story.append(Paragraph("The Numbers", styles['section']))
    
    # Table: Model Performance (Using correct test set values)
    xgb_auc = results['auc_roc']
    xgb_ci_lower = results['ci_lower']
    xgb_ci_upper = results['ci_upper']
    
    data = [
        ["Model", "AUC-ROC", "95% CI", "Significant?"],
        ["Simple Rule (FCI > 90%)", "0.5115", "N/A", "N/A"],
        ["XGBoost", f"{xgb_auc:.4f}", 
         f"[{xgb_ci_lower:.4f}, {xgb_ci_upper:.4f}]", "NO"],
        ["Random Forest", "0.4226", "[0.2887, 0.5460]", "NO"],
        ["Logistic Regression", "0.3004", "[0.2073, 0.4051]", "NO"],
    ]
    
    table = Table(data, colWidths=[1.3 * inch, 0.6 * inch, 1.1 * inch, 0.7 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.85, 0.85, 0.85)),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
    ]))
    story.append(table)
    story.append(Spacer(1, 1))
    
    story.append(Paragraph(
        f"<b>At the optimal threshold, 1 in {false_alarms_per_correct} alerts is correct.</b> "
        f"XGBoost and Random Forest confidence intervals include 0.5, indicating no statistically significant predictive performance. "
        f"Logistic Regression performed significantly worse than random (AUC = 0.3004).",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 6: WHY IT DIDN'T WORK (Honest Assessment)
    # ========================================================================
    story.append(Paragraph("What We Found", styles['section']))
    story.append(Paragraph(
        "All feature correlations were below 0.1 — the strongest (log VIX) was 0.077. "
        "This suggests the available features carry very little individual predictive signal.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "With only 27 test events, the statistical power to detect a real effect is limited. "
        "The null hypothesis cannot be rejected, but this may be due to insufficient power rather than "
        "the absence of any relationship.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 7: WHAT THIS MEANS (Practical Takeaways)
    # ========================================================================
    story.append(Paragraph("Implications", styles['section']))
    
    takeaways = [
        "<b>For Portfolio Managers:</b> Public data alone may not predict factor crowding. Consider positioning, flows, or short interest data.",
        "<b>For Quant Researchers:</b> Use this as a reproducible baseline. Test your proprietary data against it.",
        "<b>For Risk Managers:</b> Focus on structural risk monitoring (VaR, stress testing) rather than prediction from public data.",
    ]
    
    for takeaway in takeaways:
        story.append(Paragraph(takeaway, styles['bullet']))
    
    story.append(Spacer(1, 1))
    
    # ========================================================================
    # SECTION 8: THE TAKEAWAY
    # ========================================================================
    story.append(Paragraph("Takeaway", styles['section']))
    story.append(Paragraph(
        "This project tested a well-defined hypothesis with rigorous methodology. "
        "<b>The result: no statistically significant predictive signal from public data.</b>",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "That's a useful finding, even if it's not the answer we hoped for.",
        styles['body']
    ))
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    story.append(Paragraph(
        "<font size=6><b>Full methodology, code, and results are open-source:</b> "
        "https://github.com/kira-ml/factor-exposure-sentinel</font>",
        styles['footer']
    ))
    story.append(Paragraph(
        "<font size=5 color='grey'>Disclaimer: Educational purposes only. Not investment advice.</font>",
        styles['footer']
    ))
    
    # ========================================================================
    # BUILD PDF (1 Page Only)
    # ========================================================================
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=LETTER,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.3 * inch,
    )
    
    doc.build(story)
    
    print(f"\nPDF generated: {OUTPUT_PDF}")
    print(f"File size: {OUTPUT_PDF.stat().st_size / 1024:.1f} KB")
    
    # Verify it's 1 page
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(OUTPUT_PDF))
        num_pages = len(reader.pages)
        print(f"Pages: {num_pages}")
        if num_pages > 1:
            print("⚠️ WARNING: PDF is more than 1 page! Reduce content or font sizes.")
    except:
        pass
    
    return OUTPUT_PDF


if __name__ == "__main__":
    print("="*60)
    print("FACTOR EXPOSURE SENTINEL - EXECUTIVE SUMMARY GENERATOR")
    print("="*60)
    print()
    
    try:
        generate_pdf()
        print()
        print("="*60)
        print("Done.")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()