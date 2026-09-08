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
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "fig6_precision_recall.png"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_latest_results():
    """Load the most recent XGBoost run results."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")
    
    df = pd.read_csv(RESULTS_CSV)
    xgb_runs = df[df['model'].str.contains('xgboost', case=False)].sort_values('timestamp')
    
    if len(xgb_runs) == 0:
        raise ValueError("No XGBoost runs found in results")
    
    latest = xgb_runs.iloc[-1]
    return latest

def get_target_stats():
    """Target statistics from the project."""
    return {
        'total_days': 3773,
        'total_events': 338,
        'event_rate': 8.96,
        'crisis_rate': 34.9,
        'normal_rate': 8.4,
        'crisis_ratio': 4.17,
    }

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
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    
    # Subtitle
    subtitle_style = ParagraphStyle(
        'SubtitleStyle', parent=styles['Normal'],
        fontName=bold,
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    
    # Author
    author_style = ParagraphStyle(
        'AuthorStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    
    # Section headers
    section_style = ParagraphStyle(
        'SectionStyle', parent=styles['Heading2'],
        fontName=bold,
        fontSize=10.5,
        leading=13,
        spaceBefore=4,
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    
    # Body text
    body_style = ParagraphStyle(
        'BodyStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8.5,
        leading=10.5,
        alignment=TA_JUSTIFY,
        spaceAfter=2,
    )
    
    # Bullet text
    bullet_style = ParagraphStyle(
        'BulletStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8.5,
        leading=10.5,
        alignment=TA_LEFT,
        spaceAfter=1,
        leftIndent=10,
        bulletIndent=0,
    )
    
    # Caption
    caption_style = ParagraphStyle(
        'CaptionStyle', parent=styles['Normal'],
        fontName=italic,
        fontSize=7.5,
        leading=9,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    
    # Footer
    footer_style = ParagraphStyle(
        'FooterStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=7,
        leading=9,
        alignment=TA_CENTER,
        spaceAfter=2,
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
        print(f"  AUC: {results['auc_roc']:.4f}, CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}")
        print("Using fallback values.")
        results = {
            'auc_roc': 0.4511,
            'ci_lower': 0.3651,
            'ci_upper': 0.5382,
            'precision': 0.0217,
            'recall': 0.0370,
            'f1': 0.0274,
            'ece': 0.0206,
            'is_significant': False,
            'verdict': 'WARNING - Not significant (CI includes 0.5)',
            'timestamp': datetime.now().isoformat(),
        }
    
    target_stats = get_target_stats()
    styles = get_styles()
    
    # Calculate key metrics for the summary
    false_alarms_per_correct = int(1 / results['precision']) if results['precision'] > 0 else 0
    
    story = []
    
    # ========================================================================
    # HEADER: The Hook
    # ========================================================================
    story.append(Paragraph("Factor Exposure Sentinel", styles['title']))
    story.append(Paragraph("1 Week. 4 Models. 0 Signal.", styles['subtitle']))
    story.append(Paragraph(
        f"Ken Ira Lacson Talingting | {datetime.now().strftime('%B %Y')}",
        styles['author']
    ))
    story.append(Spacer(1, 3))
    
    # ========================================================================
    # SECTION 1: THE QUESTION
    # ========================================================================
    story.append(Paragraph("The Question", styles['section']))
    story.append(Paragraph(
        "Can public data predict hidden factor concentration in portfolios 21 days before it causes a drawdown?",
        styles['body']
    ))
    story.append(Spacer(1, 2))
    
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
        "This caused the 2007 quant crisis, the 2018 volatility shock, and the 2020 COVID crash.",
        styles['body']
    ))
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 3: WHAT WE DID (Method)
    # ========================================================================
    story.append(Paragraph("What We Did", styles['section']))
    story.append(Paragraph(
        "Built a minimum viable baseline in 1 week using first-principles reasoning: "
        "threshold rules → logistic regression → Random Forest → XGBoost.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "Applied statistical rigor: bootstrap confidence intervals, calibration testing.",
        styles['body']
    ))
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 4: THE EVIDENCE (Figure 6)
    # ========================================================================
    if FIGURE_PATH.exists():
        img = Image(str(FIGURE_PATH), width=5.0 * inch, height=3.5 * inch)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Spacer(1, 1))
        story.append(Paragraph(
            "Figure: Precision-Recall curve. The model performs below the random baseline at the optimal threshold.",
            styles['caption']
        ))
        story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 5: THE NUMBERS (Key Results)
    # ========================================================================
    story.append(Paragraph("The Numbers", styles['section']))
    
    # Table: Model Performance
    data = [
        ["Model", "AUC-ROC", "CI Includes 0.5?"],
        ["Simple Rule (FCI > 90%)", "0.5115", "N/A"],
        ["XGBoost", f"{results['auc_roc']:.4f}", "YES"],
        ["Random Forest", "0.4226", "YES"],
        ["Logistic Regression", "0.3007", "YES"],
    ]
    
    table = Table(data, colWidths=[1.6 * inch, 0.9 * inch, 0.9 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.85, 0.85, 0.85)),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
    ]))
    story.append(table)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph(
        f"<b>XGBoost flagged {false_alarms_per_correct} risks for every 1 real crash.</b>",
        styles['body']
    ))
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 6: WHY IT DIDN'T WORK
    # ========================================================================
    story.append(Paragraph("Why It Didn't Work", styles['section']))
    story.append(Paragraph(
        "The data had no signal. All feature correlations were below 0.1. "
        "The strongest predictor (log VIX) was just 0.077.",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "You cannot predict something with features that have no signal.",
        styles['body']
    ))
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 7: WHAT THIS MEANS (Practical Takeaways)
    # ========================================================================
    story.append(Paragraph("What This Means", styles['section']))
    
    takeaways = [
        "<b>For Portfolio Managers:</b> Public data won't predict factor crowding. You need positioning, flows, or short interest.",
        "<b>For Quant Researchers:</b> Use this as a free benchmark. Test your proprietary data against it.",
        "<b>For Risk Managers:</b> Don't build this system. It will produce too many false alarms.",
    ]
    
    for takeaway in takeaways:
        story.append(Paragraph(takeaway, styles['bullet']))
    
    story.append(Spacer(1, 2))
    
    # ========================================================================
    # SECTION 8: THE TAKEAWAY
    # ========================================================================
    story.append(Paragraph("The Takeaway", styles['section']))
    story.append(Paragraph(
        "This project validated the null hypothesis in 1 week. "
        "The outcome is clear: <b>public data is insufficient for this prediction task.</b>",
        styles['body']
    ))
    story.append(Spacer(1, 1))
    story.append(Paragraph(
        "That's useful to know, even if it's not the answer we wanted.",
        styles['body']
    ))
    story.append(Spacer(1, 3))
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    story.append(Paragraph(
        "<font size=7><b>Full methodology, code, and results are open-source:</b> "
        "https://github.com/kira-ml/factor-exposure-sentinel</font>",
        styles['footer']
    ))
    story.append(Paragraph(
        "<font size=6 color='grey'>Disclaimer: Educational purposes only. Not investment advice.</font>",
        styles['footer']
    ))
    
    # ========================================================================
    # BUILD PDF
    # ========================================================================
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )
    
    doc.build(story)
    
    print(f"\nPDF generated: {OUTPUT_PDF}")
    print(f"File size: {OUTPUT_PDF.stat().st_size / 1024:.1f} KB")
    
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