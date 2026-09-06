"""
generate_executive_summary_pdf.py
---------------------------------
Generates a one-page executive summary PDF for the Factor Exposure Sentinel project.
Designed for LinkedIn attachment and PM distribution.

Design principles:
- First-principles thinking: Question -> Deconstruct -> Evidence -> Conclusion
- Executive-summary tone: Concise, structured, evidence-based
- No overclaiming: Objective, data-driven, appropriately cautious
- One page maximum, balanced font sizes for readability
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

# Try to register Times New Roman (Windows)
try:
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', 'Times New Roman Bold.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-Italic', 'Times New Roman Italic.ttf'))
    pdfmetrics.registerFont(TTFont('TimesNewRoman-BoldItalic', 'Times New Roman Bold Italic.ttf'))
    FONT_AVAILABLE = True
except:
    # Fallback to built-in fonts
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

# ⚠️ CHANGED: Now using Figure 6 (Precision-Recall) - The most impactful graph
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
# STYLES - TIMES NEW ROMAN
# ============================================================================

def get_styles():
    """Create paragraph styles for the executive summary with Times New Roman."""
    styles = getSampleStyleSheet()
    
    # Font names
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
    
    # Title
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'],
        fontName=bold,
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    
    # Subtitle / Author
    subtitle_style = ParagraphStyle(
        'SubtitleStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    
    # Section headers
    section_style = ParagraphStyle(
        'SectionStyle', parent=styles['Heading2'],
        fontName=bold,
        fontSize=10.5,
        leading=12,
        spaceBefore=5,
        spaceAfter=3,
        alignment=TA_LEFT,
    )
    
    # Body text
    body_style = ParagraphStyle(
        'BodyStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8.5,
        leading=10.5,
        alignment=TA_JUSTIFY,
        spaceAfter=3,
    )
    
    # Bullet text (for deconstruction)
    bullet_style = ParagraphStyle(
        'BulletStyle', parent=styles['Normal'],
        fontName=regular,
        fontSize=8.5,
        leading=10.5,
        alignment=TA_JUSTIFY,
        spaceAfter=2,
        leftIndent=12,
        bulletIndent=0,
    )
    
    # Caption
    caption_style = ParagraphStyle(
        'CaptionStyle', parent=styles['Normal'],
        fontName=italic,
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        spaceAfter=4,
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

def format_ci(lower, upper, decimals=4):
    """Format confidence interval."""
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
            'auc_roc': 0.4949,
            'ci_lower': 0.4057,
            'ci_upper': 0.5842,
            'precision': 0.0377,
            'recall': 0.0741,
            'f1': 0.0500,
            'ece': 0.0258,
            'is_significant': False,
            'verdict': 'WARNING - Not significant (CI includes 0.5)',
            'timestamp': datetime.now().isoformat(),
        }
    
    target_stats = get_target_stats()
    styles = get_styles()
    
    # Build story
    story = []
    
    # ========================================================================
    # HEADER
    # ========================================================================
    story.append(Paragraph("Factor Exposure Sentinel", styles['title']))
    story.append(Paragraph("An Empirical Assessment of Public Data", styles['subtitle']))
    story.append(Paragraph(
        f"Ken Ira Lacson Talingting | {datetime.now().strftime('%B %Y')}",
        styles['subtitle']
    ))
    story.append(Spacer(1, 4))
    
    # ========================================================================
    # SECTION 1: THE CORE QUESTION (First-Principles Framing)
    # ========================================================================
    story.append(Paragraph("The Core Question", styles['section']))
    story.append(Paragraph(
        "Traditional risk systems monitor asset-level concentration. "
        "Before building a model, one fundamental question was addressed: "
        "<i>Can a portfolio's hidden factor concentration be reliably predicted 21 days in advance, "
        "using only publicly available market data?</i>",
        styles['body']
    ))
    story.append(Spacer(1, 3))
    
    # ========================================================================
    # SECTION 2: DECONSTRUCTING THE REQUIREMENT
    # ========================================================================
    story.append(Paragraph("Deconstructing the Requirement", styles['section']))
    story.append(Paragraph(
        "For a model to be practically useful, it must clear three foundational hurdles:",
        styles['body']
    ))
    story.append(Paragraph(
        "<b>1. A Definable Target:</b> Can an objective 'factor event' be defined? "
        "(Yes: Drawdown < -3% over 21 days).",
        styles['bullet']
    ))
    story.append(Paragraph(
        "<b>2. Sufficient Information:</b> Do public features contain non-zero predictive signal? "
        "(This required empirical testing).",
        styles['bullet']
    ))
    story.append(Paragraph(
        "<b>3. Measurable Value:</b> Does nonlinear complexity outperform a simple rule?",
        styles['bullet']
    ))
    story.append(Spacer(1, 4))
    
    # ========================================================================
    # SECTION 3: THE EVIDENCE (Figure 6)
    # ========================================================================
    if FIGURE_PATH.exists():
        img = Image(str(FIGURE_PATH), width=5.0 * inch, height=3.5 * inch)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Spacer(1, 1))
        story.append(Paragraph(
            "Figure: Precision-Recall curve. The model (blue) cannot break above the random baseline (gray) at the reported threshold (red dot).",
            styles['caption']
        ))
        story.append(Spacer(1, 3))
    
    # ========================================================================
    # SECTION 4: WHAT THE DATA SAYS (Objective Findings)
    # ========================================================================
    story.append(Paragraph("What the Data Indicates", styles['section']))
    story.append(Paragraph(
        f"<b>Signal Strength:</b> The strongest individual public predictor (log VIX) holds a "
        f"correlation of only 0.077 with the target. This is below the threshold of meaningful information content.",
        styles['body']
    ))
    story.append(Paragraph(
        f"<b>Model Performance:</b> XGBoost achieved an AUC of {results['auc_roc']:.4f} "
        f"(95% CI: {format_ci(results['ci_lower'], results['ci_upper'], 4)}). "
        f"The confidence interval includes 0.5, meaning the result is statistically indistinguishable from random noise.",
        styles['body']
    ))
    story.append(Paragraph(
        "<b>Complexity Value:</b> A simple heuristic performed comparably, suggesting that "
        "adding architectural complexity did not extract additional signal from these features.",
        styles['body']
    ))
    story.append(Spacer(1, 3))
    
    # ========================================================================
    # SECTION 5: CONCLUSION (Humble, Data-Driven)
    # ========================================================================
    story.append(Paragraph("Conclusion", styles['section']))
    story.append(Paragraph(
        "The fundamental limitation appears to be the information content of the inputs, "
        "rather than the model architecture. When feature correlations are < 0.1, "
        "no amount of complexity can extract signal that isn't present. "
        "Public data appears insufficient for this specific task. "
        "Reliable prediction may require proprietary inputs (positioning, flows, short interest).",
        styles['body']
    ))
    story.append(Spacer(1, 3))
    
    # ========================================================================
    # SECTION 6: CONTRIBUTION
    # ========================================================================
    story.append(Paragraph("Contribution", styles['section']))
    story.append(Paragraph(
        "This project establishes a reproducible benchmark. It allows researchers "
        "to test whether their proprietary data adds measurable signal beyond what "
        "public factors provide.",
        styles['body']
    ))
    story.append(Spacer(1, 4))
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    story.append(Paragraph(
        "<font size=7><b>Full code, methodology, and statistical tests are open-source:</b> "
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
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
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