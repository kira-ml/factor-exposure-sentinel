"""
generate_paper.py
-----------------
Generates a 5-10 page academic mini research paper (PDF) for the
Factor Exposure Sentinel project using ReportLab.
"""

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib import colors

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_PDF = "Factor_Exposure_Sentinel_Research_Paper.pdf"
FIGURE_DIR = r"D:\quant-finance-ml\factor-exposure-sentinel\outputs\figures"

# ============================================================================
# STYLES (Academic/Journal)
# ============================================================================
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'TitleStyle', parent=styles['Title'], fontName='Times-Roman',
    fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=20
)
author_style = ParagraphStyle(
    'AuthorStyle', parent=styles['Normal'], fontName='Times-Roman',
    fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=5
)
heading1_style = ParagraphStyle(
    'Heading1', parent=styles['Heading1'], fontName='Times-Bold',
    fontSize=14, leading=18, spaceBefore=15, spaceAfter=10
)
heading2_style = ParagraphStyle(
    'Heading2', parent=styles['Heading2'], fontName='Times-Bold',
    fontSize=12, leading=16, spaceBefore=10, spaceAfter=6
)
heading3_style = ParagraphStyle(
    'Heading3', parent=styles['Heading3'], fontName='Times-BoldItalic',
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=4
)
body_style = ParagraphStyle(
    'Body', parent=styles['Normal'], fontName='Times-Roman',
    fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=8
)
quote_style = ParagraphStyle(
    'Quote', parent=styles['Normal'], fontName='Times-Italic',
    fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=8,
    leftIndent=20, rightIndent=20
)
caption_style = ParagraphStyle(
    'Caption', parent=styles['Normal'], fontName='Times-Italic',
    fontSize=9, leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12
)

# Table style
table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
])

# ============================================================================
# HELPER: Add figures + captions
# ============================================================================

def add_figure(story, filename, caption, width=6.0 * inch):
    path = os.path.join(FIGURE_DIR, filename)
    if os.path.exists(path):
        img = Image(path, width=width, height=width * 0.75)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Paragraph(caption, caption_style))
    else:
        story.append(Paragraph(f"[Figure not found: {filename}]", body_style))


# ============================================================================
# DOCUMENT BUILDER
# ============================================================================

def build_paper():
    story = []

    # ========================================================================
    # PAGE 1: TITLE & ABSTRACT
    # ========================================================================
    story.append(Paragraph(
        "Detecting Hidden Factor Overconcentration Risk in Multi-Asset Portfolios: "
        "A Machine Learning Early-Warning Approach", title_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Ken Ira Lacson Talingting", author_style))
    story.append(Paragraph("Independent Research / Portfolio Project", author_style))
    story.append(Paragraph("September 2026", author_style))
    story.append(Paragraph("JEL Classification: G11, G17, C45, C53", author_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Abstract", heading1_style))
    story.append(Paragraph(
        "Traditional portfolio risk systems monitor asset-level concentration while "
        "systematically underemphasizing hidden factor concentration. A portfolio may "
        "appear well-diversified across hundreds of positions while simultaneously "
        "harboring massive, undiversified overweight positions to latent factors such as "
        "Value, Momentum, Carry, Quality, or Low-Beta. During market stress—exemplified by "
        "the COVID-19 crash of 2020—these hidden exposures produce severe, unexpected "
        "drawdowns that asset-level metrics fail to predict. "
        "<br/><br/>"
        "This project develops a reproducible machine learning framework to detect emerging "
        "factor overconcentration in multi-asset portfolios before it translates into "
        "catastrophic losses. The problem is framed as a supervised binary classification task. "
        "After rigorous experimentation with proper validation methodology, no statistically "
        "significant predictive relationship was found. The null hypothesis cannot be rejected "
        "with the available data. This negative result is a valuable contribution, establishing "
        "a reproducible benchmark and saving others from pursuing weak signals with public data.",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Keywords:</b> Quantitative Risk Management, Factor Concentration, Machine Learning, "
        "Early Warning Systems, Negative Result, Reproducibility", body_style
    ))
    story.append(PageBreak())

    # ========================================================================
    # PAGE 2: INTRODUCTION & RESEARCH QUESTION
    # ========================================================================
    story.append(Paragraph("1. Introduction", heading1_style))
    story.append(Paragraph(
        "The August 2007 quant crisis, the February 2018 volatility shock, and the March 2020 "
        "COVID-19 drawdown all featured significant losses driven by factor crowding invisible "
        "to standard risk dashboards. Risk management infrastructure typically focuses on "
        "name concentration (e.g., max 5% in a single stock) and sector concentration (e.g., max "
        "20% in Technology). These systems treat factor exposures as secondary outputs rather "
        "than primary risk drivers. A portfolio manager constructing a 'diversified' portfolio "
        "of 200 stocks may inadvertently create a portfolio where 80% of variance is driven by "
        "a single unobserved factor.",
        body_style
    ))
    story.append(Paragraph(
        "This project addresses the resulting gap by designing a rigorous, data-driven risk "
        "surveillance framework for institutional investors and multi-asset portfolio managers.",
        body_style
    ))

    story.append(Paragraph("1.1 Research Questions", heading2_style))
    story.append(Paragraph("<b>Primary Research Question:</b>", body_style))
    story.append(Paragraph(
        "Given a portfolio's current holdings, factor loadings, and prevailing market "
        "conditions, can a machine learning model accurately predict whether the portfolio is "
        "entering a state of dangerous factor overconcentration that will result in a "
        "significant drawdown?", quote_style
    ))
    story.append(Paragraph("<b>Secondary Research Questions:</b>", body_style))
    story.append(Paragraph(
        "1. Which crowding proxies (FCI, VIX, credit spreads) carry the strongest predictive "
        "signal for downside risk?<br/>"
        "2. Do nonlinear machine learning models offer a meaningful improvement over "
        "interpretable linear baselines?<br/>"
        "3. How does the predictive signal vary across distinct market regimes?",
        body_style
    ))
    story.append(Paragraph(
        "We adhere to a baseline-first, rigorous-evaluation philosophy. Simple threshold rules "
        "and logistic regression are established before introducing tree-based ensemble methods. "
        "Complexity is only adopted if it yields statistically significant improvements in "
        "out-of-sample AUC-ROC. This approach ensures that the practical value of any model is "
        "empirically justified rather than assumed.",
        body_style
    ))
    story.append(PageBreak())

    # ========================================================================
    # PAGE 3: PROBLEM FORMULATION & DATA
    # ========================================================================
    story.append(Paragraph("2. Problem Formulation", heading1_style))
    story.append(Paragraph("2.1 Target Variable", heading2_style))
    story.append(Paragraph(
        "A binary target indicates a 'Factor Concentration Event'. Based on empirical testing, "
        "the optimal drawdown threshold of -3% over 21 trading days was validated. "
        "The prediction horizon is 21 trading days (1 month).",
        body_style
    ))

    # Table 1: Target Analysis
    target_data = [
        ["Metric", "Value"],
        ["Total Days", "3,773"],
        ["Total Events", "338"],
        ["Event Rate", "8.96%"],
        ["Crisis Event Rate (COVID-19)", "34.9%"],
        ["Normal Event Rate", "8.4%"],
        ["Crisis / Normal Ratio", "4.17x"],
        ["Total Clusters", "33"],
        ["Largest Cluster", "29 events (Jan-Mar 2020)"]
    ]
    target_table = Table(target_data, colWidths=[2.5 * inch, 2.5 * inch])
    target_table.setStyle(table_style)
    story.append(target_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Key Update:</b> Based on empirical testing, the attribution threshold (originally >60%) "
        "was removed because it destroyed the predictive signal. The optimal drawdown threshold "
        "of -3% was validated through grid search (AUC 0.7528, p < 0.001 on validation set).",
        body_style
    ))

    story.append(Paragraph("2.2 Data Sources", heading2_style))
    data_sources = [
        ["Source", "Variable", "Frequency"],
        ["Kenneth French Library", "Fama-French 5-Factors", "Daily"],
        ["Yahoo Finance (yfinance)", "ETF Prices (SPY, AGG, GLD, IJS, EFA)", "Daily"],
        ["Yahoo Finance (yfinance)", "VIX (^VIX)", "Daily"],
        ["Yahoo Finance (yfinance)", "Credit Spread (HYG/LQD)", "Daily"]
    ]
    source_table = Table(data_sources, colWidths=[1.8 * inch, 2.5 * inch, 0.8 * inch])
    source_table.setStyle(table_style)
    story.append(source_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Portfolio Construction:</b> Synthetic multi-asset portfolios constructed from ETFs "
        "with equal weights (20% each), providing controlled experimentation with known factor "
        "tilts and full reproducibility.",
        body_style
    ))

    # Figure 1
    add_figure(story, "fig1_event_timeline.png",
               "Figure 1: Portfolio performance and factor concentration event timeline. "
               "Events cluster during crisis periods, with event rates 4.17x higher during "
               "the COVID-19 crisis. The bottom panel shows the 30-day rolling event rate.")
    story.append(PageBreak())

    # ========================================================================
    # PAGE 4: FEATURES & METHODOLOGY
    # ========================================================================
    story.append(Paragraph("3. Feature Engineering", heading1_style))
    story.append(Paragraph(
        "The feature set is designed to capture portfolio factor tilts, concentration dynamics, "
        "and macro-financial stress conditions. All features are computed point-in-time to avoid "
        "look-ahead bias.",
        body_style
    ))

    # Table 2: Features
    feature_data = [
        ["Category", "Specific Features", "Justification"],
        ["Factor Betas", "Rolling 252-day exposures to Mkt-RF, SMB, HML, RMW, CMA", "Captures portfolio factor tilts"],
        ["FCI", "HHI: sum(|beta_k| / sum |beta_j|)^2", "Primary concentration metric"],
        ["FCI Dynamics", "25-day MA, 20-day/30-day changes", "Captures trend and velocity"],
        ["Macro/Market", "VIX level, VIX change, VIX volatility, log VIX", "Market stress proxies"],
        ["Credit Spread", "HYG/LQD ratio, credit_high indicator", "Credit market stress"],
        ["Persistence", "FCI_high, VIX_high, stress_confirm, stress_persistence", "Reduces false positives"],
        ["Volatility", "60-day rolling volatility", "Risk magnitude"]
    ]
    feat_table = Table(feature_data, colWidths=[1.3 * inch, 2.7 * inch, 2.0 * inch])
    feat_table.setStyle(table_style)
    story.append(feat_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Rejected Features (Based on Empirical Testing):</b> Momentum features (5d, 10d, 20d returns) "
        "killed too many true positives. Ratio/correlation features killed predictions. FCI change "
        "features (5d, 10d) were too noisy.",
        body_style
    ))

    # Figure 3
    add_figure(story, "fig3_feature_correlations.png",
               "Figure 3: Feature correlations with the target variable. All features "
               "exhibit correlations below |0.1|, indicating very weak individual "
               "predictive signal. The strongest predictor is log(VIX) at 0.077.",
               width=5.0 * inch)

    story.append(Paragraph("4. Methodology & Evaluation Framework", heading1_style))
    story.append(Paragraph(
        "The methodology adheres to a strict baseline-first, rigorous-evaluation philosophy. "
        "Statistical rigor is non-negotiable. Bootstrap confidence intervals and calibration "
        "testing are mandatory for all models.",
        body_style
    ))

    story.append(Paragraph("4.1 Time-Aware Splitting (No Look-Ahead Bias)", heading2_style))
    split_data = [
        ["Split", "Period", "Purpose", "Samples"],
        ["Training", "2010 - 2016", "Model training, feature engineering", "1,509"],
        ["Validation", "2017 - 2022", "Threshold tuning, hyperparameter selection", "1,510"],
        ["Test", "2023 - 2024", "One-time final evaluation (no leakage)", "500 (27 events)"]
    ]
    split_table = Table(split_data, colWidths=[1.0 * inch, 1.2 * inch, 2.0 * inch, 1.4 * inch])
    split_table.setStyle(table_style)
    story.append(split_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Statistical Rigor (Mandatory):</b> Bootstrap Confidence Intervals (95% CI, 1,000 "
        "iterations), Expected Calibration Error (ECE), and significance criteria (CI excludes "
        "0.5). No test set leakage: all tuning performed on validation set only.",
        body_style
    ))
    story.append(PageBreak())

    # ========================================================================
    # PAGE 5: EMPIRICAL RESULTS
    # ========================================================================
    story.append(Paragraph("5. Empirical Results", heading1_style))
    story.append(Paragraph("5.1 Model Performance (Test Set - One-Time Evaluation)", heading2_style))

    # Table 3: Model Performance
    model_data = [
        ["Model", "AUC-ROC", "95% CI", "Significant?", "Precision", "Recall", "F1"],
        ["Heuristic (FCI 90%)", "0.5198", "N/A", "N/A", "0.0595", "0.4074", "0.1038"],
        ["Enhanced (FCI+VIX)", "0.4948", "N/A", "N/A", "0.0476", "0.0741", "0.0580"],
        ["Logistic Regression", "0.3478", "[0.26, 0.43]", "No", "0.0879", "0.0741", "0.0800"],
        ["Random Forest", "0.2872", "[0.19, 0.39]", "No", "0.0571", "0.1481", "0.0825"],
        ["XGBoost", "0.4798", "[0.38, 0.58]", "No", "0.1025", "1.0000", "0.1862"]
    ]
    model_table = Table(model_data, colWidths=[1.2 * inch, 0.6 * inch, 0.7 * inch, 0.7 * inch, 0.6 * inch, 0.5 * inch, 0.5 * inch])
    model_table.setStyle(table_style)
    story.append(model_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Key Findings:</b> The heuristic rule (FCI > 90%) outperforms all ML models (AUC 0.5198 vs 0.4798). "
        "No model is statistically significant. XGBoost achieves perfect recall (1.000) but at the "
        "cost of low precision (0.1025), meaning 9 out of 10 alerts are false alarms.",
        body_style
    ))

    # Figure 4
    add_figure(story, "fig4_model_comparison.png",
               "Figure 4: Model performance with 95% confidence intervals. All models "
               "have confidence intervals including 0.5, indicating no model is "
               "statistically significant.")

    story.append(Paragraph("5.2 Statistical Significance (XGBoost - Best Model)", heading2_style))
    sig_data = [
        ["Test", "Value", "Interpretation"],
        ["Observed AUC", "0.4798", "Below random (0.5)"],
        ["95% CI Lower", "0.3795", "Below 0.5"],
        ["95% CI Upper", "0.5805", "Above 0.5"],
        ["CI includes 0.5?", "YES", "NOT significant"],
        ["ECE", "0.0318", "Well-calibrated"],
        ["Verdict", "WARNING", "Not significant (CI includes 0.5)"]
    ]
    sig_table = Table(sig_data, colWidths=[1.5 * inch, 1.2 * inch, 2.0 * inch])
    sig_table.setStyle(table_style)
    story.append(sig_table)
    story.append(PageBreak())

    # ========================================================================
    # PAGE 6: DISCUSSION
    # ========================================================================
    story.append(Paragraph("6. Discussion", heading1_style))
    story.append(Paragraph(
        "The primary finding is that no statistically significant predictive relationship was found. "
        "The XGBoost model, despite being well-calibrated (ECE = 0.032), fails on all practical criteria. "
        "The primary bottleneck is the weakness of available features (all correlations < 0.1), which "
        "provides insufficient predictive signal.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Why does this happen?</b> With only 27 test events, the statistical power to detect any "
        "real effect is extremely limited. The observed predictive signal is statistically "
        "indistinguishable from random noise. The features are too weak: even the strongest feature "
        "(log VIX) has a correlation of only 0.077 with the target.",
        body_style
    ))

    # Figure 5
    add_figure(story, "fig5_calibration_curve.png",
               "Figure 5: Calibration curve for the XGBoost model. The model is "
               "well-calibrated but cannot discriminate between positive and negative "
               "classes. Predictions are clustered around 0.09.",
               width=4.5 * inch)

    # Figure 6
    add_figure(story, "fig6_precision_recall.png",
               "Figure 6: Precision-Recall curve. The model achieves perfect recall "
               "at the cost of low precision, illustrating the practical limitation "
               "of the approach.",
               width=4.5 * inch)

    # Figure 2
    add_figure(story, "fig2_regime_comparison.png",
               "Figure 2: Factor concentration event rates by market regime. Events "
               "are 4.2x more frequent during the COVID-19 crisis (34.9% vs 8.4%).",
               width=4.5 * inch)

    story.append(Paragraph(
        "This negative result is a valuable contribution. It demonstrates that with public factors, "
        "synthetic portfolios, and the optimized target definition, factor concentration events "
        "cannot be reliably predicted. Future work would require alternative data sources "
        "(options flow, 13F filings, proprietary positioning data) and better features.",
        body_style
    ))
    story.append(PageBreak())

    # ========================================================================
    # PAGE 7: LIMITATIONS & FUTURE WORK
    # ========================================================================
    story.append(Paragraph("7. Limitations", heading1_style))
    story.append(Paragraph(
        "The following limitations should be considered when interpreting the results:",
        body_style
    ))
    limitations = [
        "<b>Data Constraints:</b> Utilizes price data only; does not incorporate proprietary flow data, 13F institutional ownership, or short-interest metrics.",
        "<b>Feature Strength:</b> All feature correlations < 0.1, indicating very weak signal. Even the strongest feature (log VIX) has a correlation of only 0.077.",
        "<b>Factor Coverage:</b> Limited to standard publicly available factor families (Fama-French 5-factor).",
        "<b>Synthetic Portfolios:</b> May not capture real institutional portfolio complexity.",
        "<b>Test Period:</b> Only 27 events in test period (2023-2024), limiting statistical power.",
        "<b>Power:</b> With only 27 test events, the power to detect any real effect is limited.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"• {lim}", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("8. Hypotheses for Future Work", heading1_style))
    future_data = [
        ["Hypothesis", "Test", "Rationale"],
        ["H1: Proprietary data reveals signal", "Add options data, 13F filings, short interest", "Public factors are insufficient"],
        ["H2: Alternative target definition works", "Predict factor crowding directly", "Current target may be too broad"],
        ["H3: Different feature engineering improves signal", "Nonlinear transformations, interaction terms", "Current features too linear"],
        ["H4: Crisis vs normal separation helps", "Model regimes separately", "Relationship may be non-stationary"],
        ["H5: More data (to 2029) reveals signal", "Extend data timeframe", "More events may reveal pattern"]
    ]
    future_table = Table(future_data, colWidths=[1.8 * inch, 2.2 * inch, 2.0 * inch])
    future_table.setStyle(table_style)
    story.append(future_table)
    story.append(PageBreak())

    # ========================================================================
    # PAGE 8: CONCLUSION & REFERENCES
    # ========================================================================
    story.append(Paragraph("9. Conclusion", heading1_style))
    story.append(Paragraph(
        "This project set out to determine whether factor concentration events can be reliably "
        "predicted using public data and machine learning. After rigorous experimentation with "
        "proper validation methodology, no statistically significant predictive relationship was "
        "found. The null hypothesis cannot be rejected with the available data.",
        body_style
    ))
    story.append(Paragraph(
        "The XGBoost model, despite being well-calibrated (ECE = 0.0318), fails on all practical "
        "criteria: AUC-ROC 0.4798 (below random), 95% CI [0.3795, 0.5805] includes 0.5 (NOT "
        "significant), and precision 0.1025 (9 out of 10 alerts are false alarms). The primary "
        "bottleneck is the weakness of available features (all correlations < 0.1).",
        body_style
    ))
    story.append(Paragraph(
        "<b>This negative result is a valuable contribution:</b> it demonstrates that with public "
        "factors, synthetic portfolios, and the optimized target definition, factor concentration "
        "events cannot be reliably predicted; it establishes a reproducible benchmark for future "
        "research; and it saves others from pursuing weak signals with public data.",
        body_style
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("References", heading1_style))
    references = [
        "Arnott, R., Kalesnik, V., & Wu, L. (2019). The incredible shrinking factor return. Journal of Portfolio Management.",
        "Campbell, J. Y., & Shiller, R. J. (1988). Stock prices, earnings, and expected dividends. Journal of Finance.",
        "Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics.",
        "Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics.",
        "Khandani, A. E., & Lo, A. W. (2007). What happened to the quants in August 2007? Journal of Investment Management.",
    ]
    for i, ref in enumerate(references, 1):
        story.append(Paragraph(f"[{i}] {ref}", body_style))

    # ========================================================================
    # BUILD PDF
    # ========================================================================
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=LETTER,
        rightMargin=1 * inch,
        leftMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )
    doc.build(story)
    print(f"✅ PDF generated successfully: {OUTPUT_PDF}")
    print(f"   Location: {os.path.abspath(OUTPUT_PDF)}")


if __name__ == "__main__":
    build_paper()