# Detecting Hidden Factor Overconcentration Risk in Multi-Asset Portfolios: A Machine Learning Early-Warning Approach

**Author:** Ken Ira Lacson Talingting  
**Affiliation:** Independent Research / Portfolio Project  
**Date:** September 2026  
**Status:** Research Complete — Empirical Findings Documented  
**JEL Classification:** G11, G17, C45, C53  
**License:** MIT

---

## Abstract

Traditional portfolio risk systems monitor asset-level concentration (single-name limits, sector caps) while systematically underemphasizing hidden factor concentration. A portfolio may appear well-diversified across hundreds of positions while simultaneously harboring massive, undiversified overweight positions to latent factors such as Value, Momentum, Carry, Quality, or Low-Beta. During market stress—exemplified by the COVID-19 crash of 2020—these hidden exposures produce severe, unexpected drawdowns that asset-level metrics fail to predict.

This project develops a rigorous, reproducible machine learning framework to detect emerging factor overconcentration in multi-asset portfolios before it translates into catastrophic losses. The problem is framed as a supervised binary classification task: given a portfolio's historical returns, current holdings, factor betas, and macro-financial conditions, the model predicts whether a "Factor Concentration Event" (drawdown < -3% over 21 days) will occur.

The methodology adheres to a strict **baseline-first, rigorous-evaluation** philosophy. Simple threshold rules and logistic regression are established before introducing tree-based ensemble methods (Random Forest, XGBoost). Complexity is only adopted if it yields statistically significant (95% CI excludes 0.5) improvements in out-of-sample AUC-ROC. The project utilizes entirely open-source data (Fama-French factors, ETF returns via `yfinance`) to ensure full reproducibility.

**Empirical Finding:** After rigorous experimentation with proper validation methodology, **no statistically significant predictive relationship was found**. The null hypothesis cannot be rejected with the available data. This negative result is a valuable contribution, establishing a reproducible benchmark and saving others from pursuing weak signals with public data.

---

## 1. Research Question & Motivation

### 1.1 Primary Research Question

> *Given a portfolio's current holdings, factor loadings, and prevailing market conditions, can a machine learning model accurately predict whether the portfolio is entering a state of dangerous factor overconcentration that will result in a significant drawdown?*

### 1.2 Secondary Research Questions

1. Which crowding proxies (FCI, VIX, credit spreads) carry the strongest predictive signal for downside risk?
2. Do nonlinear machine learning models offer a meaningful improvement over interpretable linear baselines?
3. How does the predictive signal vary across distinct market regimes?

### 1.3 Motivation

The August 2007 quant crisis, the February 2018 volatility shock, and the March 2020 COVID-19 drawdown all featured significant losses driven by factor crowding invisible to standard risk dashboards. This project addresses the resulting gap by designing a rigorous, data-driven risk surveillance framework for institutional investors and multi-asset portfolio managers.

---

## 2. Optimized Problem Formulation

### 2.1 Target Variable (Updated Based on Empirical Evidence)

A binary target \( Y_{t, h} \) indicates a "Factor Concentration Event":

\[
Y_{t, h} = 
\begin{cases} 
1, & \text{if } \text{Drawdown}_{t, t+h} < -3\% \\
0, & \text{otherwise}
\end{cases}
\]

**Key Update:** Based on empirical testing, the attribution threshold (originally >60%) was **removed** because it destroyed the predictive signal. The optimal drawdown threshold of **-3%** was validated through grid search (validation AUC 0.7528, p < 0.001).

- **Prediction Horizon:** \( h = 21 \) trading days (1 month)
- **Data Period:** January 2010 – December 2024 (3,773 trading days)
- **Total Events:** 338 (8.96% event rate)

### 2.2 Features Available at Prediction Time

| Feature Category | Specific Features | Justification |
|------------------|-------------------|---------------|
| **Factor Betas** | Rolling 252-day exposures to Mkt-RF, SMB, HML, RMW, CMA | Captures portfolio factor tilts |
| **Factor Concentration Index (FCI)** | HHI: \( \sum (|\beta_k| / \sum |\beta_j|)^2 \) | Primary concentration metric |
| **FCI Dynamics** | 25-day MA, 20-day/30-day changes | Captures trend and velocity |
| **Macro/Market** | VIX level, VIX change, VIX volatility, log VIX | Market stress proxies |
| **Credit Spread** | HYG/LQD ratio, credit_high indicator | Credit market stress |
| **Persistence Features** | FCI_high, VIX_high, stress_confirm, stress_persistence | Reduces false positives |
| **Portfolio Volatility** | 60-day rolling volatility | Risk magnitude |

**Rejected Features (Based on Empirical Testing):**
- Momentum features (5d, 10d, 20d returns) — killed too many true positives
- Ratio/correlation features — killed predictions
- FCI change features (5d, 10d) — too noisy

### 2.3 Data Sources

| Data Source | Variable | Frequency |
| :--- | :--- | :--- |
| Kenneth French Data Library | Fama-French 5-Factor Returns | Daily |
| Yahoo Finance (`yfinance`) | ETF Prices (SPY, AGG, GLD, IJS, EFA) | Daily |
| Yahoo Finance (`yfinance`) | VIX (^VIX) | Daily |
| Yahoo Finance (`yfinance`) | Credit Spread (HYG/LQD) | Daily |

**Portfolio Construction:** Synthetic multi-asset portfolios constructed from ETFs with equal weights, providing controlled experimentation with known factor tilts and full reproducibility.

---

## 3. Methodology: Progressive Complexity with Rigor

### 3.1 Research Philosophy

> **Baseline first. Add complexity only when the data and empirical evidence justify it. Statistical rigor is non-negotiable.**

Every significant modeling decision has a clear rationale. Complexity is introduced only when there is evidence that it addresses a demonstrated limitation of the current approach. **Bootstrap confidence intervals and calibration testing are mandatory for all models.**

### 3.2 Baseline Models (Tiers 1 & 2)

| Model | Description |
|-------|-------------|
| **Threshold-Based Rule** | Flag when FCI exceeds historical percentile (90th) |
| **Enhanced Threshold** | FCI > 90th percentile AND VIX > 20 AND Credit > median |
| **Logistic Regression** | Linear model with SMOTE for class imbalance |

### 3.3 Advanced Machine Learning (Tier 3 - Conditional)

| Model | Description |
|-------|-------------|
| **Random Forest** | Tree-based ensemble with 100 estimators |
| **XGBoost** | Gradient boosting with Platt scaling calibration (primary model) |

**Justification Criteria:** Advanced models are only adopted if they demonstrate:
1. Statistically significant improvement (95% CI excludes 0.5)
2. Practical improvement (AUC > 0.70, precision > 0.30)
3. Measurable economic benefit (precision > 0.30)

---

## 4. Experimental Design & Evaluation

### 4.1 Time-Aware Splitting (No Look-Ahead Bias)

| Split | Period | Purpose | Samples |
|-------|--------|---------|---------|
| **Training** | January 2010 – December 2016 | Model training, feature engineering | 1,509 |
| **Validation** | January 2017 – December 2022 | Threshold tuning, hyperparameter selection | 1,510 |
| **Test** | January 2023 – December 2024 | **ONE-TIME** final evaluation | 500 (27 events) |

### 4.2 Primary Evaluation Metrics

| Category | Metrics | Rationale |
| :--- | :--- | :--- |
| **ML Performance** | AUC-ROC, AUC-PR, F1, Precision, Recall | Class imbalance robustness |
| **Statistical Rigor** | **95% Bootstrap CI**, ECE | Significance + calibration assessment |
| **Practical Utility** | Precision, False Positive Ratio | Economic cost of false alarms |

### 4.3 Statistical Rigor (Mandatory)

- **Bootstrap Confidence Intervals:** 95% CI for AUC-ROC (1,000 iterations)
- **Calibration Testing:** Expected Calibration Error (ECE)
- **Significance Criterion:** CI excludes 0.5 → statistically significant
- **No Test Set Leakage:** All tuning performed on validation set only

---

## 5. Empirical Results

### 5.1 Target Analysis

| Metric | Value |
|--------|-------|
| Total days | 3,773 |
| Total events | 338 |
| Event rate | 8.96% |
| Crisis event rate (COVID-19) | 34.9% |
| Normal event rate | 8.4% |
| Crisis/Normal ratio | **4.17x** |
| Total clusters | 33 |
| Largest cluster | 29 events (Jan-Mar 2020) |

**Key Finding:** Events cluster heavily during crises, validating the target's ability to capture stress periods. The -3% threshold captures significantly more events than the original -5% threshold.

### 5.2 Feature Analysis

| Feature | Correlation with Target |
|---------|------------------------|
| log_vix | 0.0771 |
| vix_level | 0.0636 |
| fci_change_30 | 0.0585 |
| fci_change_20 | 0.0539 |
| beta_CMA | 0.0418 |

**Key Finding:** All feature correlations are < 0.1, indicating very weak individual predictive signal. This explains why ML models struggle to outperform simple rules.

### 5.3 Model Performance (Test Set — One-Time Evaluation)

| Model | AUC-ROC | 95% CI | Significant? | Precision | Recall | F1 |
|-------|---------|--------|--------------|-----------|--------|-----|
| Heuristic (FCI 90%) | 0.5198 | N/A | N/A | 0.0595 | 0.4074 | 0.1038 |
| Enhanced (FCI+VIX) | 0.4948 | N/A | N/A | 0.0476 | 0.0741 | 0.0580 |
| Logistic Regression | 0.3478 | [0.2614, 0.4350] | ❌ | 0.0879 | 0.0741 | 0.0800 |
| Random Forest | 0.2872 | [0.1865, 0.3885] | ❌ | 0.0571 | 0.1481 | 0.0825 |
| **XGBoost** | **0.4798** | **[0.3795, 0.5805]** | **❌** | **0.1025** | **1.0000** | **0.1862** |

### 5.4 Statistical Significance (XGBoost — Best Model)

| Test | Value | Interpretation |
|------|-------|----------------|
| Observed AUC | 0.4798 | Below random (0.5) |
| 95% CI Lower | 0.3795 | Below 0.5 |
| 95% CI Upper | 0.5805 | Above 0.5 |
| **CI includes 0.5?** | **YES** | **NOT significant** |
| ECE | 0.0318 | Well-calibrated |
| Verdict | WARNING | Not significant (CI includes 0.5) |

### 5.5 Key Findings

1. **Heuristic rule outperforms ML models:** FCI > 90% (AUC 0.5198) beats XGBoost (AUC 0.4798)
2. **No model is statistically significant:** All 95% CIs include 0.5
3. **XGBoost achieves perfect recall but low precision:** Recall = 1.000, Precision = 0.1025
4. **Model is well-calibrated but useless:** ECE = 0.0318, but no discriminative power
5. **Null hypothesis cannot be rejected:** The data does not support reliable prediction
6. **Features are too weak:** All correlations < 0.1

---

## 6. Success Criteria & Actual Outcomes

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Outperform threshold baseline | AUC > 0.70 | AUC 0.4798 | ❌ |
| Statistical significance | CI excludes 0.5 | CI [0.3795, 0.5805] | ❌ |
| Detect > 60% of events | Recall > 0.6 | Recall 1.000 | ✅ |
| False positive rate < 30% | FPR < 0.3 | FPR 0.50 | ❌ |
| Precision > 0.30 | Precision > 0.30 | Precision 0.1025 | ❌ |
| Calibration | ECE < 0.10 | ECE 0.0318 | ✅ |

**Overall Status:** ❌ **Not successful for practical use.** Model does not meet criteria for reliable early warning system.

---

## 7. What the Data Tells Us

### 7.1 The Null Hypothesis Cannot Be Rejected

After rigorous testing with proper validation methodology:
- The observed predictive signal is statistically indistinguishable from random noise
- With only 27 test events, power to detect any real effect is limited
- The data does not support the existence of a reliable predictive relationship

### 7.2 What Would Be Needed for a Practical System

| Requirement | Current | Needed |
|-------------|---------|--------|
| Feature correlation | < 0.1 | > 0.2 |
| Precision | 0.1025 | > 0.30 |
| Test events | 27 | 100+ |
| Data timeframe | 2010-2024 | Extended to 2029+ |

### 7.3 What We Learned

1. **Target definition matters:** Removing attribution threshold improved signal significantly
2. **Validation methodology is essential:** Without it, results were misleading
3. **Statistical testing is non-negotiable:** Bootstrap CI revealed noise
4. **Negative results are valuable:** Save others from pursuing weak signals
5. **Public data is insufficient:** Proprietary data (flows, positioning) likely required

---

## 8. Repository Structure

```
factor-exposure-sentinel/
├── README.md                    # This document
├── problem_framing.md           # Full research methodology
├── requirements.txt             # Python dependencies
├── main.py                      # Pipeline orchestrator
│
├── src/                         # Core Python modules
│   ├── data_loader.py           # Data fetching with caching
│   ├── target.py                # Target variable definition
│   ├── features.py              # Feature engineering (validated set)
│   ├── models.py                # Model factory (LR, RF, XGBoost)
│   ├── evaluate.py              # Statistical rigor framework
│   ├── target_analysis.py       # Target validation
│   └── visualization.py         # Publication-quality figures
│
├── data/                        # (gitignored) Cached data
└── outputs/                     # (gitignored) Results tracking
    ├── all_runs.csv             # All experiment results
    ├── run_*/metrics.json       # Per-run metrics
    └── figures/                 # Publication-quality figures
        ├── fig1_event_timeline.{pdf,png}
        ├── fig2_regime_comparison.{pdf,png}
        ├── fig3_feature_correlations.{pdf,png}
        ├── fig4_model_comparison.{pdf,png}
        ├── fig5_calibration_curve.{pdf,png}
        └── fig6_precision_recall.{pdf,png}
```

---

## 9. Project Roadmap

- [x] **Phase 0: Problem Framing** — Formal definition of target, features, methodology
- [x] **Phase 1: Data Pipeline** — Leakage-free data fetching and preprocessing
- [x] **Phase 2: Feature Engineering** — Rolling betas, FCI, macro transforms
- [x] **Phase 3: Baseline Models** — Threshold rules and logistic regression
- [x] **Phase 4: Advanced ML** — Random Forest and XGBoost with proper validation
- [x] **Phase 5: Statistical Testing** — Bootstrap CI and calibration testing
- [x] **Phase 6: Publication** — Final report and open-source benchmark

---

## 10. Project Contributions

1. **A Reproducible Benchmark:** A rigorous, time-aware pipeline for factor risk monitoring using public data
2. **Prevention of Look-Ahead Bias:** Practical implementation strategies to avoid data leakage in financial ML
3. **Empirical Evidence:** Honest documentation that with public factors, synthetic portfolios, and the optimized target definition, factor concentration events cannot be reliably predicted
4. **Statistical Rigor:** Bootstrap confidence intervals and calibration testing for model validation
5. **Negative Results Documented Openly:** A valuable contribution—saving others from pursuing weak signals
6. **Publication-Quality Visualizations:** 6 figures in PDF + PNG format for research papers

---

## 11. Limitations

- **Data Constraints:** Utilizes price data only; does not incorporate proprietary flow data, 13F institutional ownership, or short-interest metrics
- **Feature Strength:** All feature correlations < 0.1, indicating very weak signal
- **Factor Coverage:** Limited to standard publicly available factor families
- **Synthetic Portfolios:** May not capture real institutional portfolio complexity
- **Test Period:** Only 27 events in test period (2023-2024)

---

## 12. Hypotheses for Future Work

If this research were to continue, the following hypotheses should be tested:

| Hypothesis | Test | Rationale |
|------------|------|-----------|
| H1: Proprietary data reveals signal | Add options data, 13F filings, short interest | Public factors are insufficient |
| H2: Alternative target definition works | Predict factor crowding directly | Current target may be too broad |
| H3: Different feature engineering improves signal | Nonlinear transformations, interaction terms | Current features too linear |
| H4: Crisis vs normal separation helps | Model regimes separately | Relationship may be non-stationary |
| H5: More data (to 2029) reveals signal | Extend data timeframe | More events may reveal pattern |

---

## 13. Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kira-ml/factor-exposure-sentinel.git
cd factor-exposure-sentinel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Run full pipeline with caching
python main.py --use-cache

# Or run without cache for fresh data
python main.py
```

### View Results

```bash
# Check results
cat outputs/all_runs.csv

# View visualizations
explorer outputs/figures/  # On Windows
```

---

## 14. License & Disclaimer

**License:** MIT

**Disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not investment advice and does not claim to generate alpha or predict market movements. Past performance is not indicative of future results.

---

## 15. Author Information

**Ken Ira Lacson Talingting**
- GitHub: [github.com/kira-ml](https://github.com/kira-ml)
- LinkedIn: [linkedin.com/in/ken-ira-lacson-852026343](https://www.linkedin.com/in/ken-ira-lacson-852026343/)

---

*Last Updated: September 2026*