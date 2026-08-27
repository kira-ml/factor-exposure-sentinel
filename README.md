# Detecting Hidden Factor Overconcentration Risk in Multi-Asset Portfolios: A Machine Learning Early-Warning Approach

**Author:** Ken Ira Lacson Talingting  
**Affiliation:** Independent Research / Portfolio Project  
**Date:** August 2026  
**Status:** Research Complete — Empirical Findings Documented  
**JEL Classification:** G11, G17, C45, C53  

---

## Abstract

Traditional portfolio risk systems monitor asset-level concentration (single-name limits, sector caps) while systematically underemphasizing hidden factor concentration. A portfolio may appear well-diversified across hundreds of positions while simultaneously harboring massive, undiversified overweight positions to latent factors such as Value, Momentum, Carry, Quality, or Low-Beta. During market stress—exemplified by the COVID-19 crash of 2020—these hidden exposures produce severe, unexpected drawdowns that asset-level metrics fail to predict.

This project develops a machine-learning-driven early-warning system to detect emerging factor overconcentration in multi-asset portfolios before it translates into catastrophic losses. The problem is framed as a supervised anomaly detection task: given a portfolio's historical returns, current holdings, factor betas, and macro-financial conditions, the model predicts whether a "Factor Concentration Event" (drawdown < -5% over 21 days, with >60% attributable to factor exposures) will occur.

The methodology adheres to a strict *baseline-first* philosophy. Simple threshold rules and logistic regression are established before introducing tree-based ensemble methods (Random Forest, XGBoost). Complexity is only adopted if it yields statistically significant (p < 0.05) improvements in out-of-sample AUC-ROC and demonstrable economic value on a walk-forward validation set. The project utilizes entirely open-source data (Fama-French factors, ETF returns via `yfinance`, and FRED macro indicators) to ensure full reproducibility.

**Empirical Finding:** After rigorous experimentation with proper validation methodology, no statistically significant predictive relationship was found. The null hypothesis could not be rejected with the available data.

---

## 1. Research Question & Motivation

### 1.1 Primary Research Question
> *Given a portfolio's current holdings, factor loadings, and prevailing market conditions, can a machine learning model accurately predict whether the portfolio is entering a state of dangerous factor overconcentration that will result in a significant, factor-driven drawdown?*

### 1.2 Secondary Research Questions
1. Which crowding proxies (FCI, pairwise correlation, HHI) carry the strongest predictive signal for downside risk?
2. Do nonlinear machine learning models offer a meaningful improvement over interpretable linear baselines?
3. How does the predictive signal vary across distinct market regimes?

### 1.3 Motivation
The August 2007 quant crisis, the February 2018 volatility shock, and the March 2020 COVID-19 drawdown all featured significant losses driven by factor crowding invisible to standard risk dashboards. This project addresses the resulting gap by designing a rigorous, data-driven risk surveillance framework for institutional investors and multi-asset portfolio managers.

---

## 2. Problem Formulation

### 2.1 Target Variable
A binary target \( Y_{t, h} \) indicates a "Factor Concentration Event":

\[
Y_{t, h} = 
\begin{cases} 
1, & \text{if } \text{Drawdown}_{t, t+h} < -5\% \text{ AND } \text{Factor Attribution}_{t, t+h} > 60\% \\
0, & \text{otherwise}
\end{cases}
\]

- **Prediction Horizon:** \( h = 21 \) trading days (1 month)
- **Factor Attribution:** Measured using a rolling linear factor model (Fama-French 5-Factor)

### 2.2 Features Available at Prediction Time

| Feature Category | Specific Features |
|------------------|-------------------|
| **Factor Betas** | Rolling 252-day exposures to Mkt-RF, SMB, HML, RMW, CMA |
| **Factor Concentration Index (FCI)** | Proper HHI: \( \sum (|\beta_k| / \sum |\beta_j|)^2 \) |
| **Portfolio Structure** | HHI of weights, turnover |
| **Macro/Market** | VIX level, VIX change, VIX volatility, credit spread |
| **Derived Features** | Log transformations, rolling volatility, factor correlations |

### 2.3 Data Sources

| Data Source | Variable | Frequency |
| :--- | :--- | :--- |
| Kenneth French Data Library | Fama-French 5-Factor Returns | Daily |
| Yahoo Finance (`yfinance`) | ETF Prices (SPY, AGG, GLD, IJS, EFA) | Daily |
| Yahoo Finance (`yfinance`) | VIX (^VIX) | Daily |

**Portfolio Construction:** Synthetic multi-asset portfolios constructed from ETFs with equal weights, providing controlled experimentation with known factor tilts and full reproducibility.

---

## 3. Methodology: Progressive Complexity

### 3.1 Research Philosophy

> **Baseline first. Add complexity only when the data and empirical evidence justify it.**

Every significant modeling decision has a clear rationale. Complexity is introduced only when there is evidence that it addresses a demonstrated limitation of the current approach.

### 3.2 Baseline Models (Tiers 1 & 2)

| Model | Description |
|-------|-------------|
| **Threshold-Based Rule** | Flag when FCI exceeds historical percentile (e.g., 90th) |
| **VIX-Enhanced Threshold** | FCI > 90th percentile AND VIX > threshold |
| **FCI Trend-Enhanced** | FCI > 90th percentile AND FCI > rolling average |
| **Combined Rule** | FCI > 90% + MA25 + VIX19 + Vol60 + Credit > median |
| **Logistic Regression** | Linear model with SMOTE for class imbalance |

### 3.3 Advanced Machine Learning (Tier 3 - Conditional)

| Model | Description |
|-------|-------------|
| **Random Forest** | Tree-based ensemble with built-in feature importance |
| **XGBoost** | Gradient boosting with Platt scaling calibration |

**Justification Criteria:** Advanced models are only adopted if they demonstrate:
1. Statistically significant improvement (p < 0.05 in permutation test)
2. Practical improvement (AUC > 0.70, precision > 0.30)
3. Measurable economic benefit

---

## 4. Experimental Design & Evaluation

### 4.1 Time-Aware Splitting

| Split | Period | Purpose |
|-------|--------|---------|
| **Training** | January 2010 – December 2018 | Model training, feature engineering |
| **Validation** | January 2019 – December 2020 | Threshold tuning, hyperparameter selection |
| **Test** | January 2021 – December 2024 | ONE-TIME final evaluation |

### 4.2 Primary Evaluation Metrics

| Category | Metrics |
| :--- | :--- |
| **ML Performance** | AUC-ROC, AUC-PR, F1 Score, Precision, Recall |
| **Statistical Rigor** | Permutation test (p-value), Power analysis |
| **Practical Utility** | False Positive Ratio, Precision threshold |

### 4.3 Statistical Rigor

- **Permutation Testing:** Empirical p-value for model significance
- **Power Analysis:** Assessment of sufficient events for reliable detection
- **No Test Set Leakage:** All tuning performed on validation set only

---

## 5. Empirical Results

### 5.1 Target Analysis

| Metric | Value |
|--------|-------|
| Total days | 3,773 |
| Total events | 96 |
| Event rate | 2.54% |
| Crisis event rate (COVID) | 27.7% |
| Normal event rate | 2.0% |
| Crisis/Normal ratio | 14.0x |

**Key Finding:** Events cluster heavily during crises, validating the target's ability to capture stress periods.

### 5.2 Feature Analysis

| Feature | Correlation with Target |
|---------|------------------------|
| log_vix | 0.0916 |
| fci | 0.0826 |
| vix_level | 0.0775 |
| fci_change_20 | 0.0752 |
| vix_vol | 0.0744 |

**Key Finding:** All feature correlations are < 0.1, indicating very weak individual predictive signal.

### 5.3 Model Performance (Validation Set)

| Model | AUC-ROC | F1 | TP | FP | FN |
|-------|---------|-----|-----|-----|-----|
| Threshold Baseline (90%) | 0.4203 | 0.0000 | 0 | 38 | 23 |
| VIX-Enhanced Threshold | 0.4203 | 0.0000 | 0 | 77 | 23 |
| FCI Trend-Enhanced | 0.4482 | 0.0000 | 0 | 50 | 23 |
| Combined Rule | 0.4482 | 0.0000 | 0 | 50 | 23 |
| Logistic Regression | 0.1592 | 0.0000 | 0 | 44 | 23 |
| **Random Forest** | **0.5465** | **0.1288** | **19** | **253** | **4** |
| XGBoost | N/A | 0.0000 | 0 | N/A | 23 |

### 5.4 Statistical Significance (Random Forest)

| Test | Value | Interpretation |
|------|-------|----------------|
| Observed AUC | 0.5465 | Barely above random |
| Permutation AUC mean | 0.5000 | Random expectation |
| Permutation AUC std | 0.0614 | Natural variation |
| **p-value** | **0.2180** | **NOT significant** |
| Effect size | 0.0465 | Very small |
| Events needed | 200-300 | For reliable detection |

### 5.5 Key Findings

1. **Rule-based approaches failed completely:** All threshold rules detected zero events in validation
2. **Random Forest showed weak signal:** AUC 0.5465 but not statistically significant (p=0.2180)
3. **False positive ratio is unacceptable:** 13:1 (253 FP for 19 TP)
4. **Signal is statistically indistinguishable from noise:** Permutation test confirms
5. **Limited data is the bottleneck:** Only 23 validation events; 200-300 events needed
6. **Features are too weak:** All correlations < 0.1

---

## 6. Success Criteria & Actual Outcomes

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Outperform threshold baseline | AUC > 0.7 | AUC 0.5465 | ❌ |
| Detect > 60% of events | Recall > 0.6 | Recall 0.826 | ✅ |
| False positive rate < 30% | FPR < 0.3 | FPR 0.50 | ❌ |
| Statistical significance | p < 0.05 | p = 0.2180 | ❌ |
| Precision > 0.30 | Precision > 0.30 | Precision 0.0699 | ❌ |

**Overall Status:** ❌ **Not successful.** Model does not meet criteria for practical use.

---

## 7. What the Data Tells Us

### 7.1 The Null Hypothesis Cannot Be Rejected

After rigorous testing with proper validation methodology:
- The observed predictive signal is statistically indistinguishable from random noise
- With only 23 validation events, power to detect any real effect is severely limited
- The data does not support the existence of a reliable predictive relationship

### 7.2 What Would Be Needed

| Requirement | Current | Needed |
|-------------|---------|--------|
| Validation events | 23 | 200-300 |
| Feature correlation | < 0.1 | > 0.2 |
| Event rate | 2.5% | > 5% |
| Data timeframe | 2010-2024 | Extended to 2029+ |

### 7.3 What We Learned

1. **Mathematical corrections matter:** FCI formula and factor attribution were fixed
2. **Validation methodology is essential:** Without it, results were misleading
3. **Statistical testing is non-negotiable:** Permutation testing revealed noise
4. **Negative results are valuable:** Save others from pursuing weak signals

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
│   ├── features.py              # Feature engineering (FCI, betas, etc.)
│   ├── models.py                # Model factory (LR, RF, XGBoost)
│   ├── evaluate.py              # Evaluation metrics
│   └── target_analysis.py       # Target validation
│
├── tests/                       # Experimental scripts
│   ├── test_target_horizons.py
│   ├── test_fci_windows.py
│   ├── test_combined_improvements.py
│   └── ...
│
├── data/                        # (gitignored) Cached data
└── outputs/                     # (gitignored) Results tracking
```

---

## 9. Project Roadmap

- [x] **Phase 0: Problem Framing** — Formal definition of target, features, methodology
- [x] **Phase 1: Data Pipeline** — Leakage-free data fetching and preprocessing
- [x] **Phase 2: Feature Engineering** — Rolling betas, FCI, macro transforms
- [x] **Phase 3: Baseline Models** — Threshold rules and logistic regression
- [x] **Phase 4: Advanced ML** — Random Forest and XGBoost with proper validation
- [x] **Phase 5: Statistical Testing** — Permutation tests and power analysis
- [ ] **Phase 6: Publication** — Final report and open-source benchmark

---

## 10. Project Contributions

1. **A Reproducible Benchmark:** A rigorous, time-aware pipeline for factor risk monitoring using public data
2. **Prevention of Look-Ahead Bias:** Practical implementation strategies to avoid data leakage in financial ML
3. **Empirical Evidence:** Honest documentation that with public factors, synthetic portfolios, and the current target definition, factor concentration events cannot be reliably predicted
4. **Statistical Rigor:** Permutation testing and power analysis for model validation
5. **Negative Results Documented Openly:** A valuable contribution—saving others from pursuing weak signals

---

## 11. Limitations

- **Data Constraints:** Utilizes price data only; does not incorporate proprietary flow data, 13F institutional ownership, or short-interest metrics
- **Event Count:** Only 96 total events, insufficient for reliable ML (200-300 events needed)
- **Factor Coverage:** Limited to standard publicly available factor families
- **Synthetic Portfolios:** May not capture real institutional portfolio complexity
- **Feature Strength:** All feature correlations < 0.1, indicating very weak signal

---

## 12. License & Disclaimer

**License:** MIT

**Disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not investment advice and does not claim to generate alpha or predict market movements. Past performance is not indicative of future results.

---

## 13. Author Information

**Ken Ira Lacson Talingting**
- GitHub: [github.com/kira-ml](https://github.com/kira-ml)
- LinkedIn: [linkedin.com/in/ken-ira-lacson-852026343](https://www.linkedin.com/in/ken-ira-lacson-852026343/)

---

*Last Updated: August 2026*