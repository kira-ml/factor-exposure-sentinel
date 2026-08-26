# Detecting Hidden Factor Overconcentration Risk in Multi-Asset Portfolios: A Machine Learning Early-Warning Approach

**Author:** Ken Ira Lacson Talingting  
**Affiliation:** Independent Research / Portfolio Project  
**Date:** August 2026  
**Status:** Working Paper — Pre-Registration Phase (Design & Problem Framing)  
**JEL Classification:** G11, G17, C45, C53  

---

## Abstract

Traditional portfolio risk systems exhibit a critical blind spot: they monitor asset-level concentration (single-name limits, sector caps) while systematically underemphasizing hidden factor concentration. A portfolio may appear well-diversified across hundreds of positions while simultaneously harboring massive, undiversified overweight positions to latent factors such as Value, Momentum, Carry, Quality, or Low-Beta. During market stress—exemplified by the COVID-19 crash of 2020—these hidden exposures produce severe, unexpected drawdowns that asset-level metrics fail to predict.

This research project develops a machine-learning-driven early-warning system to detect emerging factor overconcentration in multi-asset portfolios before it translates into catastrophic losses. We frame the problem as a supervised anomaly detection task: given a portfolio's historical returns, current holdings, factor betas, and macro-financial conditions, the model predicts whether a "Factor Concentration Event" (drawdown > -5% over 21 days, with >60% attributable to factor exposures) will occur. 

The methodology adheres to a strict *baseline-first* philosophy. Simple threshold rules and logistic regression are established before introducing tree-based ensemble methods (Random Forest, Gradient Boosting). Complexity is only adopted if it yields statistically significant (p < 0.05) improvements in out-of-sample AUC-ROC and demonstrable economic value (drawdown reduction) on a walk-forward validation set. The project utilizes entirely open-source data (Fama-French factors, AQR factors, ETF returns via `yfinance`, and FRED macro indicators) to ensure full reproducibility.

---

## 1. Research Question & Motivation

### 1.1 Primary Research Question
> *Given a portfolio's current holdings, factor loadings, and prevailing market conditions, can a machine learning model accurately predict whether the portfolio is entering a state of dangerous factor overconcentration that will result in a significant, factor-driven drawdown?*

### 1.2 Secondary Research Questions
1.  Which crowding proxies (pairwise correlation, Herfindahl-Hirschman Index, valuation spreads) carry the strongest predictive signal for *downside risk*, as opposed to *alpha decay*?
2.  Do nonlinear machine learning models offer a meaningful improvement over interpretable linear baselines when predicting regime-specific risk?
3.  How does the predictive signal vary across distinct market regimes (low volatility, high volatility, crisis, and recovery)?

### 1.3 Motivation
The August 2007 quant crisis, the February 2018 volatility shock, and the March 2020 COVID-19 drawdown all featured significant losses driven by factor crowding that was invisible to standard risk dashboards. This project addresses the resulting gap by designing a rigorous, data-driven risk surveillance tool for institutional investors and multi-asset portfolio managers.

---

## 2. Problem Formulation

### 2.1 Target Variable
We define a binary target variable \( Y_{t, h} \) indicating a "Factor Concentration Event":

\[
Y_{t, h} = 
\begin{cases} 
1, & \text{if } \text{Drawdown}_{t, t+h} < -5\% \text{ AND } \text{Factor Attribution}_{t, t+h} > 60\% \\
0, & \text{otherwise}
\end{cases}
\]

where:
- **Prediction Horizon** \( h = 21 \) trading days (1 month).
- **Factor Attribution** is measured using a rolling linear factor model (Fama-French 5-Factor or AQR factor set).

### 2.2 Features Available at Prediction Time
The information set \( X_t \) strictly excludes future data and includes:
- **Factor Betas:** Rolling 252-day exposures to Value, Momentum, Carry, Market, Size, and Quality.
- **Factor Concentration Index (FCI):** \( \sum_k \beta_{k,t}^2 / (\sum_k |\beta_{k,t}|)^2 \).
- **Portfolio Structure:** HHI of weights, effective number of bets, and recent turnover.
- **Macro Regime:** VIX level, credit spreads (BAA-10Y), and Treasury yield curve slope.

### 2.3 Data Sources
To ensure academic reproducibility, only open-source data is utilized:

| Data Source | Variable | Frequency |
| :--- | :--- | :--- |
| Kenneth French Data Library | Fama-French 5-Factor Returns | Monthly / Daily |
| AQR Capital Management | Alternative Factor Returns | Monthly |
| Yahoo Finance (`yfinance`) | ETF Prices (SPY, AGG, GLD, etc.) | Daily |
| FRED (Federal Reserve) | VIX, BAA-10Y Spread, 10Y-2Y Slope | Daily |

*Portfolio Construction:* Since institutional holdings are difficult to acquire, we construct synthetic multi-asset portfolios from ETFs with known, varying factor tilts.

---

## 3. Methodology: Progressive Complexity

This research adheres to the principle of Occam's razor. Additional model complexity is introduced *only when empirical evidence justifies it*.

### 3.1 Baseline Models (Tiers 1 & 2)
- **Threshold-Based Rule:** Flag when FCI exceeds a historical percentile (e.g., 90th percentile).
- **Moving Z-Score (Statistical Process Control):** Flag when FCI exceeds 3 standard deviations from its rolling mean.
- **Logistic Regression:** A linear, interpretable model predicting event probability based on current betas and macro features.

### 3.2 Advanced Machine Learning (Tier 3 - Conditional)
- **Random Forest:** Implemented if linear models fail to capture factor interactions.
- **Gradient Boosting (XGBoost/LightGBM):** Implemented for complex non-linear relationships and higher predictive power.

**Justification Criteria:** Advanced models are only adopted if they demonstrate:
1.  A statistically significant improvement in AUC-ROC (\( \Delta AUC > 0.05 \), McNemar's test, \( p < 0.05 \)).
2.  A measurable economic benefit (e.g., \( >50 \) bps reduction in simulated drawdowns).
3.  Robust performance across out-of-sample walk-forward validation windows.

---

## 4. Experimental Design & Evaluation

### 4.1 Time-Aware Splitting
To prevent look-ahead bias, data is split strictly chronologically:
- **Training:** January 2010 – December 2018 (9 years)
- **Validation:** January 2019 – December 2020 (2 years, includes COVID crisis)
- **Test:** January 2021 – December 2024 (4 years, out-of-sample)

### 4.2 Primary Evaluation Metrics
| Category | Metrics |
| :--- | :--- |
| **ML Performance** | AUC-ROC, AUC-PR, Precision @ Recall = 0.8, F1 Score |
| **Risk Management Utility** | Detection Delay (days ahead of event), False Alarm Rate |
| **Economic Value** | Simulated Sharpe Ratio improvement, Max Drawdown reduction |

---

## 5. Repository Structure (Pre-Implementation)

```
factor-exposure-sentinel/
├── README.md                    # This document - Project Overview
├── problem_framing.md           # Full academic problem framing (research design)
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── data/                        # (gitignored) Raw & processed data
│   └── README.md                # Data fetching instructions
│
├── notebooks/                   # EDA & prototyping
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/                         # Core Python modules
│   ├── data_loader.py           # Fetches from yfinance, FF, FRED
│   ├── features.py              # Rolling betas, FCI, macro transforms
│   ├── models.py                # Baselines & ML implementations
│   └── backtest.py              # Position sizing simulation
│
├── tests/                       # Unit tests (specifically for time leakage)
│   └── test_time_series_split.py
│
└── docs/                        # Final paper & visualizations (generated)
    └── figures/
```

---

## 6. Project Roadmap

- [x] **Phase 0: Problem Framing** — Formal definition of target variable, features, and baseline methodology.
- [ ] **Phase 1: Data Pipeline** — Implement leakage-free data fetching and preprocessing.
- [ ] **Phase 2: Feature Engineering** — Construct rolling factor betas and regime indicators.
- [ ] **Phase 3: Baseline Models** — Threshold rules and logistic regression.
- [ ] **Phase 4: Advanced ML** — Conditional implementation of tree-based methods.
- [ ] **Phase 5: Economic Backtest** — Simulate continuous hedging strategies.
- [ ] **Phase 6: Publication** — Generate final academic-style report and open-source benchmark.

---

## 7. Anticipated Contributions

Regardless of whether complex ML outperforms simple baselines, this project aims to contribute:

1.  **A Reproducible Benchmark:** A rigorous, time-aware pipeline for factor risk monitoring using public data.
2.  **Prevention of Look-Ahead Bias:** Practical implementation strategies to avoid data leakage in financial ML.
3.  **Empirical Evidence:** Honest documentation of when simple statistical rules are sufficient, and when complexity adds genuine economic value.

---

## 8. Limitations

- **Data Constraints:** Utilizes price data only; does not incorporate proprietary flow data, 13F institutional ownership, or short-interest metrics.
- **Factor Coverage:** Limited to standard publicly available factor families.
- **Simplified Trading:** Backtests assume frictionless trading and do not model transaction costs or market impact.

---

## 9. License & Disclaimer

**License:** MIT  
**Disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not investment advice and does not claim to generate alpha or predict market movements. Past performance is not indicative of future results.

---

## 10. Author Information

**Ken Ira Lacson Talingting**  
- GitHub: [github.com/kira-ml](https://github.com/kira-ml)  
- LinkedIn: [linkedin.com/in/ken-ira-lacson-852026343](https://www.linkedin.com/in/ken-ira-lacson-852026343/)

---

*Last Updated: August 2026*
