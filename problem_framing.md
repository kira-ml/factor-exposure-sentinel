# Factor Exposure Sentinel: Problem Framing and Research Methodology

**Project Repository:** `factor-exposure-sentinel`  
**Author:** Ken Ira Lacson Talingting  
**Date:** September 2026  
**Status:** Research Complete — Empirical Findings Documented  
**JEL Classification:** G11, G17, C45, C53  
**License:** MIT

---

## 1. Executive Summary

Traditional portfolio risk systems monitor asset-level concentration—single-name limits, sector caps, and geographic ceilings—while systematically underemphasizing a more insidious threat: **hidden factor concentration**. A portfolio may appear well-diversified across hundreds of positions while simultaneously harboring massive, unobserved overweight positions to latent factors such as Value, Momentum, Carry, Quality, or Low-Beta.

During market stress or sudden factor reversals, these hidden exposures can produce severe, unexpected drawdowns that asset-level metrics fail to explain. This project develops a rigorous, reproducible framework to detect emerging factor overconcentration in multi-asset portfolios before it translates into catastrophic losses.

**Empirical Finding:** After rigorous experimentation with proper validation methodology, **no statistically significant predictive relationship was found**. The null hypothesis cannot be rejected with the available data. This negative result is a valuable contribution, establishing a reproducible benchmark and saving others from pursuing weak signals.

---

## 2. The Real-World Problem / Pain Point

### 2.1 The Gap in Current Risk Systems

Risk management infrastructure typically focuses on:
- **Name concentration** (e.g., max 5% in a single stock)
- **Sector/Industry concentration** (e.g., max 20% in Technology)

These systems treat factor exposures as secondary outputs rather than primary risk drivers. A portfolio manager constructing a "diversified" portfolio of 200 stocks may inadvertently create a portfolio where 80% of variance is driven by a single unobserved factor.

### 2.2 Historical Evidence

The 2020 COVID-19 crash provides a textbook example. Multi-asset portfolios experienced severe losses—far beyond what asset-level sector metrics predicted. Post-facto analysis attributed the bulk of these losses to extreme crowding in **Carry** and **Momentum** factors, exposures invisible to standard risk dashboards. Risk managers were left scrambling to explain losses after they had already occurred, rather than mitigating them proactively.

### 2.3 Who Experiences This

- **Multi-Asset Portfolio Managers**: Facing drawdowns they cannot explain to their CIOs
- **Risk Managers**: Tasked with monitoring risk but lacking tools to aggregate factor risk across silos
- **Institutional Investors**: Unable to discern whether underlying managers are taking correlated factor bets
- **Quantitative Researchers**: Seeking robust methods to monitor factor crowding and inform dynamic risk budgeting

---

## 3. Why This Problem Matters

| Dimension | Impact |
| :--- | :--- |
| **Economic Cost** | Factor crowding contributed to the quant crisis (Aug 2007), Volmageddon (Feb 2018), and the COVID drawdown (Mar 2020), causing portfolio losses of 15–30% within weeks. |
| **Regulatory/Fiduciary** | Institutional investors increasingly demand transparency into systematic risk exposures. Undetected factor crowding creates significant reputational and fiduciary risk. |
| **Decision Utility** | Better detection enables informed hedging, dynamic risk budgeting, capital reallocation, and improved narrative explanation during stress events. |

---

## 4. Optimized Machine Learning Problem Formulation

### 4.1 Research Philosophy

This project adheres to a **baseline-first, rigorous-evaluation** philosophy:

1. **Problem framing over model complexity**
2. **Baseline-first development** (threshold rules → logistic regression → ensembles)
3. **Simple, defensible methods before advanced methods**
4. **Evidence and evaluation over impressive-sounding claims**
5. **Practical research value over artificial novelty**
6. **Open-source reproducibility over unnecessary sophistication**
7. **Statistical rigor is non-negotiable** (bootstrap CI, calibration, significance testing)

### 4.2 Primary Research Question

> *Given a portfolio's current holdings, factor loadings, and prevailing market conditions, can a machine learning model accurately predict whether the portfolio is entering a state of dangerous factor overconcentration that will result in a significant, factor-driven drawdown?*

### 4.3 Secondary Research Questions

1. Which crowding proxies (FCI, pairwise correlation, persistence) carry the strongest predictive signal for downside risk?
2. Do nonlinear machine learning models offer a meaningful improvement over interpretable linear baselines?
3. How does the predictive signal vary across distinct market regimes?

### 4.4 Target Variable (Updated Based on Empirical Evidence)

A binary target \( Y_{t, h} \) indicates a "Factor Concentration Event":

\[
Y_{t, h} = 
\begin{cases} 
1, & \text{if Drawdown}_{t, t+h} < -3\% \\
0, & \text{otherwise}
\end{cases}
\]

**Key Update:** Based on empirical testing, the attribution threshold (originally >60%) was **removed** because it destroyed the predictive signal. The optimal drawdown threshold of **-3%** was validated through grid search (AUC 0.7528, p < 0.001 on validation set).

- **Prediction Horizon:** \( h = 21 \) trading days (1 month)
- **Portfolio Construction:** Synthetic multi-asset portfolio (equal-weighted ETFs)
- **Data Period:** January 2010 – December 2024 (3,753 trading days)
- **Total Events:** 338 (9.01% event rate)

### 4.5 Features Available at Prediction Time

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

---

## 5. Data Sources & Portfolio Construction

### 5.1 Data Sources

| Data Source | Variable | Frequency |
| :--- | :--- | :--- |
| Kenneth French Data Library | Fama-French 5-Factor Returns | Daily |
| Yahoo Finance (`yfinance`) | ETF Prices (SPY, AGG, GLD, IJS, EFA) | Daily |
| Yahoo Finance (`yfinance`) | VIX (^VIX) | Daily |
| Yahoo Finance (`yfinance`) | Credit Spread (HYG/LQD) | Daily |
| FRED (via `pandas_datareader`) | Yield Curve (T10Y2Y), Baa-10yr Spread (BAA10YM) | Daily |

### 5.2 Portfolio Construction

Synthetic multi-asset portfolios constructed from ETFs with equal weights:

| ETF | Asset Class | Ticker |
|-----|-------------|--------|
| SPY | US Equities | SPY |
| AGG | US Bonds | AGG |
| GLD | Gold | GLD |
| IJS | Small Cap Value | IJS |
| EFA | Developed ex-US | EFA |

**Rationale:** Equal-weighted, multi-asset portfolio provides controlled experimentation with known factor tilts and full reproducibility.

---

## 6. Key Constraints and Assumptions

### 6.1 Constraints
- **Public data only**: No proprietary flow data or expensive subscriptions
- **Reproducibility**: All data accessible via open-source libraries
- **Computational**: Moderate hardware (standard university/individual researcher)
- **Interpretability**: Risk managers need to understand *why* a flag is raised

### 6.2 Core Assumptions
- Fama-French 5-factor model provides a reasonable first-order approximation of portfolio exposures
- The relationship between factor concentration and downside risk evolves slowly enough to be learned from historical data
- Linear factor attribution is a valid baseline for identifying the source of returns

### 6.3 Data Limitations (Empirically Validated)

| Limitation | Implication |
|------------|-------------|
| **Event count: 338 total (9.01% rate)** | Sufficient for modeling, but signal remains weak |
| **Feature correlations < 0.1** | Very weak individual predictive signal |
| **Public factors only** | May miss proprietary crowding signals |
| **Synthetic portfolios** | May not capture real institutional portfolio complexity |
| **Limited test period: 2023-2024** | Only 27 test events |
| **Statistical power** | Limited ability to detect small-to-moderate effects |

---

## 7. Evaluation Framework

### 7.1 Statistical Rigor (Mandatory)

- **Time-Aware Splitting**: Strict chronological order maintained
- **Three-Way Split**: 
  - Train: 2010-2016 (1,497 samples)
  - Validation: 2017-2022 (1,499 samples)
  - Test: 2023-2024 (477 samples, 27 events)
- **Bootstrap Confidence Intervals**: 95% CI for AUC-ROC (1,000 iterations)
- **Calibration Testing**: Expected Calibration Error (ECE)
- **Significance Testing**: CI includes 0.5 → not significant
- **No Test Set Leakage**: All tuning performed on validation set only

### 7.2 Primary ML Metrics

| Metric | Priority | Rationale |
|--------|----------|-----------|
| **AUC-ROC** | Primary | Class imbalance robustness |
| **95% Bootstrap CI** | Primary | Statistical significance assessment |
| **AUC-PR** | Secondary | More informative for rare events |
| **F1 Score** | Secondary | Balance precision and recall |
| **Precision** | Critical | Economic cost of false alarms |
| **ECE** | Secondary | Probability calibration assessment |

### 7.3 Practical Utility Criteria

For the model to be practically useful:
- **AUC-ROC > 0.70** (meaningful separation)
- **95% CI excludes 0.5** (statistically significant)
- **Precision > 0.30** (at least 1 in 3 alerts correct)
- **ECE < 0.10** (well-calibrated)

---

## 8. Solution Strategy: Progressive Complexity

### 8.1 Baseline Solutions (Tiers 1 & 2)

| Model | Description | Why It's Included |
|-------|-------------|-------------------|
| **Heuristic Rule** | FCI > 90th percentile | Simple, interpretable, domain-informed baseline |
| **Enhanced Rule** | FCI > 90% AND VIX > 20 AND Credit > median | Combines multiple stress signals |
| **Logistic Regression** | Linear model with SMOTE | Provides interpretable ML baseline |

### 8.2 Advanced ML Solutions (Tier 3 - Conditional)

| Model | Description | Adoption Criteria |
|-------|-------------|-------------------|
| **Random Forest** | Tree-based ensemble with 100 estimators | Only if AUC > 0.70 on validation |
| **XGBoost** | Gradient boosting with Platt scaling | Primary model due to stability |
| **XGBoost with Calibration** | CalibratedClassifierCV (sigmoid, cv=3) | Reduces overfitting risk |

**Justification for Complexity:** Complexity is only adopted if it yields:
- Statistically significant improvement (95% CI excludes 0.5)
- Practical improvement (AUC > 0.70, precision > 0.30)
- Measurable economic benefit (precision > 0.30)

---

## 9. Empirical Results

### 9.1 Target Analysis (Updated)

| Metric | Value |
|--------|-------|
| Total days | 3,753 |
| Total events | 338 |
| Event rate | 9.01% |
| Crisis event rate (COVID) | 34.9% |
| Normal event rate | 8.4% |
| Crisis/Normal ratio | 4.15x |
| Total clusters | 33 |
| Largest cluster | 29 events (Jan-Mar 2020) |

**Key Finding:** Events cluster heavily during crises, validating the target's ability to capture stress periods. The -3% threshold captures significantly more events than the original -5% threshold.

### 9.2 Feature Analysis (Updated)

| Feature | Correlation with Target |
|---------|------------------------|
| log_vix | 0.0766 |
| vix_level | 0.0630 |
| fci_change_30 | 0.0584 |
| fci_change_20 | 0.0538 |
| beta_CMA | 0.0428 |

**Key Finding:** All feature correlations are < 0.1, indicating very weak individual predictive signal. This explains why ML models struggle to outperform simple rules.

### 9.3 Model Performance (Test Set — One-Time Evaluation)

| Model | AUC-ROC | 95% CI | Significant? | Precision | Recall | F1 |
|-------|---------|--------|--------------|-----------|--------|-----|
| Heuristic (FCI 90%) | 0.5115 | N/A | N/A | 0.0598 | 0.4074 | 0.1043 |
| Enhanced (FCI+VIX) | 0.4926 | N/A | N/A | 0.0476 | 0.0741 | 0.0580 |
| Logistic Regression | 0.3013 | [0.2077, 0.4057] | ❌ | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.4226 | [0.2887, 0.5460] | ❌ | 0.2143 | 0.0556 | 0.0950 |
| **XGBoost** | **0.4756** | **[0.3906, 0.5621]** | **❌** | **0.0256** | **0.0370** | **0.0303** |

### 9.4 Statistical Significance (XGBoost — Best Model)

| Test | Value | Interpretation |
|------|-------|----------------|
| Observed AUC | 0.4756 | Below random (0.5) |
| 95% CI Lower | 0.3906 | Below 0.5 |
| 95% CI Upper | 0.5621 | Above 0.5 |
| **CI includes 0.5?** | **YES** | **NOT significant** |
| ECE | 0.0199 | Well-calibrated |
| Verdict | WARNING | Not significant (CI includes 0.5) |

### 9.5 Key Findings

1. **Heuristic rule outperforms ML models**: FCI > 90% (AUC 0.5115) beats XGBoost (AUC 0.4756)
2. **No model is statistically significant**: All 95% CIs include 0.5
3. **XGBoost achieves low precision and recall**: Precision = 0.0256, Recall = 0.0370
4. **Model is well-calibrated but useless**: ECE = 0.0199, but no discriminative power
5. **Null hypothesis cannot be rejected**: The data does not support reliable prediction
6. **Features are too weak**: All correlations < 0.1
7. **Macroeconomic data does not add signal**: FRED yield curve and credit spread features showed no correlation with target

---

## 10. Success Criteria & Actual Outcomes

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Outperform threshold baseline | AUC > 0.70 | AUC 0.4756 | ❌ |
| Statistical significance | CI excludes 0.5 | CI [0.3906, 0.5621] | ❌ |
| Detect > 60% of events | Recall > 0.6 | Recall 0.0370 | ❌ |
| False positive rate < 30% | FPR < 0.3 | FPR 0.0844 | ✅ |
| Precision > 0.30 | Precision > 0.30 | Precision 0.0256 | ❌ |
| Calibration | ECE < 0.10 | ECE 0.0199 | ✅ |

**Overall Status:** ❌ **Not successful for practical use.** Model does not meet criteria for reliable early warning system.

---

## 11. What the Data Tells Us

### 11.1 The Null Hypothesis Cannot Be Rejected

After rigorous testing with proper validation methodology:
- The observed predictive signal is statistically indistinguishable from random noise
- With only 27 test events, power to detect any real effect is limited
- The data does not support the existence of a reliable predictive relationship

### 11.2 What Would Be Needed for a Practical System

| Requirement | Current | Needed |
|-------------|---------|--------|
| Feature correlation | < 0.1 | > 0.2 |
| Precision | 0.0256 | > 0.30 |
| Test events | 27 | 100+ |
| Data timeframe | 2010-2024 | Extended to 2029+ |

### 11.3 What We Learned

1. **Target definition matters**: Removing attribution threshold improved signal significantly
2. **Validation methodology is essential**: Without it, results were misleading
3. **Statistical testing is non-negotiable**: Bootstrap CI revealed noise
4. **Negative results are valuable**: Save others from pursuing weak signals
5. **Public data is insufficient**: Proprietary data (flows, positioning) likely required
6. **Macroeconomic data does not add signal**: FRED yield curve and credit spread features showed no correlation with target

---

## 12. Statistical Power Analysis

With only 27 test events, the statistical power to detect a real effect is limited.

| Parameter | Value |
|-----------|-------|
| Test samples | 477 |
| Positive events | 27 |
| Negative events | 450 |
| Observed AUC (XGBoost) | 0.4756 |
| 95% CI | [0.3906, 0.5621] |

**Implication:** The failure to reject the null hypothesis may be due to:
1. Insufficient statistical power (small test set)
2. Weak features (all correlations < 0.1)
3. Both factors combined

A larger test set (100+ events) would be needed to detect small-to-moderate effects (AUC > 0.55). With the current sample size, only very large effects (AUC > 0.68) would be detectable at 80% power.

---

## 13. Hypotheses for Future Work

If this research were to continue, the following hypotheses should be tested:

| Hypothesis | Test | Rationale |
|------------|------|-----------|
| H1: Proprietary data reveals signal | Add options data, 13F filings, short interest | Public factors are insufficient |
| H2: Alternative target definition works | Predict factor crowding directly | Current target may be too broad |
| H3: Different feature engineering improves signal | Nonlinear transformations, interaction terms | Current features too linear |
| H4: Crisis vs normal separation helps | Model regimes separately | Relationship may be non-stationary |
| H5: More data (to 2029) reveals signal | Extend data timeframe | More events may reveal pattern |

---

## 14. Open-Source Contribution Value

This project contributes a **reproducible, rigorous benchmark** for the quant finance community:

1. ✅ Clean, object-oriented Python implementation for monitoring factor exposure
2. ✅ Comprehensive handling of look-ahead bias in financial ML
3. ✅ Honest assessment of where simple rules fail and ML does not add value
4. ✅ Bootstrap confidence intervals and calibration testing for statistical rigor
5. ✅ Negative results documented openly
6. ✅ Publication-quality visualizations (6 figures, PDF+PNG)
7. ✅ Complete experiment tracking with `outputs/all_runs.csv`

### 14.1 Repository Structure

```
factor-exposure-sentinel/
├── README.md                    # Project overview and results
├── problem_framing.md           # This document
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

## 15. References & Related Work

- Arnott, R., Kalesnik, V., & Wu, L. (2019). The incredible shrinking factor return. *Journal of Portfolio Management*.
- Khandani, A. E., & Lo, A. W. (2007). What happened to the quants in August 2007? *Journal of Investment Management*.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*.
- Campbell, J. Y., & Shiller, R. J. (1988). Stock prices, earnings, and expected dividends. *Journal of Finance*.

---

## 16. Conclusion

This project set out to determine whether factor concentration events can be reliably predicted using public data and machine learning. After rigorous experimentation with proper validation methodology:

**No statistically significant predictive relationship was found.**

The XGBoost model, despite being well-calibrated (ECE = 0.0199), fails on all practical criteria:
- AUC-ROC: 0.4756 (below random)
- 95% CI: [0.3906, 0.5621] includes 0.5 → NOT significant
- Precision: 0.0256 (39 out of 40 alerts are false alarms)
- F1: 0.0303 (poor precision-recall balance)

The primary bottleneck is the weakness of available features (all correlations < 0.1), which provides insufficient predictive signal. Even with 338 events over 15 years, the model cannot distinguish signal from noise.

**This negative result is a valuable contribution:**
- It demonstrates that with public factors, synthetic portfolios, and the optimized target definition, factor concentration events cannot be reliably predicted
- It establishes a reproducible benchmark for future research
- It saves others from pursuing weak signals with public data

Future work would require alternative data sources (options flow, 13F filings, proprietary positioning data), better features (nonlinear transformations, interaction terms), or a revised target definition.

---

## 17. Results Source

**Definitive results:** See `outputs/all_runs.csv` and `outputs/run_20260909_012949/metrics.json` for the complete set of metrics.

**Key values used in this document:**

| Metric | Value |
|--------|-------|
| XGBoost AUC-ROC | 0.4756 |
| XGBoost 95% CI | [0.3906, 0.5621] |
| XGBoost Precision | 0.0256 |
| XGBoost Recall | 0.0370 |
| XGBoost F1 | 0.0303 |
| XGBoost ECE | 0.0199 |
| XGBoost Threshold | 0.08 |
| XGBoost FPR | 0.0844 |
| Heuristic AUC-ROC | 0.5115 |
| Random Forest AUC-ROC | 0.4226 |
| Logistic Regression AUC-ROC | 0.3013 |

---

## 18. License & Disclaimer

**License:** MIT

**Disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not investment advice and does not claim to generate alpha or predict market movements. Past performance is not indicative of future results.

---

## 19. Author Information

**Ken Ira Lacson Talingting**
- GitHub: [github.com/kira-ml](https://github.com/kira-ml)
- LinkedIn: [linkedin.com/in/ken-ira-lacson-852026343](https://www.linkedin.com/in/ken-ira-lacson-852026343/)

---

*Last Updated: September 2026*
