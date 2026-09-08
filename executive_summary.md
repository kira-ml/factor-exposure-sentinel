# Factor Exposure Sentinel
## Executive Summary

**Author:** Ken Ira Lacson Talingting  
**Date:** September 2026  
**Status:** Research Complete — Empirical Findings Documented  
**JEL Classification:** G11, G17, C45, C53  
**License:** MIT

---

## 1. The Problem

### 1.1 What Risk Systems Miss

Traditional portfolio risk systems monitor **asset-level concentration**:
- Single-name limits (e.g., max 5% in one stock)
- Sector/industry caps (e.g., max 20% in Technology)
- Geographic ceilings

These systems systematically underemphasize a more insidious threat: **hidden factor concentration**.

A portfolio may appear well-diversified across 200 positions while simultaneously harboring massive, undiversified overweight positions to latent factors such as:

| Factor | Description | Historical Stress Event |
|--------|-------------|------------------------|
| **Momentum** | Trend-following exposure | February 2018 (Volmageddon) |
| **Value** | Cheap vs expensive stocks | August 2007 (Quant Crisis) |
| **Carry** | Yield-seeking strategies | March 2020 (COVID-19) |
| **Low-Beta** | Defensive positioning | 2008 Financial Crisis |
| **Quality** | Profitable, stable companies | 2020-2021 Recovery |

### 1.2 Historical Evidence

| Event | Losses | Primary Driver |
|-------|--------|----------------|
| August 2007 Quant Crisis | ~15-30% in weeks | Extreme Value/Momentum crowding |
| February 2018 Volmageddon | ~20% in days | Short volatility / Momentum reversal |
| March 2020 COVID-19 | ~25% in weeks | Carry / Momentum crowding |

In each case, standard risk dashboards failed to provide early warning because **factor exposures are treated as secondary outputs rather than primary risk drivers**.

### 1.3 Who This Affects

| Stakeholder | Pain Point |
|-------------|------------|
| **Multi-Asset Portfolio Managers** | Facing drawdowns they cannot explain to their CIOs |
| **Risk Managers** | Tasked with monitoring risk but lacking tools to aggregate factor risk across silos |
| **Institutional Investors** | Unable to discern whether underlying managers are taking correlated factor bets |
| **Quantitative Researchers** | Seeking robust methods to monitor factor crowding and inform dynamic risk budgeting |

---

## 2. The Research Question

### 2.1 Primary Research Question

> *Given a portfolio's current holdings, factor loadings, and prevailing market conditions, can a machine learning model accurately predict whether the portfolio is entering a state of dangerous factor overconcentration that will result in a significant, factor-driven drawdown?*

### 2.2 Secondary Research Questions

1. Which crowding proxies (FCI, VIX, credit spreads) carry the strongest predictive signal for downside risk?
2. Do nonlinear machine learning models offer a meaningful improvement over interpretable linear baselines?
3. How does the predictive signal vary across distinct market regimes?

---

## 3. The Approach

### 3.1 Research Philosophy

> **Baseline first. Add complexity only when the data and empirical evidence justify it. Statistical rigor is non-negotiable.**

| Principle | Implementation |
|-----------|----------------|
| **Problem framing over model complexity** | Clear research question before any code |
| **Baseline-first development** | Threshold rules → Logistic Regression → Ensembles |
| **Simple, defensible methods** | No black-box models without justification |
| **Evidence over claims** | Bootstrap CIs, calibration, significance testing |
| **Reproducibility** | Public data only, full code, fixed random seeds |
| **Honest negative results** | Documented openly, framed as contribution |

### 3.2 Target Variable

The target variable is binary: 1 if the portfolio experiences a drawdown of at least 3% over the next 21 trading days, otherwise 0.

```
Y_t = 1 if Drawdown_{t, t+21} < -3%
Y_t = 0 otherwise
```

**Key Design Decisions (Empirically Validated):**

| Decision | Rationale | Evidence |
|----------|-----------|----------|
| **Drawdown threshold: -3%** | Optimal for event detection | Validation AUC: 0.7528, p < 0.001 |
| **Horizon: 21 trading days** | 1-month forward window | Multiple horizons tested; 21-day optimal |
| **No attribution threshold** | Attribution condition destroyed signal | Removal improved AUC 0.5465 → 0.7147 |

### 3.3 Data Sources

| Source | Variable | Frequency |
|--------|----------|-----------|
| Kenneth French Data Library | Fama-French 5-Factor Returns | Daily |
| Yahoo Finance (`yfinance`) | ETF Prices (SPY, AGG, GLD, IJS, EFA) | Daily |
| Yahoo Finance (`yfinance`) | VIX (^VIX) | Daily |
| Yahoo Finance (`yfinance`) | Credit Spread (HYG/LQD Ratio) | Daily |
| FRED (via `pandas_datareader`) | Yield Curve (T10Y2Y), Baa-10yr Spread (BAA10YM) | Daily |

**Portfolio Construction:** Synthetic multi-asset portfolio with equal weights:

| ETF | Asset Class | Ticker |
|-----|-------------|--------|
| SPY | US Equities | SPY |
| AGG | US Bonds | AGG |
| GLD | Gold | GLD |
| IJS | Small Cap Value | IJS |
| EFA | Developed ex-US | EFA |

### 3.4 Features

| Category | Specific Features | Justification |
|----------|-------------------|---------------|
| **Factor Betas** | Rolling 252-day exposures to Mkt-RF, SMB, HML, RMW, CMA | Captures portfolio factor tilts |
| **Factor Concentration Index (FCI)** | HHI: sum(|beta_k| / sum |beta_j|)^2 | Primary concentration metric |
| **FCI Dynamics** | 25-day MA, 20-day/30-day changes | Captures trend and velocity |
| **Macro/Market** | VIX level, VIX change, VIX volatility, log VIX | Market stress proxies |
| **Credit Spread** | HYG/LQD ratio, credit_high indicator | Credit market stress |
| **Persistence Features** | FCI_high, VIX_high, stress_confirm, stress_persistence | Reduces false positives |
| **Portfolio Volatility** | 60-day rolling volatility | Risk magnitude |

**Rejected Features (Based on Empirical Testing):**

| Feature | Reason Rejected |
|---------|-----------------|
| Momentum features (5d, 10d, 20d returns) | Killed too many true positives |
| Ratio/correlation features | Killed predictions |
| FCI change features (5d, 10d) | Too noisy |

---

## 4. Methodology

### 4.1 Time-Aware Splitting (No Look-Ahead Bias)

| Split | Period | Purpose | Samples |
|-------|--------|---------|---------|
| **Training** | January 2010 – December 2016 | Model training, feature engineering | 1,497 |
| **Validation** | January 2017 – December 2022 | Threshold tuning, hyperparameter selection | 1,499 |
| **Test** | January 2023 – December 2024 | **ONE-TIME** final evaluation | 477 (27 events) |

### 4.2 Models Tested

| Tier | Model | Description |
|------|-------|-------------|
| **1** | Heuristic Rule | FCI > 90th percentile |
| **1** | Enhanced Rule | FCI > 90% AND VIX > 20 AND Credit > median |
| **2** | Logistic Regression | Linear model with SMOTE |
| **3** | Random Forest | Tree-based ensemble with 100 estimators |
| **3** | XGBoost | Gradient boosting with Platt scaling (primary model) |

### 4.3 Statistical Rigor (Mandatory)

| Test | Implementation | Criterion |
|------|----------------|-----------|
| **Bootstrap Confidence Intervals** | 1,000 iterations, 95% CI for AUC-ROC | CI excludes 0.5 → significant |
| **Calibration Testing** | Expected Calibration Error (ECE) | ECE < 0.10 → calibrated |
| **Significance Testing** | AUC-ROC + bootstrap CI | Significant if lower bound > 0.5 |
| **No Test Set Leakage** | All tuning on validation set only | Strict chronological order |

---

## 5. Results

### 5.1 Target Analysis

| Metric | Value |
|--------|-------|
| Total days | 3,753 |
| Total events | 338 |
| Event rate | 9.01% |
| Crisis event rate (COVID-19) | 34.9% |
| Normal event rate | 8.4% |
| Crisis/Normal ratio | **4.15x** |
| Total clusters | 33 |
| Largest cluster | 29 events (Jan-Mar 2020) |

**Key Finding:** Events cluster heavily during crises, validating the target's ability to capture stress periods.

### 5.2 Feature Analysis

| Feature | Correlation with Target |
|---------|------------------------|
| log_vix | 0.0766 |
| vix_level | 0.0630 |
| fci_change_30 | 0.0584 |
| fci_change_20 | 0.0538 |
| beta_CMA | 0.0428 |

**Key Finding:** All feature correlations are < 0.1, indicating very weak individual predictive signal. This explains why ML models struggle to outperform simple rules.

### 5.3 Model Performance (Test Set)

| Model | AUC-ROC | 95% CI | Significant? | Precision | Recall | F1 |
|-------|---------|--------|--------------|-----------|--------|-----|
| Heuristic (FCI 90%) | 0.5115 | N/A | N/A | 0.0598 | 0.4074 | 0.1043 |
| Enhanced (FCI+VIX) | 0.4926 | N/A | N/A | 0.0476 | 0.0741 | 0.0580 |
| Logistic Regression | 0.3013 | [0.2077, 0.4057] | ❌ | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.4226 | [0.2887, 0.5460] | ❌ | 0.2143 | 0.0556 | 0.0950 |
| **XGBoost** | **0.4756** | **[0.3906, 0.5621]** | **❌** | **0.0256** | **0.0370** | **0.0303** |

### 5.4 Statistical Significance (XGBoost)

| Test | Value | Interpretation |
|------|-------|----------------|
| Observed AUC | 0.4756 | Below random (0.5) |
| 95% CI Lower | 0.3906 | Below 0.5 |
| 95% CI Upper | 0.5621 | Above 0.5 |
| **CI includes 0.5?** | **YES** | **NOT significant** |
| ECE | 0.0199 | Well-calibrated |
| Verdict | WARNING | Not significant (CI includes 0.5) |

---

## 6. Key Findings

### 6.1 The Null Hypothesis Cannot Be Rejected

After rigorous testing with proper validation methodology:

1. **Heuristic rule outperforms ML models**: FCI > 90% (AUC 0.5115) beats XGBoost (AUC 0.4756)
2. **No model is statistically significant**: All 95% CIs include 0.5
3. **XGBoost achieves near-random performance**: AUC 0.4756, CI includes 0.5
4. **Model is well-calibrated but useless**: ECE = 0.0199, but no discriminative power
5. **Features are too weak**: All correlations < 0.1
6. **The data does not support reliable prediction**: Null hypothesis cannot be rejected
7. **Macroeconomic data does not add signal**: FRED yield curve and credit spread features showed no correlation with target

### 6.2 What the Data Tells Us

| Finding | Implication |
|---------|-------------|
| Feature correlations < 0.1 | Individual features have no predictive signal |
| Model CIs include 0.5 | Results are statistically indistinguishable from random |
| Only 27 test events | Statistical power is extremely limited |
| Public data only | Proprietary data (flows, positioning) likely required |

### 6.3 What Would Be Needed for a Practical System

| Requirement | Current | Needed |
|-------------|---------|--------|
| Feature correlation | < 0.1 | > 0.2 |
| Precision | 0.0256 | > 0.30 |
| Recall | 0.0370 | > 0.60 |
| Test events | 27 | 100+ |
| Data timeframe | 2010-2024 | Extended to 2029+ |
| Data type | Public only | Proprietary (flows, positioning, 13F) |

---

## 7. What We Learned

### 7.1 Key Lessons

| Lesson | Evidence |
|--------|----------|
| **Target definition matters** | Removing attribution threshold improved signal significantly |
| **Validation methodology is essential** | Without it, results were misleading |
| **Statistical testing is non-negotiable** | Bootstrap CI revealed noise |
| **Negative results are valuable** | Save others from pursuing weak signals |
| **Public data is insufficient** | Proprietary data likely required for reliable prediction |
| **Start simple** | Heuristic rule outperformed all ML models |
| **Macro data doesn't help** | FRED yield curve and credit spread features showed no correlation |

### 7.2 Implications for Practitioners

| Stakeholder | Implication |
|-------------|-------------|
| **Portfolio Managers** | Don't trust ML risk alerts built on public data. You need proprietary data — positioning, flows, or short interest. |
| **Risk Managers** | Focus on structural risk (VaR, stress testing, risk decomposition) rather than prediction. Public data cannot reliably predict factor crowding events. |
| **Quant Researchers** | Use this project as a free benchmark. Test your proprietary data against it. If you beat the baseline, that's evidence your data adds value. |
| **Institutional Investors** | Factor crowding remains a real risk, but public data alone cannot predict it. Require managers to provide positioning and flow data for proper risk oversight. |

### 7.3 Hypotheses for Future Work

| Hypothesis | Proposed Test | Rationale |
|------------|---------------|-----------|
| H1: Proprietary data reveals signal | Add options data, 13F filings, short interest | Public factors are insufficient |
| H2: Alternative target definition works | Predict factor crowding directly | Current target may be too broad |
| H3: Different feature engineering improves signal | Nonlinear transformations, interaction terms | Current features too linear |
| H4: Crisis vs normal separation helps | Model regimes separately | Relationship may be non-stationary |
| H5: More data (to 2029) reveals signal | Extend data timeframe | More events may reveal pattern |

---

## 8. Economic Value

### 8.1 Cost Savings

This negative result saves time and money for everyone who would have tried this approach:

| Scenario | Time Saved | Cost Saved |
|----------|------------|------------|
| Single quant avoiding this path | 3 months | ~$25,000 |
| Small team (5 people) | 15 person-months | ~$125,000 |
| Industry-wide (100+ quants) | 300+ person-months | ~$2,500,000+ |

### 8.2 Practical Impact

| Impact | Description |
|--------|-------------|
| **Avoided false hedges** | PMs won't overreact to false alarms (precision = 0.0256 → 97% false positives) |
| **Redirected research** | Industry can focus on proprietary data where signal exists |
| **Better benchmarking** | Established reproducible baseline for factor crowding research |
| **Honest expectation setting** | Don't expect public data to solve this prediction problem |

---

## 9. Success Criteria & Actual Outcomes

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

## 10. Statistical Power Analysis

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

## 11. Open-Source Contribution

### 11.1 What This Project Contributes

1. ✅ A reproducible, rigorous benchmark for factor crowding detection
2. ✅ Honest documentation of a negative result with full transparency
3. ✅ Bootstrap confidence intervals and calibration testing for statistical rigor
4. ✅ Prevention of look-ahead bias in financial ML
5. ✅ Publication-quality visualizations (6 figures in PDF + PNG)
6. ✅ Complete experiment tracking with `outputs/all_runs.csv`
7. ✅ Clear evidence that public data is insufficient for this prediction task

### 11.2 Repository Structure

```
factor-exposure-sentinel/
├── README.md                    # Project overview and results
├── problem_framing.md           # Full research methodology
├── executive_summary.md         # This document
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
│   ├── visualization.py         # Minimal data-driven figures
│   └── visualization_academic.py # Academic-style figures
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

## 12. Quick Start

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
```

### View Results

```bash
# Check results
cat outputs/all_runs.csv

# View visualizations
open outputs/figures/  # On macOS
explorer outputs\figures\  # On Windows
```

---

## 13. Limitations

| Limitation | Implication |
|------------|-------------|
| **Data Constraints** | Uses price data only; does not incorporate proprietary flow data, 13F institutional ownership, or short-interest metrics |
| **Feature Strength** | All feature correlations with the target are < 0.1, indicating very weak signal |
| **Factor Coverage** | Limited to standard publicly available factor families (Fama-French 5-factor) |
| **Synthetic Portfolios** | Results may differ for real institutional portfolios with more complex structures |
| **Test Period** | Only 27 events in test period (2023-2024), limiting statistical power |
| **Power** | With 27 test events, the ability to detect a real effect is limited |

---

## 14. Conclusion

### 14.1 The Bottom Line

> *"Public data cannot reliably predict factor-concentration drawdowns. No ML model improved on a simple threshold rule. The signal is too weak. Move on to proprietary data or a different problem."*

### 14.2 What We Found

This project set out to determine whether factor concentration events can be reliably predicted using public data and machine learning. After rigorous experimentation with proper validation methodology:

**No statistically significant predictive relationship was found.**

The XGBoost model, despite being well-calibrated (ECE = 0.0199), fails on all practical criteria:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUC-ROC | 0.4756 | > 0.70 | ❌ |
| 95% CI | [0.3906, 0.5621] | Excludes 0.5 | ❌ |
| Precision | 0.0256 | > 0.30 | ❌ |
| Recall | 0.0370 | > 0.60 | ❌ |
| F1 | 0.0303 | > 0.30 | ❌ |

### 14.3 Why This Result Matters

The primary bottleneck is the weakness of available features (all correlations < 0.1), which provides insufficient predictive signal. Even with 338 events over 15 years, the model cannot distinguish signal from noise.

**This negative result is a valuable contribution:**

- It demonstrates that with public factors, synthetic portfolios, and the optimized target definition, factor concentration events cannot be reliably predicted
- It establishes a reproducible benchmark for future research
- It saves others from pursuing weak signals with public data
- It highlights the need for proprietary data (positioning, flows, short interest) for reliable prediction

### 14.4 Future Work

Future work would require alternative data sources (options flow, 13F filings, proprietary positioning data), better features (nonlinear transformations, interaction terms), or a revised target definition.

**The hypotheses worth testing:**

1. **Proprietary data reveals signal** — Add options data, 13F filings, short interest
2. **Alternative target definition works** — Predict factor crowding directly
3. **Different feature engineering improves signal** — Nonlinear transformations, interaction terms
4. **Crisis vs normal separation helps** — Model regimes separately
5. **More data (to 2029) reveals signal** — Extend data timeframe

---

## 15. Results Source

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

## 16. License & Disclaimer

**License:** MIT

**Disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not investment advice and does not claim to generate alpha or predict market movements. Past performance is not indicative of future results.

---

## 17. Author Information

**Ken Ira Lacson Talingting**
- GitHub: [github.com/kira-ml](https://github.com/kira-ml)
- LinkedIn: [linkedin.com/in/ken-ira-lacson-852026343](https://www.linkedin.com/in/ken-ira-lacson-852026343/)

---

## 18. References

1. Arnott, R., Kalesnik, V., & Wu, L. (2019). The incredible shrinking factor return. *Journal of Portfolio Management*.

2. Campbell, J. Y., & Shiller, R. J. (1988). Stock prices, earnings, and expected dividends. *Journal of Finance*.

3. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*.

4. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*.

5. Khandani, A. E., & Lo, A. W. (2007). What happened to the quants in August 2007? *Journal of Investment Management*.

---

**Last Updated:** September 2026
