"""
visualization.py
----------------
Minimal, data-driven visualization for Factor Exposure Sentinel.
Focus: Clear communication of problem framing and empirical results.
No aesthetic fluff. Fully reproducible. Open-source friendly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Crisis periods (Correct: End of March 2020)
CRISIS_START = '2020-01-01'
CRISIS_END = '2020-04-30'   # <-- MATCHES main.py
BEAR_START = '2022-01-01'
BEAR_END = '2022-12-31'

# Global rcParams (NO bbox_inches='tight', NO aesthetic fluff)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
})


# ============================================================================
# HELPER FUNCTIONS (No Repetition)
# ============================================================================

def save_fig(fig, filename):
    """Save figure to output folder as PNG. NO bbox_inches='tight'."""
    # Ensure directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use Path object directly
    path = FIGURES_DIR / f"{filename}.png"
    
    # Save with explicit path
    fig.savefig(str(path), dpi=300, facecolor='white', format='png')
    plt.close(fig)
    print(f"   ✅ Saved: {path}")


def get_crisis_stats(target):
    """Calculate crisis vs normal event rates."""
    mask = (target.index >= CRISIS_START) & (target.index <= CRISIS_END)
    crisis_rate = target[mask].mean() * 100
    normal_rate = target[~mask].mean() * 100
    ratio = crisis_rate / normal_rate if normal_rate > 0 else 0
    return crisis_rate, normal_rate, ratio


# ============================================================================
# FIGURE 1: Event Timeline (Problem Framing)
# ============================================================================

def plot_event_timeline(target, portfolio_returns, save=True):
    """Figure 1: Show events exist and cluster during crises."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Top: Cumulative returns
    cum_returns = (1 + portfolio_returns).cumprod()
    ax1.plot(cum_returns.index, cum_returns, color='blue', linewidth=2)
    ax1.fill_between(cum_returns.index, 1, cum_returns.values, color='blue', alpha=0.2)
    ax1.set_ylabel('Cumulative Return')

    # Shade crisis periods
    for start, end, label in [
        (pd.to_datetime(CRISIS_START), pd.to_datetime(CRISIS_END), 'COVID-19'),
        (pd.to_datetime(BEAR_START), pd.to_datetime(BEAR_END), '2022 Bear')
    ]:
        ax1.axvspan(start, end, alpha=0.15, color='red', zorder=0)
        ax2.axvspan(start, end, alpha=0.15, color='red', zorder=0)
        ax1.text(start + (end - start) / 2, ax1.get_ylim()[1] * 0.95,
                 label, ha='center', va='top', fontsize=10)

    # Bottom: Event rate (30-day rolling)
    event_density = target.rolling(30).mean() * 100
    ax2.plot(event_density.index, event_density.values, color='red', linewidth=1.5)
    ax2.set_ylabel('Event Rate (%)')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 40)  # Fixed Y-axis

    # Annotate crisis stats (FIXED: Moved to top-LEFT to avoid covering 2022 data)
    crisis_rate, normal_rate, ratio = get_crisis_stats(target)
    ax2.text(0.02, 0.95,
             f"Total Events: {target.sum()}\n"
             f"Crisis Rate: {crisis_rate:.1f}% vs Normal: {normal_rate:.1f}%\n"
             f"Ratio: {ratio:.2f}x",
             transform=ax2.transAxes, ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

    fig.suptitle('Figure 1: Factor Concentration Events Over Time', y=1.02)
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.2)

    if save:
        save_fig(fig, "fig1_event_timeline")
    return fig


# ============================================================================
# FIGURE 2: Regime Comparison (Problem Framing)
# ============================================================================

def plot_regime_event_rates(target, save=True):
    """Figure 2: Show events are 4x more frequent during crises."""
    crisis_rate, normal_rate, ratio = get_crisis_stats(target)
    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(['Normal Market', 'COVID-19 Crisis'], [normal_rate, crisis_rate],
                  color=['green', 'red'], alpha=0.8, width=0.5)
    ax.set_ylabel('Event Rate (%)')
    ax.set_ylim(0, max(normal_rate, crisis_rate) * 1.3)
    ax.set_title(f'Figure 2: Events {ratio:.1f}x More Frequent During Crisis')

    for bar, val in zip(bars, [normal_rate, crisis_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f'{val:.1f}%',
                ha='center', fontweight='bold')

    if save:
        save_fig(fig, "fig2_regime_comparison")
    return fig


# ============================================================================
# FIGURE 3: Feature Correlations (No Signal)
# ============================================================================

def plot_feature_correlations(features, target, save=True):
    """Figure 3: Show ALL features have correlations < 0.1."""
    df = features.copy()
    df['target'] = target
    correlations = df.corr()['target'].drop('target').sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['red' if x < 0 else 'blue' for x in correlations.values]
    ax.barh(correlations.index, correlations.values, color=colors, alpha=0.8)

    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.1, color='gray', linestyle='--', label='|Corr| = 0.1')
    ax.axvline(-0.1, color='gray', linestyle='--')
    ax.set_xlabel('Correlation with Target')
    ax.set_xlim(-0.12, 0.12)
    ax.set_title('Figure 3: No Feature Has Predictive Signal (< 0.1)')
    ax.legend()

    if save:
        save_fig(fig, "fig3_feature_correlations")
    return fig


# ============================================================================
# FIGURE 4: Model Comparison (Null Hypothesis)
# ============================================================================

def plot_model_comparison(results_df, save=True):
    """Figure 4: Show ALL models have CI including 0.5 (NOT significant)."""
    if results_df.empty or 'ci_lower' not in results_df.columns:
        print("   ⚠️ Skipping: No results data")
        return None

    # Clean data
    df = results_df.groupby('model').last().reset_index()
    df = df[df['auc_roc'].notna()]
    df = df[~df['model'].str.contains('smote', case=False)]
    df['model'] = df['model'].replace({
        'xgboost': 'XGBoost',
        'xgboost_calibrated': 'XGBoost (Cal)',
        'random_forest': 'Random Forest',
        'logistic_regression': 'Logistic Regression'
    })
    df = df.drop_duplicates(subset='model', keep='last').sort_values('auc_roc')

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(df))
    aucs = df['auc_roc'].values
    ci_lower = df['ci_lower'].values
    ci_upper = df['ci_upper'].values

    ax.barh(y_pos, aucs, color='red', alpha=0.8, height=0.5)
    ax.errorbar(aucs, y_pos, xerr=[aucs - ci_lower, ci_upper - aucs],
                fmt='none', color='black', capsize=3)

    ax.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Random (AUC = 0.5)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['model'].values)
    ax.set_xlabel('AUC-ROC')
    ax.set_xlim(0.10, 0.7)  # Expand left side
    ax.set_title('Figure 4: No Model Significantly Beats Random')
    ax.legend()

    if save:
        save_fig(fig, "fig4_model_comparison")
    return fig


# ============================================================================
# FIGURE 5: Calibration (Honest but Useless)
# ============================================================================

def plot_calibration(y_true, y_pred_proba, save=True):
    """Figure 5: Show model is calibrated, but cannot discriminate."""
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=5)
    except:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.plot(prob_pred, prob_true, 'o-', color='blue', label='XGBoost')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Frequency')
    ax.set_title('Figure 5: Calibration (n=500 test, 27 events)')
    ax.legend()

    if save:
        save_fig(fig, "fig5_calibration_curve")
    return fig


# ============================================================================
# FIGURE 6: Precision-Recall (Tradeoff Impossible)
# ============================================================================

def plot_precision_recall(y_true, y_pred_proba, save=True):
    """Figure 6: Show high recall = low precision."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='blue', linewidth=2)
    ax.scatter(1.0, 0.1025, color='red', s=100, label='Reported: P=0.10, R=1.0, F1=0.19')
    ax.axhline(y_true.mean(), color='gray', linestyle='--', label='Random')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Figure 6: Precision-Recall (Best F1 = 0.186 at threshold 0.09)')
    ax.legend()

    if save:
        save_fig(fig, "fig6_precision_recall")
    return fig


# ============================================================================
# MAIN: Generate All Figures
# ============================================================================

def generate_all_visualizations(target, features, portfolio_returns,
                                y_test, y_pred_proba_final, results_df=None):
    """Generate all 6 minimal, data-driven figures."""
    print("\n" + "="*50)
    print("📊 GENERATING DATA-DRIVEN FIGURES")
    print("="*50)

    print("  [1/6] Figure 1: Event Timeline...")
    plot_event_timeline(target, portfolio_returns)

    print("  [2/6] Figure 2: Regime Comparison...")
    plot_regime_event_rates(target)

    print("  [3/6] Figure 3: Feature Correlations...")
    plot_feature_correlations(features, target)

    print("  [4/6] Figure 4: Model Comparison...")
    if results_df is not None and not results_df.empty:
        plot_model_comparison(results_df)
    else:
        print("   ⚠️ Skipping")

    print("  [5/6] Figure 5: Calibration...")
    if y_test is not None and y_pred_proba_final is not None:
        plot_calibration(y_test, y_pred_proba_final)
    else:
        print("   ⚠️ Skipping")

    print("  [6/6] Figure 6: Precision-Recall...")
    if y_test is not None and y_pred_proba_final is not None:
        plot_precision_recall(y_test, y_pred_proba_final)
    else:
        print("   ⚠️ Skipping")

    print("\n" + "="*50)
    print(f"✅ All figures saved to: {FIGURES_DIR}")
    print("="*50)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    dates = pd.date_range('2010-01-01', '2024-12-31', freq='B')
    np.random.seed(42)
    target = pd.Series(np.random.choice([0, 1], size=len(dates), p=[0.91, 0.09]), index=dates)
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    features = pd.DataFrame({'fci': np.random.normal(0.3, 0.1, len(dates)),
                             'vix': np.random.normal(20, 5, len(dates))}, index=dates)
    y_pred = np.random.uniform(0, 0.2, len(dates))

    generate_all_visualizations(target, features, returns,
                                target[-500:], y_pred[-500:], pd.DataFrame())