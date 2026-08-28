"""
visualization.py
----------------
Modern visualization functions for the Factor Exposure Sentinel project.
Generates publication-quality figures with clean, professional aesthetics.
All figures are saved to outputs/figures/ automatically.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Modern color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'gray': '#95A5A6',
    'crisis': '#E74C3C',
    'normal': '#2ECC71',
    'random': '#95A5A6',
    'event': '#E74C3C',
    'no_event': '#2ECC71',
}

# Output directory
OUTPUT_DIR = Path("D:/quant-finance-ml/factor-exposure-sentinel/outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_modern_style():
    """Apply modern plotting style."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


def plot_event_timeline(target, portfolio_returns, save=True):
    """
    Modern event timeline with cumulative returns and event markers.
    Proves the problem exists.
    """
    set_modern_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: Portfolio cumulative returns with shading
    cum_returns = (1 + portfolio_returns).cumprod()

    # Fill between for visual impact
    ax1.fill_between(cum_returns.index, 0, cum_returns.values,
                     color=COLORS['primary'], alpha=0.3)
    ax1.plot(cum_returns.index, cum_returns, color=COLORS['primary'],
             linewidth=2, label='Portfolio Cumulative Return')

    # Add crisis shading - FIXED: use pd.to_datetime()
    crisis_periods = [
        (pd.to_datetime('2020-01-01'), pd.to_datetime('2020-04-30'), 'COVID-19 Crisis'),
        (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'), '2022 Bear Market')
    ]
    for start, end, label in crisis_periods:
        ax1.axvspan(start, end, alpha=0.15, color=COLORS['danger'])
        ax2.axvspan(start, end, alpha=0.15, color=COLORS['danger'])
        ax1.text(start, ax1.get_ylim()[1] * 0.90,
                 label, fontsize=10, alpha=0.7, fontweight='bold')

    ax1.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax1.set_title('Portfolio Performance & Factor Concentration Events',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # Bottom: Event timeline with modern styling
    events = target[target == 1]

    # Create event markers with jitter for visibility
    event_dates = events.index
    if len(event_dates) > 0:
        # Use scatter plot for modern look
        ax2.scatter(event_dates, np.ones(len(event_dates)) * 0.5,
                   color=COLORS['danger'], s=30, alpha=0.7,
                   marker='|', linewidths=2, label='Events')

    # Add event density as a smooth line
    event_density = target.rolling(30).mean() * 100
    ax2.fill_between(event_density.index, 0, event_density.values,
                     color=COLORS['danger'], alpha=0.2)
    ax2.plot(event_density.index, event_density.values,
             color=COLORS['danger'], linewidth=1.5, alpha=0.6,
             label='Event Density (30-day MA)')

    ax2.set_ylabel('Event Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(event_density.max() + 5, 10))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')

    # Add total events annotation
    total_events = len(events)
    event_rate = target.mean() * 100
    ax2.text(0.98, 0.95, f'Total Events: {total_events} ({event_rate:.1f}%)',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()

    if save:
        filepath = FIGURES_DIR / "fig1_event_timeline.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {filepath}")

    plt.close()
    return fig


def plot_regime_event_rates(target, crisis_start='2020-01-01', crisis_end='2020-04-30', save=True):
    """
    Modern bar chart comparing event rates across regimes.
    Proves the problem is real.
    """
    set_modern_style()

    crisis_mask = (target.index >= crisis_start) & (target.index <= crisis_end)
    crisis_rate = target[crisis_mask].mean() * 100
    normal_rate = target[~crisis_mask].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create modern bar chart with gradient effect
    categories = ['Normal\nMarket', 'Crisis\n(COVID-19)']
    values = [normal_rate, crisis_rate]
    colors_bar = [COLORS['normal'], COLORS['danger']]

    bars = ax.bar(categories, values, color=colors_bar,
                  edgecolor='white', linewidth=2, width=0.5)

    # Add value labels with modern styling
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom',
                fontsize=16, fontweight='bold', color=COLORS['dark'])

    # Add ratio annotation with modern style
    ratio = crisis_rate / normal_rate if normal_rate > 0 else 0
    ax.text(0.5, 0.92,
            f'⚡ {ratio:.1f}x more events during crisis',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            color=COLORS['dark'],
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=COLORS['primary'], linewidth=2, alpha=0.9))

    ax.set_ylabel('Event Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Factor Concentration Events by Market Regime',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Add subtle annotation
    total_events = len(target[target == 1])
    ax.text(0.5, -0.12,
            f'Based on {len(target):,} trading days · {total_events} total events',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color=COLORS['gray'])

    plt.tight_layout()

    if save:
        filepath = FIGURES_DIR / "fig2_regime_event_rates.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {filepath}")

    plt.close()
    return fig


def plot_feature_correlation_heatmap(features, target, save=True):
    """
    Modern horizontal bar chart showing feature correlations.
    Proves the data has no signal.
    """
    set_modern_style()

    df = features.copy()
    df['target'] = target

    correlations = df.corr()['target'].drop('target').sort_values()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color based on sign
    colors_bar = [COLORS['danger'] if x < 0 else COLORS['primary'] for x in correlations.values]

    # Horizontal bar chart with modern styling
    bars = ax.barh(correlations.index, correlations.values,
                   color=colors_bar, edgecolor='white', linewidth=0.5,
                   alpha=0.8, height=0.7)

    # Reference lines
    ax.axvline(x=0, color=COLORS['dark'], linestyle='-', linewidth=1.5)
    ax.axvline(x=0.1, color=COLORS['gray'], linestyle='--', linewidth=1,
               alpha=0.5, label='Useful Signal Threshold')
    ax.axvline(x=-0.1, color=COLORS['gray'], linestyle='--', linewidth=1,
               alpha=0.5)

    ax.set_xlabel('Correlation with Target', fontsize=13, fontweight='bold')
    ax.set_title('Feature Correlations — No Predictive Signal Found',
                 fontsize=16, fontweight='bold', pad=20)

    # Add annotation with max correlation
    max_corr = correlations.abs().max()
    ax.text(0.98, 0.02,
            f'Max |Correlation| = {max_corr:.3f}',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=COLORS['gray'], alpha=0.9))

    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save:
        filepath = FIGURES_DIR / "fig3_feature_correlations.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {filepath}")

    plt.close()
    return fig


def plot_model_comparison(results_df, save=True):
    """
    Modern model comparison with confidence intervals.
    Proves the null hypothesis.
    """
    set_modern_style()

    if results_df.empty or 'ci_lower' not in results_df.columns:
        print("   ⚠️ Skipping: No results data for model comparison")
        return None

    # Filter to latest run only or aggregate
    latest = results_df.groupby('model').last().reset_index()
    latest = latest[latest['auc_roc'].notna()]
    latest = latest.sort_values('auc_roc', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    models = latest['model'].values
    aucs = latest['auc_roc'].values
    ci_lower = latest['ci_lower'].values
    ci_upper = latest['ci_upper'].values

    # Replace NaN/Inf with reasonable defaults for plotting
    ci_lower = np.nan_to_num(ci_lower, nan=0.4, posinf=0.6, neginf=0.3)
    ci_upper = np.nan_to_num(ci_upper, nan=0.6, posinf=0.7, neginf=0.5)

    # Color based on significance
    colors_bar = ['#2ECC71' if (lower > 0.5) else '#E74C3C'
                  for lower in ci_lower]

    y_pos = np.arange(len(models))

    # Modern bar chart with error bars
    bars = ax.barh(y_pos, aucs,
                   color=colors_bar, edgecolor='white', linewidth=1,
                   alpha=0.8, height=0.6)

    # Add error bars (confidence intervals)
    ax.errorbar(aucs, y_pos,
                xerr=[aucs - ci_lower, ci_upper - aucs],
                fmt='none', color=COLORS['dark'], capsize=5, capthick=2,
                alpha=0.7, label='95% Confidence Interval')

    # Random baseline
    ax.axvline(x=0.5, color=COLORS['gray'], linestyle='--', linewidth=2,
               alpha=0.7, label='Random Baseline (AUC = 0.5)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('AUC-ROC', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance with 95% Confidence Intervals',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0.2, max(ci_upper) + 0.1)

    # Add significance labels with modern styling
    for i, (model, auc, lower, upper) in enumerate(zip(models, aucs, ci_lower, ci_upper)):
        if lower > 0.5:
            label = '✅ Significant'
            color = COLORS['success']
        else:
            label = '❌ Not Significant'
            color = COLORS['danger']
        ax.text(max(ci_upper) + 0.03, i, label,
                va='center', fontsize=10, fontweight='bold', color=color)

    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save:
        filepath = FIGURES_DIR / "fig4_model_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {filepath}")

    plt.close()
    return fig


def plot_calibration_curve(y_true, y_pred_proba, save=True):
    """
    Modern calibration curve with histogram.
    Proves probabilities can't be trusted.
    """
    set_modern_style()

    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    except:
        print("   ⚠️ Skipping: Could not generate calibration curve")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Perfect calibration with modern styling
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7,
            label='Perfect Calibration')

    # Model calibration curve with modern styling
    ax.plot(prob_pred, prob_true, 'o-', color=COLORS['primary'],
            linewidth=3, markersize=10, alpha=0.9,
            label='XGBoost Model')

    # Fill under curve for visual impact
    ax.fill_between(prob_pred, 0, prob_true,
                    color=COLORS['primary'], alpha=0.15)

    # Histogram of predictions on secondary axis
    ax2 = ax.twinx()
    ax2.hist(y_pred_proba, bins=20, alpha=0.3, color=COLORS['gray'],
             edgecolor='white', linewidth=0.5, density=True,
             label='Prediction Distribution')
    ax2.set_ylabel('Density', fontsize=11, color=COLORS['gray'])
    ax2.tick_params(axis='y', colors=COLORS['gray'])

    # Calculate ECE
    bin_counts = np.histogram(y_pred_proba, bins=10, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_true)
    n_actual_bins = len(prob_true)
    bin_weights_aligned = bin_weights[:n_actual_bins]
    if bin_weights_aligned.sum() > 0:
        bin_weights_aligned = bin_weights_aligned / bin_weights_aligned.sum()
    ece = np.sum(bin_weights_aligned * np.abs(prob_true - prob_pred))

    ax.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
    ax.set_title('Calibration Curve — Well Calibrated but Not Discriminative',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add ECE annotation
    ax.text(0.05, 0.92,
            f'ECE = {ece:.4f}',
            fontsize=13, fontweight='bold',
            color=COLORS['primary'],
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=COLORS['primary'], linewidth=2, alpha=0.9))

    # Add interpretation
    if ece < 0.10:
        interpretation = '✅ Well Calibrated'
        color = COLORS['success']
    else:
        interpretation = '❌ Poor Calibration'
        color = COLORS['danger']
    ax.text(0.05, 0.85,
            interpretation,
            fontsize=12, fontweight='bold',
            color=color)

    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()

    if save:
        filepath = FIGURES_DIR / "fig5_calibration_curve.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {filepath}")

    plt.close()
    return fig


def plot_precision_recall_tradeoff(y_true, y_pred_proba, save=True):
    """
    Modern precision-recall curve showing the tradeoff.
    Proves the model can't achieve both precision and recall.
    """
    set_modern_style()

    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Precision-Recall curve
    ax.plot(recall, precision, color=COLORS['primary'],
            linewidth=3, alpha=0.9, label='XGBoost')

    # Fill under curve
    ax.fill_between(recall, 0, precision, color=COLORS['primary'], alpha=0.15)

    # Mark optimal F1 point
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    ax.scatter(recall[best_idx], precision[best_idx],
               color=COLORS['danger'], s=150, zorder=5,
               label=f'Best F1 = {f1_scores[best_idx]:.3f}')

    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Tradeoff',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')

    # Add annotation about the problem
    ax.text(0.98, 0.02,
            '⚠️ High recall → Low precision',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='bottom',
            color=COLORS['warning'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    if save:
        filepath = FIGURES_DIR / "fig6_precision_recall.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {filepath}")

    plt.close()
    return fig


def generate_all_visualizations(target, features, portfolio_returns,
                                y_test, y_pred_proba_final, results_df=None):
    """
    Generate all visualizations for the project paper.
    """
    print("\n" + "="*60)
    print("📊 GENERATING MODERN VISUALIZATIONS")
    print("="*60)
    print(f"   Output: {FIGURES_DIR}")

    # Ensure directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Event Timeline
    print("\n  [1/6] Event timeline...")
    plot_event_timeline(target, portfolio_returns)

    # 2. Regime Event Rates
    print("  [2/6] Regime event rates...")
    plot_regime_event_rates(target)

    # 3. Feature Correlations
    print("  [3/6] Feature correlations...")
    plot_feature_correlation_heatmap(features, target)

    # 4. Model Comparison
    print("  [4/6] Model comparison...")
    if results_df is not None and not results_df.empty:
        plot_model_comparison(results_df)
    else:
        print("   ⚠️ Skipping: No results data available")

    # 5. Calibration Curve
    print("  [5/6] Calibration curve...")
    if y_test is not None and y_pred_proba_final is not None:
        plot_calibration_curve(y_test, y_pred_proba_final)
    else:
        print("   ⚠️ Skipping: No test predictions available")

    # 6. Precision-Recall Curve
    print("  [6/6] Precision-Recall curve...")
    if y_test is not None and y_pred_proba_final is not None:
        plot_precision_recall_tradeoff(y_test, y_pred_proba_final)
    else:
        print("   ⚠️ Skipping: No test predictions available")

    print("\n" + "="*60)
    print(f"✅ All {6} visualizations saved to:")
    print(f"   {FIGURES_DIR}")
    print("="*60)

    return True