"""
target_analysis.py
------------------
Provides comprehensive analysis of the target variable for the Factor Exposure Sentinel project.
Analyzes event timing, clustering, seasonality, and regime-specific patterns.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def analyze_target(target: pd.Series, 
                   crisis_start: Optional[str] = "2020-01-01",
                   crisis_end: Optional[str] = "2020-04-30") -> Dict[str, Any]:
    """
    Comprehensive analysis of target variable.
    
    Parameters:
    -----------
    target : pd.Series
        Binary target variable with datetime index
    crisis_start : str, optional
        Start date of crisis period for regime analysis
    crisis_end : str, optional
        End date of crisis period for regime analysis
    
    Returns:
    --------
    Dict with all analysis results
    """
    target_series = target.dropna()
    event_dates = target_series[target_series == 1].index
    total_days = len(target_series)
    total_events = len(event_dates)
    
    results = {
        'total_days': total_days,
        'total_events': total_events,
        'event_rate': total_events / total_days,
        'event_dates': event_dates,
    }
    
    # 1. Event timing analysis
    results['event_timing'] = analyze_event_timing(event_dates)
    
    # 2. Yearly distribution
    results['yearly_distribution'] = analyze_yearly_distribution(target_series)
    
    # 3. Monthly seasonality
    results['monthly_distribution'] = analyze_monthly_distribution(target_series)
    
    # 4. Crisis vs normal regime analysis
    results['regime_analysis'] = analyze_regime_patterns(
        target_series, crisis_start, crisis_end
    )
    
    # 5. Event duration and clustering
    results['clustering'] = analyze_event_clustering(event_dates)
    
    return results


def analyze_event_timing(event_dates: pd.DatetimeIndex) -> Dict[str, Any]:
    """
    Analyze timing patterns of events.
    
    Returns:
    --------
    Dict with timing statistics
    """
    if len(event_dates) == 0:
        return {
            'first_event': None,
            'last_event': None,
            'date_range_days': 0,
            'sample_dates': []
        }
    
    return {
        'first_event': event_dates[0].strftime('%Y-%m-%d'),
        'last_event': event_dates[-1].strftime('%Y-%m-%d'),
        'date_range_days': (event_dates[-1] - event_dates[0]).days,
        'sample_dates': [d.strftime('%Y-%m-%d') for d in event_dates[:10]]
    }


def analyze_yearly_distribution(target_series: pd.Series) -> Dict[int, int]:
    """
    Count events by year.
    
    Returns:
    --------
    Dict with year -> event_count
    """
    events = target_series[target_series == 1]
    yearly_counts = events.groupby(events.index.year).count()
    return yearly_counts.to_dict()


def analyze_monthly_distribution(target_series: pd.Series) -> Dict[int, int]:
    """
    Count events by month (1-12).
    
    Returns:
    --------
    Dict with month -> event_count
    """
    events = target_series[target_series == 1]
    monthly_counts = events.groupby(events.index.month).count()
    
    # Ensure all months are represented
    full_counts = {month: monthly_counts.get(month, 0) for month in range(1, 13)}
    return full_counts


def analyze_regime_patterns(target_series: pd.Series, 
                           crisis_start: str,
                           crisis_end: str) -> Dict[str, Any]:
    """
    Compare event rates during crisis vs normal periods.
    
    Returns:
    --------
    Dict with regime comparison
    """
    crisis_mask = (target_series.index >= crisis_start) & (target_series.index <= crisis_end)
    
    # Crisis period
    crisis_events = target_series[crisis_mask].sum()
    crisis_total = target_series[crisis_mask].count()
    crisis_rate = crisis_events / crisis_total if crisis_total > 0 else 0
    
    # Normal period
    normal_mask = ~crisis_mask
    normal_events = target_series[normal_mask].sum()
    normal_total = target_series[normal_mask].count()
    normal_rate = normal_events / normal_total if normal_total > 0 else 0
    
    return {
        'crisis': {
            'start': crisis_start,
            'end': crisis_end,
            'days': crisis_total,
            'events': crisis_events,
            'rate': crisis_rate
        },
        'normal': {
            'days': normal_total,
            'events': normal_events,
            'rate': normal_rate
        },
        'rate_ratio': crisis_rate / normal_rate if normal_rate > 0 else np.nan
    }


def analyze_event_clustering(event_dates: pd.DatetimeIndex) -> Dict[str, Any]:
    """
    Analyze how events cluster together.
    
    Returns:
    --------
    Dict with clustering statistics
    """
    if len(event_dates) < 2:
        return {
            'min_gap': None,
            'max_gap': None,
            'mean_gap': None,
            'median_gap': None,
            'clusters': []
        }
    
    # Calculate gaps between consecutive events
    gaps = (event_dates[1:] - event_dates[:-1]).days
    
    # Convert to numpy array for statistical operations
    gaps_array = np.array(gaps)
    
    # Identify clusters (events within 5 days of each other)
    clusters = []
    current_cluster = []
    
    for i, gap in enumerate(gaps_array):
        if gap <= 5:
            if not current_cluster:
                current_cluster.append(event_dates[i])
            current_cluster.append(event_dates[i+1])
        else:
            if current_cluster:
                clusters.append({
                    'start': current_cluster[0].strftime('%Y-%m-%d'),
                    'end': current_cluster[-1].strftime('%Y-%m-%d'),
                    'size': len(current_cluster)
                })
                current_cluster = []
    
    if current_cluster:
        clusters.append({
            'start': current_cluster[0].strftime('%Y-%m-%d'),
            'end': current_cluster[-1].strftime('%Y-%m-%d'),
            'size': len(current_cluster)
        })
    
    return {
        'min_gap': int(gaps_array.min()) if len(gaps_array) > 0 else None,
        'max_gap': int(gaps_array.max()) if len(gaps_array) > 0 else None,
        'mean_gap': float(gaps_array.mean()) if len(gaps_array) > 0 else None,
        'median_gap': int(np.median(gaps_array)) if len(gaps_array) > 0 else None,
        'total_clusters': len(clusters),
        'clusters': clusters
    }


def print_target_analysis(results: Dict[str, Any]) -> None:
    """
    Pretty print target analysis results.
    """
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n[Basic Statistics]")
    print(f"   Total days: {results['total_days']:,}")
    print(f"   Total events: {results['total_events']:,}")
    print(f"   Event rate: {results['event_rate']*100:.3f}%")
    
    # Timing
    timing = results['event_timing']
    print(f"\n[Event Timing]")
    print(f"   First event: {timing['first_event']}")
    print(f"   Last event: {timing['last_event']}")
    print(f"   Date range: {timing['date_range_days']} days")
    print(f"   Sample event dates: {timing['sample_dates'][:5]}...")
    
    # Yearly distribution
    print(f"\n[Yearly Distribution]")
    for year, count in sorted(results['yearly_distribution'].items()):
        print(f"   {year}: {count} events")
    
    # Monthly distribution
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"\n[Monthly Distribution]")
    for month, count in results['monthly_distribution'].items():
        print(f"   {month_names[month-1]}: {count} events")
    
    # Regime analysis
    regime = results['regime_analysis']
    print(f"\n[Regime Analysis]")
    print(f"   Crisis period ({regime['crisis']['start']} to {regime['crisis']['end']}):")
    print(f"      {regime['crisis']['events']} events / {regime['crisis']['days']} days = {regime['crisis']['rate']*100:.1f}%")
    print(f"   Normal period:")
    print(f"      {regime['normal']['events']} events / {regime['normal']['days']} days = {regime['normal']['rate']*100:.1f}%")
    print(f"   Rate ratio (crisis / normal): {regime['rate_ratio']:.2f}x")
    
    # Clustering
    clustering = results['clustering']
    if clustering['min_gap'] is not None:
        print(f"\n[Event Clustering]")
        print(f"   Min gap between events: {clustering['min_gap']} days")
        print(f"   Max gap: {clustering['max_gap']} days")
        print(f"   Mean gap: {clustering['mean_gap']:.1f} days")
        print(f"   Median gap: {clustering['median_gap']} days")
        print(f"   Total clusters: {clustering['total_clusters']}")
        if clustering['clusters']:
            print(f"   Largest clusters:")
            sorted_clusters = sorted(clustering['clusters'], key=lambda x: x['size'], reverse=True)[:3]
            for cluster in sorted_clusters:
                print(f"      {cluster['size']} events from {cluster['start']} to {cluster['end']}")
    
    print("\n" + "="*60)


def get_analysis_summary(results: Dict[str, Any]) -> str:
    """
    Get a one-line summary of target analysis.
    """
    return (f"Target: {results['total_events']} events ({results['event_rate']*100:.2f}%), "
            f"crisis rate {results['regime_analysis']['crisis']['rate']*100:.1f}% vs "
            f"normal {results['regime_analysis']['normal']['rate']*100:.1f}%")