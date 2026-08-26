"""
main.py
-------
Simple pipeline orchestrator for Week 1 implementation.
Runs the entire workflow from data loading to evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import modules
from src.data_loader import fetch_all_data, fetch_etf_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features
from src.evaluate import evaluate_model, print_evaluation, save_results

# Simple Logistic Regression (no complex ML yet)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

def main():
    print("="*60)
    print("FACTOR EXPOSURE SENTINEL - WEEK 1 IMPLEMENTATION")
    print("="*60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    
    factors = data['factors']
    etf_prices = data['etf_prices']
    
    # 2. Create simple portfolio (equal weights for now)
    print("\n[2] Creating portfolio...")
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    
    # Calculate portfolio returns
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    # 3. Create target
    print("\n[3] Creating target variable...")
    target = create_target(portfolio_returns, factors)
    print(f"   Event rate: {target.mean():.3f} ({target.sum()} events)")
    
    # 4. Create features
    print("\n[4] Creating features...")
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights)
    
    # 5. Prepare data for modeling (chronological split)
    print("\n[5] Preparing train/test split...")
    train_end = "2018-12-31"
    test_start = "2019-01-01"
    
    X_train = features.loc[:train_end].dropna()
    y_train = target.loc[X_train.index]
    
    X_test = features.loc[test_start:].dropna()
    y_test = target.loc[X_test.index]
    
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # 6. Train simple model
    print("\n[6] Training logistic regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 7. Predict and evaluate
    print("\n[7] Evaluating model...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold from training data
    precision, recall, thresholds = precision_recall_curve(y_train, 
                                                           model.predict_proba(X_train_scaled)[:, 1])
    # Use threshold that balances precision and recall
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0.5
    
    results = evaluate_model(y_test, y_pred_proba, threshold=optimal_threshold)
    print_evaluation(results)
    
    # Save results to outputs directory
    run_id = save_results(
        results=results,
        feature_names=X_train.columns.tolist(),
        coefficients=model.coef_[0].tolist(),
        model_name="logistic_regression"
    )
    print(f"   Run ID: {run_id}")
    
    # 8. Feature importance
    print("\n[8] Feature importance:")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(importance.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("WEEK 1 IMPLEMENTATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()