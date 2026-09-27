import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib

try:
    from src.features import extract_features
except ModuleNotFoundError:
    from features import extract_features

def train_and_optimize(candidates_df, ground_truth_df=None):
    merged_df = candidates_df.copy()
    
    # Auto-detect label column
    label_col = None
    for c in ['label', 'is_match', 'match', 'target']:
        if c in merged_df.columns:
            label_col = c
            break
            
    if label_col is None:
        raise KeyError("Train file me label / is_match column nahi mili! Check train file columns.")

    merged_df['label'] = merged_df[label_col].astype(int)
    
    X = extract_features(merged_df)
    y = merged_df['label']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f05 = 0.0
    
    print("\n--- Tuning Probability Threshold for Max F0.5 ---")
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (val_probs >= thresh).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        
        print(f"Threshold: {thresh:.2f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F0.5: {f05:.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            
    print(f"\nBest Threshold: {best_threshold:.2f} | Best F0.5 Score: {best_f05:.4f}")
    joblib.dump(clf, 'baseline_rf.pkl')
    return clf, best_threshold