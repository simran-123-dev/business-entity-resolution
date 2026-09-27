import pandas as pd
import joblib

try:
    from src.features import extract_features
except ModuleNotFoundError:
    from features import extract_features

def run_evaluation(test_candidates_df, threshold=0.5):
    clf = joblib.load('baseline_rf.pkl')
    
    print("  -> Extracting features for test set...")
    X_test = extract_features(test_candidates_df)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    results = pd.DataFrame()
    
    if 'id_1' in test_candidates_df.columns and 'id_2' in test_candidates_df.columns:
        results['id_1'] = test_candidates_df['id_1']
        results['id_2'] = test_candidates_df['id_2']
    elif 'pair_id' in test_candidates_df.columns:
        results['pair_id'] = test_candidates_df['pair_id']
    else:
        results['row_id'] = test_candidates_df.index
        
    results['match_probability'] = probs
    results['is_match'] = preds
    
    results.to_csv('matching_results.tsv', sep='\t', index=False)
    print("\nSUCCESS! 'matching_results.tsv' generated successfully.")