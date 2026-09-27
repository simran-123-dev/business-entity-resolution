import os
import pandas as pd
import numpy as np

try:
    from src.features import extract_features
    from src.model import train_and_optimize
    from src.evaluate import run_evaluation
except ModuleNotFoundError:
    from features import extract_features
    from model import train_and_optimize
    from evaluate import run_evaluation

def load_and_merge_sources(base_dir, split='train', max_samples=20000):
    print(f"\n--- Processing {split.upper()} Data Sources ---")
    split_dir = os.path.join(base_dir, split)
    
    # Load multi-source files
    s1 = pd.read_csv(os.path.join(split_dir, f'{split}_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(split_dir, f'{split}_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(split_dir, f'{split}_source3.tsv'), sep='\t')
    
    s23 = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=['entity_id'])
    
    if split == 'train':
        gt = pd.read_csv(os.path.join(split_dir, 'train_ground_truth.tsv'), sep='\t')
        
        # Build Positive Pairs (Label = 1)
        gt_clean = gt.dropna(subset=['matched_entity_ids']).copy()
        gt_clean['id_2'] = gt_clean['matched_entity_ids'].astype(str).str.split()
        gt_expanded = gt_clean.explode('id_2')
        gt_expanded = gt_expanded.rename(columns={'source1_entity_id': 'id_1'})
        gt_expanded['label'] = 1
        
        pos_df = gt_expanded.sample(n=min(max_samples, len(gt_expanded)), random_state=42)
        
        # Build Negative Pairs (Label = 0)
        neg_s1 = s1['entity_id'].sample(n=len(pos_df), random_state=42, replace=True).values
        neg_s23 = s23['entity_id'].sample(n=len(pos_df), random_state=42, replace=True).values
        neg_df = pd.DataFrame({'id_1': neg_s1, 'id_2': neg_s23, 'label': 0})
        
        pairs_df = pd.concat([pos_df[['id_1', 'id_2', 'label']], neg_df], ignore_index=True)
    else:
        # Test candidate pairs sample
        sample_s1 = s1.sample(n=min(3000, len(s1)), random_state=42)
        sample_s23 = s23.sample(n=min(3000, len(s23)), random_state=42)
        
        sample_s1['key'] = 1
        sample_s23['key'] = 1
        pairs_df = sample_s1[['entity_id', 'key']].merge(sample_s23[['entity_id', 'key']], on='key').drop(columns=['key'])
        pairs_df.columns = ['id_1', 'id_2']
        pairs_df['label'] = 0

    # Merge entity attribute details (name, address, country)
    merged = pairs_df.merge(s1, left_on='id_1', right_on='entity_id', how='left')
    merged = merged.rename(columns={'business_name': 'business_name_1', 'business_address': 'business_address_1', 'country': 'country_1'})
    
    merged = merged.merge(s23, left_on='id_2', right_on='entity_id', how='left')
    merged = merged.rename(columns={'business_name': 'business_name_2', 'business_address': 'business_address_2', 'country': 'country_2'})
    
    print(f"Successfully loaded {len(merged)} pairs with 0s and 1s!")
    return merged

if __name__ == '__main__':
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir) and os.path.exists('student_resource/dataset'):
        dataset_dir = 'student_resource/dataset'
        
    print("--- 1. Preparing Candidate Pairs ---")
    train_merged = load_and_merge_sources(dataset_dir, split='train')
    test_merged = load_and_merge_sources(dataset_dir, split='test')
    
    print("\n--- 2. Training Random Forest & Threshold Optimization ---")
    model, best_threshold = train_and_optimize(train_merged)
    
    print("\n--- 3. Generating Final Test Predictions ---")
    run_evaluation(test_merged, threshold=best_threshold)