"""
Member 1 — Quick EDA
Run: python notebooks/quick_eda.py > reports/eda_summary.txt
"""
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_sample
import pandas as pd


def describe(df, name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate entity_ids: {df['entity_id'].duplicated().sum()}")
    print(f"\nCountry distribution:")
    print(df['country'].value_counts(dropna=False).head(10).to_string())
    print(f"\nSample business names:")
    for n in df['business_name'].dropna().sample(5, random_state=1):
        print(f"  - {n}")
    print(f"\nSample addresses:")
    for a in df['business_address'].dropna().sample(5, random_state=1):
        print(f"  - {a}")


if __name__ == "__main__":
    s1 = load_sample("dataset/train/train_source1.tsv")
    s2 = load_sample("dataset/train/train_source2.tsv")
    s3 = load_sample("dataset/train/train_source3.tsv")
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv",
                     sep='\t', dtype=str, encoding='latin-1')

    describe(s1, "SOURCE 1 (Reference)")
    describe(s2, "SOURCE 2")
    describe(s3, "SOURCE 3")

    print(f"\n{'='*60}")
    print(f"  GROUND TRUTH")
    print(f"{'='*60}")
    gt['n_matches'] = gt['matched_entity_ids'].fillna('').apply(
        lambda x: len([i for i in x.split(',') if i.strip()]) if x else 0
    )
    print(f"Total S1 entities in GT: {len(gt):,}")
    print(f"Singletons (0 matches): {(gt['n_matches']==0).sum():,} "
          f"({(gt['n_matches']==0).mean()*100:.1f}%)")
    print(f"Avg matches per S1: {gt['n_matches'].mean():.2f}")
    print(f"Max matches: {gt['n_matches'].max()}")
    print(f"\nMatch count distribution:")
    print(gt['n_matches'].value_counts().sort_index().head(15).to_string())