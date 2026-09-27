"""
Member 1 — Batch preprocessing
Processes full 1 GB files in chunks.

Run: python src/run_preprocessing.py
"""
import pandas as pd
from src.preprocessing import preprocess_dataframe

CHUNK = 200_000


def process_and_save(in_path, out_path):
    """Read in chunks, preprocess, write incrementally."""
    first = True
    n_rows = 0
    for chunk in pd.read_csv(in_path, sep='\t', chunksize=CHUNK,
                          dtype=str, encoding='utf-8',
                          encoding_errors='replace'):
        chunk = preprocess_dataframe(chunk)
        chunk.to_csv(out_path, sep='\t', index=False,
                     mode='w' if first else 'a',
                     header=first, encoding='utf-8')
        first = False
        n_rows += len(chunk)
        print(f"  {out_path}: {n_rows:,} rows", end='\r')
    print(f"\n  ✅ {out_path}: {n_rows:,} rows done")


if __name__ == "__main__":
    print("Processing SOURCE 1...")
    process_and_save("dataset/train/train_source1.tsv",
                     "cleaned/source1_clean.tsv")

    print("Processing SOURCE 2...")
    process_and_save("dataset/train/train_source2.tsv",
                     "cleaned/source2_clean.tsv")

    print("Processing SOURCE 3...")
    process_and_save("dataset/train/train_source3.tsv",
                     "cleaned/source3_clean.tsv")

    print("\n✅ ALL DONE")