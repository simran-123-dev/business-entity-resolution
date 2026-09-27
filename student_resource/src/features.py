import pandas as pd
import numpy as np
from rapidfuzz import distance

def fast_levenshtein(str1, str2):
    if pd.isna(str1) or pd.isna(str2) or not str1 or not str2:
        return 0.0
    return distance.Levenshtein.normalized_similarity(str(str1).lower(), str(str2).lower())

def fast_jaccard(str1, str2):
    if pd.isna(str1) or pd.isna(str2) or not str1 or not str2:
        return 0.0
    set1 = set(str(str1).lower().split())
    set2 = set(str(str2).lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def fast_token_overlap(str1, str2):
    if pd.isna(str1) or pd.isna(str2) or not str1 or not str2:
        return 0.0
    set1 = set(str(str1).lower().split())
    set2 = set(str(str2).lower().split())
    if not set1 or not set2:
        return 0.0
    min_len = min(len(set1), len(set2))
    return len(set1.intersection(set2)) / min_len if min_len > 0 else 0.0

def exact_match(val1, val2):
    if pd.isna(val1) or pd.isna(val2) or not val1 or not val2:
        return 0
    return 1 if str(val1).strip().lower() == str(val2).strip().lower() else 0

def extract_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Extract text columns safely
    c1_name = df['business_name_1'].fillna('') if 'business_name_1' in df.columns else pd.Series(['']*len(df), index=df.index)
    c2_name = df['business_name_2'].fillna('') if 'business_name_2' in df.columns else pd.Series(['']*len(df), index=df.index)
    
    c1_addr = df['business_address_1'].fillna('') if 'business_address_1' in df.columns else pd.Series(['']*len(df), index=df.index)
    c2_addr = df['business_address_2'].fillna('') if 'business_address_2' in df.columns else pd.Series(['']*len(df), index=df.index)
    
    c1_cntry = df['country_1'].fillna('') if 'country_1' in df.columns else pd.Series(['']*len(df), index=df.index)
    c2_cntry = df['country_2'].fillna('') if 'country_2' in df.columns else pd.Series(['']*len(df), index=df.index)

    print("  -> Calculating Name Similarities...")
    features['name_levenshtein'] = [fast_levenshtein(a, b) for a, b in zip(c1_name, c2_name)]
    features['name_jaccard'] = [fast_jaccard(a, b) for a, b in zip(c1_name, c2_name)]
    features['name_token_overlap'] = [fast_token_overlap(a, b) for a, b in zip(c1_name, c2_name)]

    print("  -> Calculating Address Similarities...")
    features['addr_levenshtein'] = [fast_levenshtein(a, b) for a, b in zip(c1_addr, c2_addr)]
    features['addr_jaccard'] = [fast_jaccard(a, b) for a, b in zip(c1_addr, c2_addr)]
    features['addr_token_overlap'] = [fast_token_overlap(a, b) for a, b in zip(c1_addr, c2_addr)]

    print("  -> Calculating Country Match...")
    features['country_match'] = [exact_match(a, b) for a, b in zip(c1_cntry, c2_cntry)]

    return features