# MEMBER 1 HANDOFF

## Files Delivered
- `src/preprocessing.py` — normalization functions
- `src/run_preprocessing.py` — batch script for 1 GB data
- `notebooks/quick_eda.py` — fast EDA script
- `cleaned/source{1,2,3}_clean.tsv` — cleaned data (12.5M rows total)
- `reports/eda_summary.txt` — raw EDA output

## Cleaned Data Row Counts
| File | Rows |
|------|------|
| cleaned/source1_clean.tsv | 2,206,821 |
| cleaned/source2_clean.tsv | 5,034,616 |
| cleaned/source3_clean.tsv | 5,285,603 |

## New Columns Added
| Column | Meaning | Example |
|--------|---------|---------|
| business_name_norm | lowercased, punct-stripped, suffix-removed | "starbucks" |
| business_name_key | first 4 chars of alpha-only name | "star" |
| business_address_norm | abbreviation-expanded address | "123 main street" |
| country_norm | canonical country | "united states" |
| postal_code | ZIP (US) / PIN (India) | "98101" |
| house_number | first numeric token | "123" |

## Key EDA Numbers (Full Dataset)
- **Total S1 entities in GT:** 2,206,821
- **Singletons (0 matches):** 123,247 (**5.6%**)
- **Avg matches per S1:** 3.46
- **Max matches:** 11
- **Match distribution peak:** 3 matches (~24%)
- **Country split (sample):** US ~60%, India ~40%
- **Missing addresses:** ~3% in S2, ~3% in S3
- **TEST HAS FRANCE** — do NOT hard-code {US, India}

## For MEMBER 2 (Blocking)
1. Primary key: `country_norm`
2. Secondary: `postal_code` (very selective)
3. Tertiary: `business_name_key` (first 4 chars)
4. Union of blocks → dedupe candidates
5. Keep candidate set SMALL — hackathon scores candidate set size

### Target candidate set size:
- Country block: ~60% US + ~40% India
- Postal code block: reduces to local area
- Name key block: catches typos
- Target: < 100 candidates per S1

## For MEMBER 3 (Features)
- **Name:** Levenshtein on `business_name_norm`, Jaccard on tokens
- **Address:** Levenshtein on `business_address_norm`
- **Exact matches:** `country_norm`, `postal_code`, `house_number`
- **Missing addresses:** ~3% in S2/S3 — handle as empty string
- **F0.5 is precision-heavy** → tune threshold HIGH

## Warnings
- No external APIs (disqualified)
- No hard-coded countries
- Every S1 must appear in output (singletons → empty)
- Final matches must be a SUBSET of candidates
- Encoding: mixed UTF-8/Latin-1; some records have garbled non-Latin text

## How to Load Cleaned Data
```python
import pandas as pd
s1 = pd.read_csv("cleaned/source1_clean.tsv", sep="\t", encoding="utf-8")
s2 = pd.read_csv("cleaned/source2_clean.tsv", sep="\t", encoding="utf-8")
s3 = pd.read_csv("cleaned/source3_clean.tsv", sep="\t", encoding="utf-8")