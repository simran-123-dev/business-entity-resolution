
from src.preprocessing import load_sample, preprocess_dataframe

s1 = load_sample('dataset/train/train_source1.tsv', n=5000)
out = preprocess_dataframe(s1)

print('New columns:', out.columns.tolist())
print()
print(out[['business_name', 'business_name_norm', 'business_name_key',
           'country', 'country_norm', 'postal_code', 'house_number']].head(10).to_string())