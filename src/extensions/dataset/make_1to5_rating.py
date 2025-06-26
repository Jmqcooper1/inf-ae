import pandas as pd
from collections import Counter
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python make_1to5_rating.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
input_path = os.path.join(base_dir, f"inf-ae/data/{dataset}/{dataset}.inter")
output_path = os.path.join(base_dir, f"inf-ae/data/{dataset}/{dataset}_grouped.inter")

df = pd.read_csv(input_path, sep='\t')

df = df[['user_id:token', 'rating:float', 'item_id:token']]
df.columns = ['user_id', 'rating', 'item_id']

# Convert play hours using quantile binning 
df = df[df['rating'] > 0]

quantiles = df['rating'].quantile([0.2, 0.4, 0.6, 0.8]).values

def map_hours_to_rating(hours):
    if hours <= quantiles[0]:
        return 1.0
    elif hours <= quantiles[1]:
        return 2.0
    elif hours <= quantiles[2]:
        return 3.0
    elif hours <= quantiles[3]:
        return 4.0
    else:
        return 5.0

df['rating'] = df['rating'].apply(map_hours_to_rating)
print(df['rating'].value_counts(normalize=True))

df = df[['user_id', 'item_id', 'rating']]

df.columns = ['user_id:token', 'item_id:token', 'rating:float']

df.to_csv(output_path, sep='\t', index=False)
print(f"Saved processed file to {output_path}")
