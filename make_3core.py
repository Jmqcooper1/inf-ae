import pandas as pd
from collections import Counter
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python make_3core.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]
input_path = f"data/{dataset}/{dataset}.inter"
output_path = f"data/{dataset}/{dataset}_3core.inter"

# Load the dataset
df = pd.read_csv(input_path, sep='\t')

def filter_3core(df, user_col, item_col, min_inter=3):
    """Iteratively filter users and items with < min_inter interactions"""
    while True:
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()

        valid_users = user_counts[user_counts >= min_inter].index
        valid_items = item_counts[item_counts >= min_inter].index

        original_size = len(df)
        df = df[df[user_col].isin(valid_users) & df[item_col].isin(valid_items)]

        if len(df) == original_size:
            break

    return df

# Run 3-core filtering
filtered_df = filter_3core(df, 'user_id:token', 'item_id:token', min_inter=3)

# Save filtered result
filtered_df.to_csv(output_path, sep='\t', index=False)

# Report final stats
print(f"✅ Filtered dataset saved to: {output_path}")
print(f"Remaining users: {filtered_df['user_id:token'].nunique()}")
print(f"Remaining items: {filtered_df['item_id:token'].nunique()}")
print(f"Remaining interactions: {len(filtered_df)}")
