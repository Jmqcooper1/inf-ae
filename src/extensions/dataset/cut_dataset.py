import pandas as pd
import numpy as np
import sys
import os

NUM_USERS = 25000

if len(sys.argv) < 2:
    print("Usage: python sample_users.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]
input_path = f"data/{dataset}/{dataset}.inter"
output_path = f"data/{dataset}/{dataset}_25k.inter"

df = pd.read_csv(input_path, sep="\t")

unique_users = df['user_id:token'].unique()

if NUM_USERS > len(unique_users):
    print(f"Requested {NUM_USERS} users, but only {len(unique_users)} available.")
    sys.exit(1)

sampled_users = np.random.choice(unique_users, size=NUM_USERS, replace=False)

filtered_df = df[df['user_id:token'].isin(sampled_users)]
filtered_df.to_csv(output_path, sep="\t", index=False)

print(f"Saved {len(filtered_df)} interactions from {NUM_USERS} users to: {output_path}")
