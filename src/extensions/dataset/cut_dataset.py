import pandas as pd
import numpy as np
import sys
import os

DEFAULT_NUM_USERS = 25000

if len(sys.argv) < 2:
    print("Correct usage: python cut_dataset.py <dataset_name> [num_users] [dataset_filename (without .inter)]")
    sys.exit(1)

dataset = sys.argv[1]
num_users = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_NUM_USERS
dataset_filename = sys.argv[3] if len(sys.argv) >= 4 else dataset

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
input_path = os.path.join(base_dir, f"inf-ae/data/{dataset}/{dataset_filename}.inter")
output_path = os.path.join(base_dir, f"inf-ae/data/{dataset}/{dataset}_{int(num_users / 1000)}K.inter")

# Load .inter file
try:
    df = pd.read_csv(input_path, sep="\t")
except FileNotFoundError:
    print(f"File not found: {input_path}")
    sys.exit(1)

unique_users = df['user_id:token'].unique()

# let user know if enough users exist
if num_users > len(unique_users):
    print(f"Requested {num_users} users, but only {len(unique_users)} available")
    sys.exit(1)

# Sample users randomly
sampled_users = np.random.choice(unique_users, size=num_users, replace=False)
filtered_df = df[df['user_id:token'].isin(sampled_users)]

# Save samples users
filtered_df.to_csv(output_path, sep="\t", index=False)
print(f"Saved {len(filtered_df)} interactions from {num_users} users to: {output_path}")
