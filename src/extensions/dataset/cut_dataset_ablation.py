import pandas as pd
import numpy as np
import os

# Subsample sizes
SUBSAMPLE_SIZES = [5000, 10000, 25000, 50000, 100000]

# Base path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


# Input path and output dir
input_path = os.path.join(base_dir, "inf-ae/data/ml-20m/ml-20m_complete_3core.inter")
output_dir = os.path.join(base_dir, "inf-ae/data/ml-20m")

# Read the complete dataset
try:
    df = pd.read_csv(input_path, sep="\t")
except FileNotFoundError:
    print(f"File not found: {input_path}")
    exit(1)

# Count users
unique_users = df['user_id:token'].unique()

# For loop for subsample sizes 
for num_users in SUBSAMPLE_SIZES:

    print(f"Sampling {num_users} users")

    # Sample users (in a random way)
    sampled_users = np.random.choice(unique_users, size=num_users, replace=False)

    # Then filter the interactions
    filtered_df = df[df['user_id:token'].isin(sampled_users)]

    # Save the subsampled .inter file
    output_filename = f"ml-20m_{int(num_users / 1000)}K.inter"
    output_path = os.path.join(output_dir, output_filename)
    filtered_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {len(filtered_df)} interactions to {output_filename}")

print("Subsampling succesful")
