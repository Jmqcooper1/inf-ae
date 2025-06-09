import pandas as pd
import numpy as np

INPUT_PATH = 'ease_rec/dataset/netflix/netflix.inter'
OUTPUT_PATH = 'ease_rec/dataset/netflix/netflix_25k.inter'
NUM_USERS = 25000  # target

# Load the .inter file
df = pd.read_csv(INPUT_PATH, sep="\t")

# Randomly sample 25K unique users
unique_users = df['user_id:token'].unique()
sampled_users = np.random.choice(unique_users, size=NUM_USERS, replace=False)

# Filter the dataframe
filtered_df = df[df['user_id:token'].isin(sampled_users)]

# Save to new file
filtered_df.to_csv(OUTPUT_PATH, sep="\t", index=False)

print(f"✅ Saved subsampled file with {len(filtered_df)} interactions from {NUM_USERS} users to {OUTPUT_PATH}")