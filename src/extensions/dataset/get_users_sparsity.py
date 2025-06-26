import pandas as pd
import os

datasets = ['ml-1m', 'ml-20m', 'steam', 'netflix', 'douban', 'magazine']

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

print(f"{'Dataset':<10} | {'#Items':>7} | {'Sparsity (%)':>13} | {'Avg. Interactions/User':>24} | {'Users':>10} | {'Interactions'}")

for dataset in datasets:
    path = os.path.join(base_dir, f"inf-ae/data/{dataset}/{dataset}.inter")
    df = pd.read_csv(path, sep='\t')

    num_users = df['user_id:token'].nunique()
    num_items = df['item_id:token'].nunique()
    num_interactions = len(df)

    sparsity = 1 - (num_interactions / (num_users * num_items))
    sparsity_percent = sparsity * 100

    avg_inter_per_user = num_interactions / num_users

    print(f"{dataset:<10} | {num_items:7} | {sparsity_percent:12.2f}% | {avg_inter_per_user:24.2f} | {num_users:10f} | {num_interactions}")
