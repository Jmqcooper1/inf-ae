import pandas as pd
import numpy as np
import os
import h5py

from recbole.quick_start import run_recbole

import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

baselines = ['LightGCN', 'EASE', 'MultiVAE', 'Pop']
datasets = ['douban', 'magazine', 'ml-1m', 'ml-20m', 'netflix', 'steam']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

for dataset_name in datasets:
    for model_name in baselines:
        print(f"===== Running {model_name} on {dataset_name} =====")

        DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', dataset_name))

        try:
            # --- Data Preparation ---
            print("Preparing data...")
            os.makedirs(DATASET_PATH, exist_ok=True)

            hdf5_path = os.path.join(DATASET_PATH, "total_data.hdf5")
            npz_path = os.path.join(DATASET_PATH, "index.npz")

            if not (os.path.exists(hdf5_path) and os.path.exists(npz_path)):
                print(f"Data files (total_data.hdf5 or index.npz) not found for {dataset_name}, checking for existing .inter files.")
                train_path = os.path.join(DATASET_PATH, f"{dataset_name}.train.inter")
                if not os.path.exists(train_path):
                    print(f"No usable data found for {dataset_name}, skipping.")
                    continue

            else:
                with h5py.File(hdf5_path, "r") as f:
                    users = f["user"][:]
                    items = f["item"][:]
                    ratings = f["rating"][:]
                df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})

                index = np.load(npz_path)["data"]
                train_df = df[index == 0]
                val_df = df[index == 1]
                test_df = df[index == 2]

                train_df.to_csv(f'{DATASET_PATH}/{dataset_name}.train.inter', sep='\t', index=False)
                val_df.to_csv(f'{DATASET_PATH}/{dataset_name}.valid.inter', sep='\t', index=False)
                test_df.to_csv(f'{DATASET_PATH}/{dataset_name}.test.inter', sep='\t', index=False)
                print("Data preparation complete.")

            # --- RecBole Configuration ---
            parameter_dict = {
                'data_path': DATASET_PATH,
                'dataset': dataset_name,
                'USER_ID_FIELD': 'user_id',
                'ITEM_ID_FIELD': 'item_id',
                'RATING_FIELD': 'rating',
                'load_col': {
                    'inter': ['user_id', 'item_id', 'rating']
                },
                'eval_setting': 'RO',
                'train_neg_sample_args': None,
                'model': model_name,
                'metrics': ['AUC', 'NDCG', 'HR', 'PSP'],
                'topk': [10, 100],
                'valid_metric': 'NDCG@10',
                'metric_decimal_place': 4,
                'seed': SEED,
            }

            # --- Run Experiment ---
            run_recbole(config_dict=parameter_dict)

        except Exception as e:
            print(f"Error running {model_name} on {dataset_name}: {e}")
            print("Continuing to next run...")