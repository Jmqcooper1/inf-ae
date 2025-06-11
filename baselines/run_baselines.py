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

DATASET_NAME = 'steam'
MODEL_NAME = 'MultiVAE'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

SAVE_TO_CSV = True

if SAVE_TO_CSV: 
    os.makedirs(DATASET_PATH, exist_ok=True)

    with h5py.File(os.path.join(DATASET_PATH, DATASET_NAME, "total_data.hdf5"), "r") as f:
        users = f["user"][:]
        items = f["item"][:]
        ratings = f["rating"][:]
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})

    index = np.load(os.path.join(DATASET_PATH, "index.npz"))["data"]
    train_df = df[index == 0]
    val_df = df[index == 1]
    test_df = df[index == 2]

    train_df.to_csv(f'{DATASET_PATH}/{DATASET_NAME}.train.inter', sep='\t', index=False)
    val_df.to_csv(f'{DATASET_PATH}/{DATASET_NAME}.valid.inter', sep='\t', index=False)
    test_df.to_csv(f'{DATASET_PATH}/{DATASET_NAME}.test.inter', sep='\t', index=False)

parameter_dict = {
    'data_path': DATASET_PATH,
    'dataset': DATASET_NAME,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'load_col': {
        'inter': ['user_id', 'item_id', 'rating']
    },
    'eval_setting': 'RO',
    'train_neg_sample_args': None,
    'model': MODEL_NAME,

    'metrics': ['NDCG'],     #AUS#HR #PSP
    'topk': [10, 100],
    'valid_metric': 'NDCG@10', 
    'metric_decimal_place': 4
}

run_recbole(config_dict=parameter_dict)