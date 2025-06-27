import pandas as pd
import numpy as np
import os
import h5py
import argparse

from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_topk

baselines = ['LightGCN', 'EASE', 'MultiVAE']
datasets = ['douban', 'magazine', 'ml-1m', 'ml-20m', 'netflix', 'steam']

parser = argparse.ArgumentParser(description="Run RecBole model on a dataset.")
parser.add_argument('--model', type=str, choices=baselines, default='MultiVAE', help=f"Model to run. Choices: {baselines}")
parser.add_argument('--dataset', type=str, choices=datasets, default='steam', help=f"Dataset to run. Choices: {datasets}")
parser.add_argument('--use_auc', action='store_true', help="Whether to compute AUC. If set, AUC will be computed.")
args = parser.parse_args()

MODEL_NAME = args.model
DATASET_NAME = args.dataset

print(f'RUNNING on model: {MODEL_NAME}')
print(f'RUNNING on dataset: {DATASET_NAME}')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', DATASET_NAME))

os.makedirs(DATASET_PATH, exist_ok=True)

with h5py.File(os.path.join(DATASET_PATH, "total_data.hdf5"), "r") as f:
    users = f["user"][:]
    items = f["item"][:]
    ratings = f["rating"][:]
df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})

index = np.load(os.path.join(DATASET_PATH, "index.npz"))["data"]
full_df = df[index != -1]

header_with_types = ['user_id:token', 'item_id:token', 'rating:float']
full_df.to_csv(f'{DATASET_PATH}/{DATASET_NAME}.inter', sep='\t', index=False, header=header_with_types)

if MODEL_NAME == 'LightGCN':
    train_neg_sample_args = {
        'distribution': 'popularity',
        'sample_num': 1,           
        'dynamic': False,          
        'candidate_num': 0     
    }
else:
    train_neg_sample_args = None

parameter_dict = {
    'data_path': os.path.dirname(DATASET_PATH), 
    'dataset': DATASET_NAME,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'field_separator': '\t',
    'load_col': {
        'inter': ['user_id', 'item_id', 'rating']
    },
    'train_neg_sample_args': train_neg_sample_args,
    'eval_args': {
        'group_by': 'user', 
        'order': 'RO',
        'split': {'RS': [0.8, 0.1, 0.1]},
        'mode': 'full'
    },
    'model': MODEL_NAME,

    'metrics': ['NDCG', 'Hit', 'ShannonEntropy', 'GiniIndex', 'itemCoverage', 'AveragePopularity', 'TailPercentage'],   
    'topk': [10, 100],
    'valid_metric': 'Hit@100',
    'metric_decimal_place': 4,

    'train_batch_size': 1024,
    'eval_batch_size': 2048,
    'epochs': 100,
    'stopping_step': 5,
    'show_progress': False,
    'workers': 4,
    'eval_step': 2,
    
    'device': 'cuda:0',

    'checkpoint_dir': f'./saved/{MODEL_NAME}/{DATASET_NAME}',
    'save_model': True
}

if MODEL_NAME == 'EASE':
    parameter_dict.pop('epochs', None)
    parameter_dict.pop('train_batch_size', None)
    parameter_dict.pop('eval_batch_size', None)
    parameter_dict.pop('stopping_step', None)
    parameter_dict.pop('eval_step', None)
    
results = run_recbole(config_dict=parameter_dict)

folder = f'./saved/{MODEL_NAME}/{DATASET_NAME}'
files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
files = [f for f in files if f.endswith('.pth')]

if files:
    model_file = max(files, key=os.path.getctime)
    print(f"Latest file: {model_file}")

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)

unique_uids = test_data.dataset.inter_feat['user_id'].unique()

if not isinstance(unique_uids, np.ndarray):
    unique_uids = unique_uids.numpy()

topk_scores, topk_items = full_sort_topk(unique_uids, model, test_data, k=100, device=config['device'])

run_output_path = os.path.join(DATASET_PATH, f'{MODEL_NAME}_{DATASET_NAME}_run.tsv')

with open(run_output_path, 'w') as f:
    header = ['user_id'] + [f'item_{i+1}' for i in range(topk_items.shape[1])]
    f.write("\t".join(header) + "\n")

    for user, items in zip(unique_uids, topk_items.cpu().numpy()):
        f.write(f"{user}\t" + "\t".join(map(str, items)) + "\n")

print(f"Run file saved to: {run_output_path}")

ground_truth_path = os.path.join(DATASET_PATH, f'{MODEL_NAME}_{DATASET_NAME}_ground_truth.tsv')
user_ids = test_data.dataset.inter_feat['user_id'].numpy()
item_ids = test_data.dataset.inter_feat['item_id'].numpy()
ground_truth_df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids})
ground_truth_df.to_csv(ground_truth_path, sep='\t', index=False)
print(f"Ground truth file saved to: {ground_truth_path}")

train_path = os.path.join(DATASET_PATH, f'{MODEL_NAME}_{DATASET_NAME}_train_file.tsv')
train_user_ids = train_data.dataset.inter_feat['user_id'].numpy()
train_item_ids = train_data.dataset.inter_feat['item_id'].numpy()
train_df = pd.DataFrame({'user_id': train_user_ids, 'item_id': train_item_ids})
train_df.to_csv(train_path, sep='\t', index=False)
print(f"Train File saved to: {train_path}")

if args.use_auc:
    parameter_dict = {
        'data_path': os.path.dirname(DATASET_PATH), 
        'dataset': DATASET_NAME,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'field_separator': '\t',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating']
        },
        'train_neg_sample_args': train_neg_sample_args,
        'eval_args': {
            'group_by': 'user', 
            'order': 'RO',
            'split': {'RS': [0.8, 0.1, 0.1]},
            'mode': 'uni100'
        },
        'model': MODEL_NAME,

        'metrics': ['AUC'],              
        'valid_metric': 'AUC', 
        'metric_decimal_place': 4,

        'train_batch_size': 1024,
        'eval_batch_size': 2048,
        'epochs': 100,
        'stopping_step': 5,
        'show_progress': False,
        'workers': 4,
        'eval_step': 2,

        'use_gpu': True,
        'gpu_id': '0'
    }

    if MODEL_NAME == 'EASE':
        parameter_dict.pop('epochs', None)
        parameter_dict.pop('train_batch_size', None)
        parameter_dict.pop('stopping_step', None)
        parameter_dict.pop('eval_step', None)

    run_recbole(config_dict=parameter_dict)