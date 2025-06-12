import os
import argparse
import pandas as pd
import numpy as np
import h5py
import yaml
import glob

from recbole.quick_start import run_recbole

def prepare_single_inter_file(dataset_name, base_data_path):
    """
    Ensures that a single {dataset_name}.inter file exists, creating it from HDF5/NPZ
    if necessary. Returns the train/validation/test split ratios.
    """
    dataset_path = os.path.join(base_data_path, dataset_name)
    inter_path = os.path.join(dataset_path, f"{dataset_name}.inter")

    # Clear any stale cache files to prevent issues
    cache_files = glob.glob(os.path.join(dataset_path, f'{dataset_name}-*.dat'))
    if cache_files:
        print(f"Found and deleting stale cache files: {cache_files}")
        for f in cache_files:
            os.remove(f)

    # If the single .inter file exists, we assume it's ready.
    # Note: This path doesn't recalculate ratios. For robustness, one might always regenerate.
    if os.path.exists(inter_path):
        print(f"Found existing .inter file for {dataset_name}. Reading ratios from source.")
    else:
        print(f"No .inter file found for {dataset_name}. Generating from HDF5/NPZ...")

    hdf5_path = os.path.join(dataset_path, "total_data.hdf5")
    npz_path = os.path.join(dataset_path, "index.npz")

    if not (os.path.exists(hdf5_path) and os.path.exists(npz_path)):
        raise FileNotFoundError(f"Source files (total_data.hdf5 or index.npz) not found in {dataset_path}")

    with h5py.File(hdf5_path, "r") as f:
        users = f["user"][:]
        items = f["item"][:]
        ratings = f["rating"][:]
    
    df = pd.DataFrame({"user_id:token": users, "item_id:token": items, "rating:float": ratings})
    
    # Save the complete file if it wasn't there
    if not os.path.exists(inter_path):
        df.to_csv(inter_path, sep='\t', index=False)
        print(f"Saved single interaction file to {inter_path}")

    # Calculate split ratios from the index file
    index = np.load(npz_path)["data"]
    total = len(index)
    train_ratio = np.sum(index == 0) / total
    val_ratio = np.sum(index == 1) / total
    test_ratio = 1.0 - train_ratio - val_ratio # Ensure it sums to 1

    print(f"Calculated splits: Train={train_ratio:.2%}, Valid={val_ratio:.2%}, Test={test_ratio:.2%}")
    return [train_ratio, val_ratio, test_ratio]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='The baseline model to run.')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to run the model on.')
    args = parser.parse_args()

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'data'))
    config_path = os.path.join(script_dir, 'configs')

    try:
        # 1. Prepare data and get split ratios
        split_ratios = prepare_single_inter_file(args.dataset, base_data_path)

        # 2. Load configurations from YAML files
        with open(os.path.join(config_path, 'base.yaml'), 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_yaml_path = os.path.join(config_path, f'{args.model}.yaml')
        if os.path.exists(model_yaml_path):
            with open(model_yaml_path, 'r') as f:
                model_config = yaml.safe_load(f)
                if model_config:
                    config_dict.update(model_config)

        # 3. Build the final config dictionary, now with the crucial eval_args
        config_dict.update({
            'model': args.model,
            'dataset': args.dataset,
            'data_path': base_data_path,
            'eval_args': {
                'split': {'RS': split_ratios},
                'group_by': 'user',
                'order': 'RO',
                'mode': 'full'
            }
        })
        
        run_recbole(config_dict=config_dict)

    except Exception as e:
        print(f"Error running {args.model} on {args.dataset}: {e}")
        print("Continuing to next run...") 