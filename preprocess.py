import numpy as np
import h5py
import sys
import os
import pandas as pd

BASE_PATH = "data/"


def prep_recbole(
    inter_file_path, user_id, item_id, rating_id, item_file_path, item_id_2
):
    """
    Process ML-1M dataset from RecBole format
    RecBole typically stores data in .inter files or in a processed format
    """

    # If the RecBole .inter file exists, process it
    if os.path.exists(inter_file_path):
        # RecBole .inter files are typically tab-separated with headers
        df = pd.read_csv(inter_file_path, sep="\t").dropna()

        # Extract user_id, item_id, and rating columns
        users = df[user_id].values
        items = df[item_id].values
        ratings = df[rating_id].values
    else:
        print("The .inter file was not found under the path:" + inter_file_path)
        # Exit early if no interaction file
        raise Exception(f"Interaction file not found at {inter_file_path}")

    # First, check if the item file exists and load it
    if not item_file_path or not os.path.exists(item_file_path):
        print("⚠️ Skipping item file processing — none provided.")
        item_df = pd.DataFrame({item_id_2: np.unique(items)})
    else:
        print("The .item file was not found under the path:" + item_file_path)
        # If you want to continue without an item file, comment out the next line
        raise Exception(f"Item file not found at {item_file_path}")

    # Convert to zero-based indexing if needed
    min_user = min(users)
    if min_user > 0:
        users = [u - min_user for u in users]

    # Create sequential mapping for users
    unique_users = sorted(set(users))
    map_user = {user: idx for idx, user in enumerate(unique_users)}

    # Create sequential mapping for items - ensuring sequential IDs from 0 to max_item
    # First, get all unique item IDs from both interactions and item file
    all_items = set()
    for item in items:
        item_val = item.item() if hasattr(item, "item") else item
        all_items.add(item_val)

    # Add items from item file if they're not already in the set
    for item in item_df[item_id_2].values:
        if pd.notna(item):  # Skip NaN values
            all_items.add(item)

    # Create sequential mapping for all items
    sorted_items = sorted(all_items)
    map_item = {item: idx for idx, item in enumerate(sorted_items)}

    # Ensure we have a complete sequential range from 0 to max_item
    max_item_id = len(map_item) - 1
    print(f"Created sequential item mapping from 0 to {max_item_id}")

    # Verify the mapping is complete
    if len(map_item) != max_item_id + 1:
        print(
            f"Warning: Item mapping has {len(map_item)} items but max ID is {max_item_id}"
        )
        # Fill in any gaps if needed
        for i in range(max_item_id + 1):
            if i not in map_item.values():
                print(f"Missing item ID in sequence: {i}")

    # Organize data by user
    num_users = len(map_user)
    data = [[] for _ in range(num_users)]

    # Add valid interactions to the data structure
    for i in range(len(users)):
        user_val = users[i].item() if hasattr(users[i], "item") else users[i]
        item_val = items[i].item() if hasattr(items[i], "item") else items[i]
        rating_val = (
            ratings[i].item() if hasattr(ratings[i], "item") else float(ratings[i])
        )

        # Skip if item is not in our mapping (shouldn't happen with our approach)
        if item_val not in map_item:
            print(f"Warning: Item {item_val} not found in mapping, skipping")
            continue

        # Skip if user is not in our mapping (shouldn't happen with our approach)
        if user_val not in map_user:
            print(f"Warning: User {user_val} not found in mapping, skipping")
            continue

        data[map_user[user_val]].append([map_item[item_val], rating_val])

    # Update the item file with new mappings
    if os.path.exists(item_file_path):
        # Create a new column with the mapped IDs
        item_df["mapped_id"] = item_df[item_id_2].apply(
            lambda x: map_item.get(x, float("nan")) if pd.notna(x) else float("nan")
        )

        # Drop rows where mapping failed
        before_count = len(item_df)
        item_df = item_df.dropna(subset=["mapped_id"])
        after_count = len(item_df)

        if before_count != after_count:
            print(f"Dropped {before_count - after_count} items that couldn't be mapped")

        # Replace the original ID column with the mapped IDs
        item_df[item_id_2] = item_df["mapped_id"]
        item_df = item_df.drop(columns=["mapped_id"])

        # Save the updated item file
        item_df.to_csv(
            item_file_path.replace("_original.", ".", 1), sep="\t", index=False
        )
        print(f"Saved updated item file with {len(item_df)} items")
    return rating_data(data)


def prep_movielens(ratings_file_path):
    """Original function to process MovieLens 1M dataset"""
    f = open(ratings_file_path, "r")
    users, items, ratings = [], [], []

    line = f.readline()
    while line:
        u, i, r, _ = line.strip().split("::")
        users.append(int(u))
        items.append(int(i))
        ratings.append(float(r))
        line = f.readline()

    f.close()

    min_user = min(users)
    num_users = len(set(users))

    data = [[] for _ in range(num_users)]
    for i in range(len(users)):
        data[users[i] - min_user].append([items[i], ratings[i]])

    return rating_data(data)


class rating_data:
    def __init__(self, data):
        self.data = data

        self.index = (
            []
        )  # 0: train, 1: validation, 2: test, -1: removed due to user frequency < 3
        for user_data in self.data:
            for _ in range(len(user_data)):
                self.index.append(42)

    def train_test_split(self):
        at = 0

        for user in range(len(self.data)):
            first_split_point = int(0.8 * len(self.data[user]))
            second_split_point = int(0.9 * len(self.data[user]))

            indices = np.arange(len(self.data[user]))
            np.random.shuffle(indices)

            for timestep, (item, rating) in enumerate(self.data[user]):
                if len(self.data[user]) < 3:
                    self.index[at] = -1
                else:
                    # Force at least one element in user history to be in test
                    if timestep == indices[0]:
                        self.index[at] = 2
                    else:
                        if timestep in indices[:first_split_point]:
                            self.index[at] = 0
                        elif timestep in indices[first_split_point:second_split_point]:
                            self.index[at] = 1
                        else:
                            self.index[at] = 2
                at += 1

        assert at == len(self.index)
        self.complete_data_stats = None

    def save_index(self, path):
        os.makedirs(path, exist_ok=True)
        with open(path + "/index.npz", "wb") as f:
            np.savez_compressed(f, data=self.index)

    def save_data(self, path):
        flat_data = []
        for u in range(len(self.data)):
            flat_data += list(map(lambda x: [u] + x, self.data[u]))
        flat_data = np.array(flat_data)

        shape = [len(flat_data)]

        os.makedirs(path, exist_ok=True)
        with h5py.File(path + "/total_data.hdf5", "w") as file:
            dset = {}
            dset["user"] = file.create_dataset(
                "user", shape, dtype="i4", maxshape=shape, compression="gzip"
            )
            dset["item"] = file.create_dataset(
                "item", shape, dtype="i4", maxshape=shape, compression="gzip"
            )
            dset["rating"] = file.create_dataset(
                "rating", shape, dtype="f", maxshape=shape, compression="gzip"
            )

            dset["user"][:] = flat_data[:, 0]
            dset["item"][:] = flat_data[:, 1]
            dset["rating"][:] = flat_data[:, 2]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("This file needs the dataset name as the first argument...")
        exit(0)

    dataset = sys.argv[1]
    print(f"\n\n!!!!!!!! STARTED PROCESSING {dataset} !!!!!!!!")

    if dataset == "ml-1m":
        total_data = prep_movielens(BASE_PATH + "/ml-1m/ratings.dat")

    elif dataset == "steam":
        total_data = prep_recbole(
            BASE_PATH + "/steam/steam.inter",
            "user_id:token",
            "product_id:token",
            "play_hours:float",
            BASE_PATH + "/steam/steam_original.item",
            "id:token",
        )

    elif dataset == "netflix":
        total_data = prep_recbole(
            BASE_PATH + "/netflix/netflix.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            item_file_path="",  # no item file needed
            item_id_2="item_id:token",
        )

    elif dataset == "ml-20m":
        total_data = prep_recbole(
            BASE_PATH + "/ml-20m/ml-20m.inter",
            "user_id:token",
            "item_id:token",
            "rating:float",
            item_file_path="",  # no item file needed
            item_id_2="item_id:token",
        )

    else:
        raise Exception("Could not understand this dataset")

    total_data.save_data(BASE_PATH + f"{dataset}/")
    total_data.train_test_split()
    total_data.save_index(BASE_PATH + f"{dataset}/")
