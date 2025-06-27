import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import math
from numba import jit, float64

def compute_shannon_entropy(user_recommendations):
    item_counts = Counter()
    for recs in user_recommendations.values():
        for rec in recs:
            item_counts[rec["id"]] += 1

    total_recommendations = sum(item_counts.values())

    entropy = 0.0
    for count in item_counts.values():
        p = count / total_recommendations
        entropy -= p * math.log2(p)

    max_entropy = math.log2(len(item_counts)) if len(item_counts) > 0 else 1.0
    return entropy, entropy / max_entropy


def compute_unfairness_gap(per_user_ndcg, user_groups, k):
    active_scores = per_user_ndcg[k][user_groups["active"]]
    inactive_scores = per_user_ndcg[k][user_groups["inactive"]]
    return abs(np.mean(active_scores) - np.mean(inactive_scores))

    
@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    idx = np.argsort(y_prob)
    sorted_y_true = y_true[idx]
    nfalse, auc = 0, 0
    for i in range(len(sorted_y_true)):
        nfalse += 1 - sorted_y_true[i]
        auc += sorted_y_true[i] * nfalse
    return auc / (nfalse * (len(sorted_y_true) - nfalse))

def get_item_count_map_from_file(train_file):
    df_train = pd.read_csv(train_file, sep="\t")
    num_items = df_train["item_id"].max() + 1
    item_count = defaultdict(int)
    for item in df_train["item_id"]:
        item_count[item] += 1
    return item_count, num_items

def get_item_propensity_from_file(train_file, A=0.55, B=1.5):
    item_count_map, num_items = get_item_count_map_from_file(train_file)
    
    item_freq = [item_count_map[i] if i in item_count_map else 0 for i in range(num_items)]
    num_interactions = sum(item_freq)

    C = (np.log(num_interactions) - 1) * np.power(B + 1, A)
    wts = 1.0 + C * np.power(np.array(item_freq) + B, -A)
    return {i: wts[i] for i in range(num_items)}


def evaluate_run_file(run_file, ground_truth, item_propensity_dict, topk=[10, 100]):
    df_run = pd.read_csv(run_file, sep="\t")
    df_truth = pd.read_csv(ground_truth, sep="\t")
    item_columns = [col for col in df_run.columns if col != 'user_id']

    recommendations = {
        row['user_id']: [row[col] for col in item_columns]
        for _, row in df_run.iterrows()
    }

    user_logits = {
        user: {item: 1.0 / (rank + 1) for rank, item in enumerate(recs)}
        for user, recs in recommendations.items()
    }

    ground_truths = df_truth.groupby("user_id").item_id.apply(set).to_dict()
    all_items = set(df_truth["item_id"].unique()).union(*recommendations.values())

    user_interaction_counts = df_truth.groupby("user_id").size().sort_values(ascending=False)
    sorted_user_ids = user_interaction_counts.index.tolist()
    num_active = max(1, int(0.2 * len(user_interaction_counts)))
    active_user_ids = set(sorted_user_ids[:num_active])
    inactive_user_ids = set(sorted_user_ids[num_active:])

    user_id_list = list(recommendations.keys())
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_id_list)}
    user_groups = {
        "active": [user_id_to_index[uid] for uid in user_id_list if uid in active_user_ids],
        "inactive": [user_id_to_index[uid] for uid in user_id_list if uid in inactive_user_ids],
    }

    metrics = {}
    per_user_ndcg = {k: np.zeros(len(recommendations)) for k in topk}

    for k in topk:
        hits, ndcgs, psps = [], [], []
        y_true_all, y_score_all = [], []
        user_recommendations = {}

        for i, (user, recs) in enumerate(recommendations.items()):
            true_items = ground_truths.get(user, set())
            recs_k = recs[:k]
            hits.append(any(item in true_items for item in recs_k))
            dcg = 0.0
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(true_items), k)))
            for idx, item in enumerate(recs_k):
                if item in true_items:
                    dcg += 1.0 / math.log2(idx + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
            per_user_ndcg[k][i] = ndcg

            if item_propensity_dict:
                num_pos = len(true_items)
                norm = min(k, num_pos)

                psp = 0.0
                for item in recs_k:
                    if item in true_items:
                        psp += item_propensity_dict.get(item, 0.0) / norm

                sorted_true_propensities = sorted(
                    [item_propensity_dict.get(item, 0.0) for item in true_items],
                    reverse=True
                )
                max_psp = sum(sorted_true_propensities[:norm]) / norm if sorted_true_propensities else 1.0

                psps.append(psp / max_psp if max_psp > 0 else 0.0)

            eval_items = list(set(recs[:k]) | true_items) 
            for item in eval_items:
                label = 1.0 if item in true_items else 0.0
                score = user_logits[user].get(item, 0.0)
                y_score_all.append(score)
                y_true_all.append(label)

            user_recommendations[user] = [{"id": item} for item in recs_k]

        metrics[f"HR@{k}"] = round(np.mean(hits), 4)
        metrics[f"NDCG@{k}"] = round(np.mean(ndcgs), 4)
        metrics[f"UnfairnessGap@{k}"] = round(compute_unfairness_gap(per_user_ndcg, user_groups, k), 4)

        entropy, norm_entropy = compute_shannon_entropy(user_recommendations)
        metrics[f"ShannonEntropy@{k}"] = round(norm_entropy, 4)

        if item_propensity_dict:
            metrics[f"PSP@{k}"] = round(np.mean(psps), 4)

    metrics["AUC"] = round(fast_auc(np.array(y_true_all), np.array(y_score_all)), 4)

    return metrics

def main():
    baselines = ['LightGCN', 'EASE', 'MultiVAE']
    datasets = ['magazine', 'douban', 'ml-1m', 'ml-20m', 'netflix', 'steam']

    for baseline in baselines:
        for dataset in datasets:
            MODEL_NAME = baseline
            DATASET_NAME = dataset
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
            DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', DATASET_NAME))

            run_output_path = os.path.join(DATASET_PATH, f'{MODEL_NAME}_{DATASET_NAME}_run.tsv')
            ground_truth_path = os.path.join(DATASET_PATH, f'{MODEL_NAME}_{DATASET_NAME}_ground_truth.tsv')
            train_path = os.path.join(DATASET_PATH, f'{MODEL_NAME}_{DATASET_NAME}_train_file.tsv')

            if not os.path.exists(run_output_path):
                print(f"[ERROR] Run file not found at: {run_output_path}")
                return
            if not os.path.exists(ground_truth_path):
                print(f"[ERROR] Ground truth file not found at: {ground_truth_path}")
                return
            if not os.path.exists(train_path):
                print(f"[ERROR] Train file not found at: {train_path}")
                return

            print(f"[INFO] Evaluating run file: {run_output_path}")
            print(f"[INFO] Using ground truth: {ground_truth_path}")
            print(f"[INFO] Using train file: {train_path}")
            
            item_propensity_dict = get_item_propensity_from_file(train_path)
            metrics = evaluate_run_file(run_output_path, ground_truth_path, item_propensity_dict, topk=[10, 100])

            print("\n[RESULTS]")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            print("")

if __name__ == "__main__":
    main()