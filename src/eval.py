import jax
import numpy as np
import jax.numpy as jnp
from numba import jit, float64
from collections import Counter, defaultdict
from random import sample
import math

from hyper_params import hyper_params

USE_GINI = hyper_params["use_gini"]
USE_UNFAIRNESS = hyper_params["use_unfairness_gap"]
USE_EXPOSURE_AWARE_RERANKING = hyper_params["use_exposure_aware_reranking"]
USE_USER_GROUP_FAIRNESS_RERANKING = hyper_params["use_user_group_fairness_reranking"]

def exposure_aware_reranking(logits, exposure_counts, alpha):
    """Adjusts logits by penalizing popular (overexposed) items."""
    adjusted_logits = logits - alpha * (exposure_counts / (np.sum(exposure_counts) + 1e-8))
    return adjusted_logits

def user_group_fairness_reranking(logits, user_idx, user_groups, exposure_counts=None, alpha=1.0):
    """ Penalize or boost logits based on user group """
    group = "active" if user_idx in user_groups["active"] else "inactive"
    
    exposure_penalty = exposure_counts / (np.max(exposure_counts) + 1e-8)

    if group == "inactive":
        adjusted_logits = logits + alpha * (1.0 - exposure_penalty)
    else:
        adjusted_logits = logits - alpha * exposure_penalty * 0.5
    return adjusted_logits

def compute_unfairness_gap(per_user_ndcg, user_groups, k):
    active_scores = per_user_ndcg[k][user_groups["active"]]
    inactive_scores = per_user_ndcg[k][user_groups["inactive"]]
    gap = abs(active_scores.mean() - inactive_scores.mean())
    return gap, active_scores.mean(), inactive_scores.mean()

class GiniCoefficient:
    def gini_coefficient(self, values):
        print(f"[GINI] Computing Gini coefficient for {len(values)} values")
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            print("[GINI] Sum of values is 0, returning 0.0")
            return 0.0
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        print(f"[GINI] Computed Gini coefficient: {gini:.4f}")
        return gini

    def calculate_list_gini(self, articles, key="category"):
        print(f"[GINI] Calculating Gini for {len(articles)} articles using key '{key}'")
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        print(f"[GINI] Found {len(freqs)} unique {key} values")
        return self.gini_coefficient(list(freqs.values()))


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


INF = float(1e6)

def evaluate(
    hyper_params,
    kernelized_rr_forward,
    data,
    item_propensity,
    train_x,
    topk=[10, 100],
    test_set_eval=False,
):
    print(f"\n[EVALUATE] Starting evaluation with topk={topk}, test_set_eval={test_set_eval}")
    print(f"[EVALUATE] Hyperparameters: num_users={hyper_params['num_users']}, num_items={hyper_params['num_items']}, lambda={hyper_params['lamda']}, alpha = {hyper_params['alpha']}")

    preds, y_binary, metrics = [], [], {}
    for kind in ["HR", "NDCG", "PSP", "GINI"]:
        for k in topk:
            metrics[f"{kind}@{k}"] = 0.0

    train_positive_list = list(map(list, data.data["train_positive_set"]))
    if test_set_eval:
        for u in range(len(train_positive_list)):
            train_positive_list[u] += list(data.data["val_positive_set"][u])

    eval_context = data.data["train_matrix"]
    if test_set_eval:
        eval_context += data.data["val_matrix"]

    to_predict = data.data["val_positive_set"]
    if test_set_eval:
        to_predict = data.data["test_positive_set"]

    item_exposures = np.zeros(hyper_params["num_items"])
    user_recommendations = {}

    # Initialize exposures from training data (approximate popularity)
    exposure_counts = np.zeros(hyper_params["num_items"])
    for user_pos_items in train_positive_list:
        for item in user_pos_items:
            exposure_counts[item] += 1

    if USE_UNFAIRNESS:
        user_interaction_counts = np.array([len(x) for x in train_positive_list])
        num_active = max(1, int(0.2 * len(user_interaction_counts)))
        sorted_indices = np.argsort(-user_interaction_counts)
        user_groups = {
            "active": list(sorted_indices[:num_active]),
            "inactive": list(sorted_indices[num_active:])
        }
        data.data["user_groups"] = user_groups
        per_user_ndcg = {k: np.zeros(hyper_params["num_users"]) for k in topk}


    bsz = 20000
    for i in range(0, hyper_params["num_users"], bsz):
        batch_end = min(i + bsz, hyper_params["num_users"])
        temp_preds = kernelized_rr_forward(
            train_x, eval_context[i:batch_end].todense(), reg=hyper_params["lamda"]
        )
        
        per_user_ndcg_batch = {k: per_user_ndcg[k][i:batch_end] for k in topk}  
        metrics, temp_preds, temp_y, user_recommendations_batch = evaluate_batch(
            data.data["negatives"][i:batch_end],
            np.array(temp_preds),
            train_positive_list[i:batch_end],
            to_predict[i:batch_end],
            item_propensity,
            topk,
            metrics,
            data,
            per_user_ndcg=per_user_ndcg_batch,
            exposure_counts=exposure_counts,
            alpha=hyper_params["alpha"],
            start_user_idx=i,
        )

        for k in topk:
            if k not in user_recommendations:
                user_recommendations[k] = []
            user_recommendations[k] += user_recommendations_batch[k]

        preds += temp_preds
        y_binary += temp_y

    y_binary, preds = np.array(y_binary), np.array(preds)
    if not np.any(np.isnan(y_binary)) and not np.any(np.isnan(preds)):
        metrics["AUC"] = round(fast_auc(y_binary, preds), 4)
    else:
        print("[EVALUATE] Warning: NaN values detected in y_binary or preds, skipping AUC calculation")

    for kind in ["HR", "NDCG", "PSP"]:
        for k in topk:
            metrics[f"{kind}@{k}"] = round(metrics[f"{kind}@{k}"] * 100.0 / hyper_params["num_users"], 4)

    for k in topk:
        entropy, normalized_entropy = compute_shannon_entropy(
            {u: [rec] for u, rec in enumerate(user_recommendations[k])}
        )
        metrics[f"ShannonEntropy@{k}"] = round(normalized_entropy, 4)

    if USE_GINI:
        for k in topk:
            metrics[f"GINI@{k}"] = GiniCoefficient().calculate_list_gini(
                user_recommendations[k], key="category"
            )

    if USE_UNFAIRNESS:
        for k in topk:
            alpha = hyper_params["alpha"]
            active_scores = per_user_ndcg[k][user_groups["active"]]
            inactive_scores = per_user_ndcg[k][user_groups["inactive"]]
            unfairness_gap = abs(active_scores.mean() - inactive_scores.mean())
            metrics[f"UnfairnessGap@{k}_alpha={alpha}"] = round(unfairness_gap, 4)
            print(f"UnfairnessGap@{k} with alpha = {alpha}: {unfairness_gap:.4f}")

    # Evaluate random model 
    if hyper_params.get("evaluate_random", False):
        random_recs = generate_random_recommendations(
            train_positive_list,
            hyper_params["num_users"],
            hyper_params["num_items"],
            topk
        )
        for k in topk:
            entropy, norm_entropy = compute_shannon_entropy(
                {u: random_recs[k][u] for u in range(hyper_params["num_users"])}
            )
            metrics[f"Baseline_ShannonEntropy@{k}"] = round(norm_entropy, 4)


    metrics["num_users"] = int(train_x.shape[0])
    metrics["num_interactions"] = int(jnp.count_nonzero(train_x.astype(np.int8)))
    return metrics


def evaluate_batch(
    auc_negatives,
    logits,
    train_positive,
    test_positive_set,
    item_propensity,
    topk,
    metrics,
    data,
    use_gini=USE_GINI,
    train_metrics=False,
    per_user_ndcg=None,
    exposure_counts=None,
    alpha=1.0,
    start_user_idx=0,
):
    print(f"[EVAL_BATCH] Starting batch evaluation with {len(logits)} users")

    # Reranking
    if exposure_counts is not None and USE_EXPOSURE_AWARE_RERANKING:
        print('reranking')
        for b in range(len(logits)):
            user_idx = start_user_idx + b
            logits[b] = exposure_aware_reranking(logits[b], exposure_counts, alpha=alpha)
            
    elif exposure_counts is not None and USE_USER_GROUP_FAIRNESS_RERANKING:
        print('reranking')
        for b in range(len(logits)):
            user_idx = start_user_idx + b
            logits[b] = user_group_fairness_reranking(logits[b], user_idx=user_idx, user_groups=data.data["user_groups"], exposure_counts=exposure_counts, alpha=alpha)

    # AUC Stuff
    temp_preds, temp_y = [], []
    for b in range(len(logits)):
        pos_count = len(test_positive_set[b])
        neg_count = len(auc_negatives[b])

        if pos_count == 0 or neg_count == 0:
            continue

        if b % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(
                f"[EVAL_BATCH] User {b}: processing {pos_count} positive and {neg_count} negative examples"
            )

        temp_preds += np.take(logits[b], np.array(list(test_positive_set[b]))).tolist()
        temp_y += [1.0 for _ in range(pos_count)]

        temp_preds += np.take(logits[b], auc_negatives[b]).tolist()
        temp_y += [0.0 for _ in range(neg_count)]

    print(f"[EVAL_BATCH] Collected {len(temp_preds)} predictions for AUC calculation")

    # Marking train-set consumed items as negative INF
    print(f"[EVAL_BATCH] Marking train-set consumed items as negative infinity")
    for b in range(len(logits)):
        if b % 1000 == 0:  # Only print every 1000 users to avoid excessive output
            print(
                f"[EVAL_BATCH] User {b}: marking {len(train_positive[b])} train positive items as -INF"
            )

        logits[b][train_positive[b]] = -INF

    print(f"[EVAL_BATCH] Sorting indices for top-{max(topk)} recommendations")
    indices = (-logits).argsort()[:, : max(topk)].tolist()

    batch_exposures = {k: np.zeros(logits.shape[1]) for k in topk}

    user_recommendations = {k: [] for k in topk}

    for k in topk:
        print(f"[EVAL_BATCH] Computing metrics for k={k}")
        hr_sum, ndcg_sum, psp_sum = 0, 0, 0

        for b in range(len(logits)):

            if use_gini:
                # Update item exposures for this batch at this k
                for item_idx in indices[b][:k]:
                    user_recommendations[k].append(
                        {
                            "id": item_idx + 1,
                            "category": data.data["item_map_to_category"].get(item_idx + 1, "UNKNOWN"),
                        }
                    )
            else:
                for item_idx in indices[b][:k]:
                    user_recommendations[k].append({"id": item_idx + 1})

            num_pos = float(len(test_positive_set[b]))
            if num_pos == 0:
                continue
            hits = len(set(indices[b][:k]) & test_positive_set[b])

            if b % 1000 == 0:  # Only print every 1000 users to avoid excessive output
                print(
                    f"[EVAL_BATCH] User {b}, k={k}: {hits} hits out of {min(num_pos, k)} possible"
                )

            hr = float(hits) / float(min(num_pos, k))
            hr_sum += hr
            metrics["HR@{}".format(k)] += hr

            test_positive_sorted_psp = sorted(
                [item_propensity[x] for x in test_positive_set[b]]
            )[::-1]

            dcg, idcg, psp, max_psp = 0.0, 0.0, 0.0, 0.0
            for at, pred in enumerate(indices[b][:k]):
                if pred in test_positive_set[b]:
                    dcg += 1.0 / np.log2(at + 2)
                    psp += float(item_propensity[pred]) / float(min(num_pos, k))
                if at < num_pos:
                    idcg += 1.0 / np.log2(at + 2)
                    max_psp += test_positive_sorted_psp[at]

            ndcg = dcg / idcg if idcg > 0 else 0
            psp_norm = psp / max_psp if max_psp > 0 else 0

            # Save users ndcg
            if per_user_ndcg is not None and isinstance(per_user_ndcg, dict):
                per_user_ndcg[k][b] = ndcg

            ndcg_sum += ndcg
            psp_sum += psp_norm

            metrics["NDCG@{}".format(k)] += ndcg
            metrics["PSP@{}".format(k)] += psp_norm

        if exposure_counts is not None:
            k_max = max(topk)
            for b in range(len(indices)):
                for item_idx in indices[b][:k_max]:
                    exposure_counts[item_idx] += 1

        print(
            f"[EVAL_BATCH] k={k} metrics - Average HR: {hr_sum/len(logits):.4f}, Average NDCG: {ndcg_sum/len(logits):.4f}, Average PSP: {psp_sum/len(logits):.4f}"
        )
        if use_gini:
            print(
                f"[EVAL_BATCH] Collected {len(user_recommendations[k])} recommendations for k={k}"
            )

    print(
        f"[EVAL_BATCH] Batch evaluation complete, returning {len(temp_preds)} predictions"
    )
    return metrics, temp_preds, temp_y, user_recommendations 


@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    # Note: Can't add prints here because this function is JIT-compiled
    y_true = y_true[np.argsort(y_prob)]
    nfalse, auc = 0, 0
    for i in range(len(y_true)):
        nfalse += 1 - y_true[i]
        auc += y_true[i] * nfalse
    return auc / (nfalse * (len(y_true) - nfalse))


def generate_random_recommendations(train_positive_list, num_users, num_items, topk):
    user_recommendations = {k: [] for k in topk}
    for user_id in range(num_users):
        interacted_items = set(train_positive_list[user_id])
        all_items = set(range(num_items))
        candidate_items = all_items - interacted_items

        for k in topk:
            if len(candidate_items) < k:
                recs = sample(candidate_items, k)
            else:
                recs = sample(candidate_items, len(candidate_items))
            user_recommendations[k].append([{"id": item} for item in recs])

    return user_recommendations
