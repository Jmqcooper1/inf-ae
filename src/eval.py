import jax
import numpy as np
import jax.numpy as jnp
from numba import jit, float64

import numpy as np
import math
from collections import Counter

from src.extensions.fairness_diversity.metrics import GiniCoefficient, compute_shannon_entropy

INF = float(1e6)


def evaluate(
    hyper_params,
    kernelized_rr_forward,
    data,
    item_propensity,
    train_x,
    topk=[10, 100],
    test_set_eval=False,
    use_gini=False,
    use_unfairness_gap=False,
):
    print(
        f"\n[EVALUATE] Starting evaluation with topk={topk}, test_set_eval={test_set_eval}"
    )
    print(
        f"[EVALUATE] Hyperparameters: num_users={hyper_params['num_users']}, num_items={hyper_params['num_items']}, lambda={hyper_params['lamda']}"
    )

    preds, y_binary, metrics = [], [], {}
    for kind in ["HR", "NDCG", "PSP", "GINI"]:
        for k in topk:
            metrics["{}@{}".format(kind, k)] = 0.0

    # Train positive set -- these items will be set to -infinity while prediction on the val/test set
    train_positive_list = list(map(list, data.data["train_positive_set"]))
    print(f"[EVALUATE] Train positive set size: {len(train_positive_list)}")

    if test_set_eval:
        print(
            "[EVALUATE] Adding validation positive set to train positive set for test evaluation"
        )
        for u in range(len(train_positive_list)):
            train_positive_list[u] += list(data.data["val_positive_set"][u])

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data["train_matrix"]
    if test_set_eval:
        print("[EVALUATE] Adding validation matrix to evaluation context")
        eval_context += data.data["val_matrix"]

    # What needs to be predicted
    to_predict = data.data["val_positive_set"]
    if test_set_eval:
        print("[EVALUATE] Using test positive set for prediction")
        to_predict = data.data["test_positive_set"]
    print(f"[EVALUATE] Prediction set size: {len(to_predict)}")

    # For GINI calculation - track item exposures across all recommendations
    item_exposures = np.zeros(hyper_params["num_items"])

    user_recommendations = {}

    if use_unfairness_gap:
            # Count how many interactions each user had in training data
            user_interaction_counts = np.array([len(x) for x in train_positive_list])

            # Split users into active / inactive groups
            num_active = max(1, int(0.05 * len(user_interaction_counts)))
            sorted_indices = np.argsort(-user_interaction_counts)

            user_groups = {
            "active": list(sorted_indices[:num_active]),
            "inactive": list(sorted_indices[num_active:])}
            per_user_ndcg = np.zeros(hyper_params["num_users"])

    bsz = 20_000  # These many users
    print(f"[EVALUATE] Processing users in batches of {bsz}")

    for i in range(0, hyper_params["num_users"], bsz):
        batch_end = min(i + bsz, hyper_params["num_users"])
        print(
            f"[EVALUATE] Processing batch of users {i} to {batch_end-1} (total: {batch_end-i})"
        )

        print(f"[EVALUATE] Running forward pass for batch {i} to {batch_end-1}")
        temp_preds = kernelized_rr_forward(
            train_x, eval_context[i:batch_end].todense(), reg=hyper_params["lamda"]
        )
        print(
            f"[EVALUATE] Forward pass complete, prediction shape: {np.array(temp_preds).shape}"
        )

        print(f"[EVALUATE] Evaluating batch {i} to {batch_end-1}")
        metrics, temp_preds, temp_y, user_recommendations_batch = evaluate_batch(
            data.data["negatives"][i:batch_end],
            np.array(temp_preds),
            train_positive_list[i:batch_end],
            to_predict[i:batch_end],
            item_propensity,
            topk,
            metrics,
            data,
            per_user_ndcg = per_user_ndcg[i:batch_end] if use_unfairness_gap else None,
        )
        print(f"[EVALUATE] Batch evaluation complete")

        # Accumulate item exposures for GINI calculation
        for k in topk:
            if k not in user_recommendations:
                user_recommendations[k] = []
            user_recommendations[k] += user_recommendations_batch[k]
            print(
                f"[EVALUATE] Accumulated {len(user_recommendations_batch[k])} recommendations for k={k}"
            )

        preds += temp_preds
        y_binary += temp_y
        print(
            f"[EVALUATE] Accumulated {len(temp_preds)} predictions and {len(temp_y)} labels"
        )

    print(f"[EVALUATE] All batches processed, computing final metrics")
    y_binary, preds = np.array(y_binary), np.array(preds)
    if (True not in np.isnan(y_binary)) and (True not in np.isnan(preds)):
        metrics["AUC"] = round(fast_auc(y_binary, preds), 4)
        print(f"[EVALUATE] Computed AUC: {metrics['AUC']}")
    else:
        print(
            "[EVALUATE] Warning: NaN values detected in y_binary or preds, skipping AUC calculation"
        )

    for kind in ["HR", "NDCG", "PSP"]:
        for k in topk:
            metrics["{}@{}".format(kind, k)] = round(
                float(100.0 * metrics["{}@{}".format(kind, k)])
                / hyper_params["num_users"],
                4,
            )
            print(f"[EVALUATE] {kind}@{k}: {metrics['{}@{}'.format(kind, k)]}")

    for k in topk:
        entropy, normalized_entropy = compute_shannon_entropy(
            {u: [rec] for u, rec in enumerate(user_recommendations[k])},
            hyper_params["num_users"]
        )
        metrics[f"ShannonEntropy@{k}"] = round(normalized_entropy, 4)


    if use_gini:
        print("[EVALUATE] Computing GINI coefficients")
        for k in topk:
            print(
                f"[EVALUATE] Computing GINI@{k} with {len(user_recommendations[k])} recommendations"
            )
            metrics["GINI@{}".format(k)] = GiniCoefficient().calculate_list_gini(
                user_recommendations[k], key="category"
            )
            print(f"[EVALUATE] GINI@{k}: {metrics['GINI@{}'.format(k)]}")


    if use_unfairness_gap:
        active_scores = per_user_ndcg[user_groups["active"]]
        inactive_scores = per_user_ndcg[user_groups["inactive"]]
        unfairness_gap = abs(active_scores.mean() - inactive_scores.mean())
        key = f"UnfairnessGap@{topk[0]}"
        metrics[key] = round(unfairness_gap, 4)
        print(f"[EVALUATE] {key}: {metrics[key]:.4f}"
              f"(Active avg={active_scores.mean():.4f}, Inactive avg={inactive_scores.mean():.4f})")

    metrics["num_users"] = int(train_x.shape[0])
    metrics["num_interactions"] = int(jnp.count_nonzero(train_x.astype(np.int8)))
    print(
        f"[EVALUATE] Final metrics: num_users={metrics['num_users']}, num_interactions={metrics['num_interactions']}"
    )
    
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
    train_metrics=False,
    per_user_ndcg=None,
):
    print(f"[EVAL_BATCH] Starting batch evaluation with {len(logits)} users")

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

    user_recommendations = {}

    for k in topk:
        print(f"[EVAL_BATCH] Computing metrics for k={k}")
        user_recommendations[k] = []
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
            if per_user_ndcg is not None:
                per_user_ndcg[b] = ndcg

            ndcg_sum += ndcg
            psp_sum += psp_norm

            metrics["NDCG@{}".format(k)] += ndcg
            metrics["PSP@{}".format(k)] += psp_norm

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
    return metrics, temp_preds, temp_y, user_recommendations if use_gini else {}


@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    # Note: Can't add prints here because this function is JIT-compiled
    y_true = y_true[np.argsort(y_prob)]
    nfalse, auc = 0, 0
    for i in range(len(y_true)):
        nfalse += 1 - y_true[i]
        auc += y_true[i] * nfalse
    return auc / (nfalse * (len(y_true) - nfalse))
