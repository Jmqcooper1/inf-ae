import numpy as np
import math
from collections import Counter

class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        print(f"[GINI] Computing Gini coefficient for {len(values)} values")
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            print("[GINI] Sum of values is 0, returning 0.0")
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        print(f"[GINI] Computed Gini coefficient: {gini:.4f}")
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        print(f"[GINI] Calculating Gini for {len(articles)} articles using key '{key}'")
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        print(f"[GINI] Found {len(freqs)} unique {key} values")
        return self.gini_coefficient(list(freqs.values()))


def compute_shannon_entropy(user_recommendations, total_users):
    item_counts = Counter()
    for recs in user_recommendations.values():
        for rec in recs:
            item_counts[rec["id"]] += 1

    entropy = 0.0
    for count in item_counts.values():
        p = count / total_users
        entropy -= p * math.log2(p)

    max_entropy = math.log2(len(item_counts)) if len(item_counts) > 0 else 1.0
    return entropy, entropy / max_entropy 