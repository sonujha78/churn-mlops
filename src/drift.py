import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI) for a single numeric feature.
    expected = training distribution (reference)
    actual   = live/production distribution (current)
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    # Adapt bucket count to sample size: too many buckets with few live
    # samples makes PSI noisy/unstable (each bucket needs enough samples
    # to be statistically meaningful). Rule of thumb: >= 5 samples/bucket.
    max_buckets_by_sample = max(1, len(actual) // 5)
    effective_buckets = max(2, min(buckets, max_buckets_by_sample))

    breakpoints = np.unique(
        np.quantile(expected, np.linspace(0, 1, effective_buckets + 1))
    )
    if len(breakpoints) < 3:
        return 0.0

    actual_clipped = np.clip(actual, breakpoints[0], breakpoints[-1])

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_clipped, bins=breakpoints)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def calculate_feature_drift(training_df: pd.DataFrame, live_df: pd.DataFrame, numeric_cols):
    scores = {}
    for col in numeric_cols:
        if col in training_df.columns and col in live_df.columns and len(live_df) > 0:
            scores[col] = calculate_psi(training_df[col], live_df[col])
        else:
            scores[col] = 0.0

    overall = float(np.mean(list(scores.values()))) if scores else 0.0
    return scores, overall
