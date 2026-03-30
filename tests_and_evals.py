"""
evaluate_models.py
==================
Step 5 — Comprehensive metrics evaluation for all IOH prediction models.

Computes for every model:
    1. AUROC with 95% CI (bootstrap)
    2. AUPRC with 95% CI (bootstrap)
    3. Optimal threshold via Youden's J on M1 validation set
    4. Sensitivity, Specificity, PPV, NPV, F1 at that threshold
    5. Mean lead time in minutes for true positive predictions
    6. False alarm rate per hour of surgery
    7. DeLong test between model pairs with Bonferroni correction

Assumptions:
    - All model classes are imported at the top.
    - Checkpoint paths are defined in MODEL_CONFIGS.
    - Test set is fixed at 698 patients, 92,751 windows.
    - Sampling rate is 0.5 Hz (one sample every 2 seconds).
    - Each window is 900 samples = 30 minutes.
    - Lead gap is 150 samples = 5 minutes.
    - Prediction window is 300 samples = 10 minutes.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from model_mamba import IOHMambaPredictor 
from model_transformer import IOHTransformer
from data_preprocessing import IOHDataset


# SECTION 1 — CONFIGURATION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR   = "/mnt/dataforioh/processed_data_denovo/train"   # confirm path
TEST_DIR    = "/mnt/dataforioh/processed_data_denovo/test"    # confirm path
OUTPUT_DIR  = "/mnt/dataforioh/processed_data_denovo"         # confirm path

# Sampling and windowing constants — must match preprocessing pipeline exactly
SAMPLING_HZ         = 0.5        # 0.5 Hz = one sample every 2 seconds
SAMPLES_PER_MIN     = SAMPLING_HZ * 60   # 30 samples per minute
OBS_WINDOW_SAMPLES  = 900        # 30 minutes
LEAD_GAP_SAMPLES    = 150        # 5 minutes
PRED_WINDOW_SAMPLES = 300        # 10 minutes
STEP_SIZE_SAMPLES   = 30         # 1 minute slide step

# Bootstrap settings
N_BOOTSTRAP         = 1000
BOOTSTRAP_SEED      = 42
CI_ALPHA            = 0.05       # 95% confidence interval

# Random guessing AUPRC baseline — positive prevalence on test set
RANDOM_AUPRC_BASELINE = 0.0657 

# Validation split seed — must match training scripts exactly
VAL_SPLIT_SEED = 42


MODEL_CONFIGS = {
    "M1_Mamba_D1": {
        "label":        "Mamba — Full signals (D1)",
        "input_dim":    4,
        "no_covariates": False,
        "drug_only":    False,
        "vitals_only":  False,
        "transformer": True,
        "checkpoints": [
            "",   # seed 42
            "",   # seed 123
            "",   # seed 7
        ],
    },
    "M2_Mamba_D2": {
        "label":        "Mamba — Vitals only (D2)",
        "input_dim":    2,
        "no_covariates": False,
        "drug_only":    False,
        "vitals_only":  True,
        "transformer": True,
        "checkpoints": [
            "",   
            "",   
            "",   
        ],
    },
    "M4_Mamba_D4": {
        "label":        "Mamba — Contaminated (D4)",
        "input_dim":    4,
        "no_covariates": False,
        "drug_only":    False,
        "vitals_only":  False,
        "transformer": True,
        "checkpoints": [
            "",   
            "",   
            "",   
        ],
    },
    "M5_Mamba_D5": {
        "label":        "Mamba — No covariates (D5)",
        "input_dim":    4,
        "no_covariates": True,
        "drug_only":    False,
        "vitals_only":  False,
        "transformer": True,
        "checkpoints": [
            "",   
            "",   
            "",   
        ],
    },
    "M6_Mamba_D6": {
        "label":        "Mamba — Drugs only (D6)",
        "input_dim":    2,
        "no_covariates": False,
        "drug_only":    True,
        "vitals_only":  False,
        "transformer": True,
        "checkpoints": [
            "",   
            "",   
            "",   
        ],
    },
    "T1_Transformer_D1": {
        "label":        "Transformer — Full signals (D1)",
        "input_dim":    4,
        "no_covariates": False,
        "drug_only":    False,
        "vitals_only":  False,
        "transformer": True,
        "checkpoints": [
            "",   
            "",   
            "",   
        ],
    },
}

# DeLong comparison pairs
DELONG_PAIRS = [
    ("M1_Mamba_D1",      "M2_Mamba_D2",      "M1 vs M2: effect of drug CE"),
    ("M1_Mamba_D1",      "M6_Mamba_D6",      "M1 vs M6: effect of vitals"),
    ("M1_Mamba_D1",      "M5_Mamba_D5",      "M1 vs M5: effect of covariates"),
    ("M1_Mamba_D1",      "M4_Mamba_D4",      "M1 vs M4: contamination effect"),
    ("M1_Mamba_D1",      "T1_Transformer_D1","M1 vs T1: Mamba vs Transformer"),
]


# SECTION 2 — DATA LOADING

def build_loaders(batch_size=256):
    """
    Build validation loader (from training manifest) and test loader.
    Validation loader is used exclusively for threshold selection on M1.
    Test loader is used for all reported metrics.
    """
    meta_path = os.path.join(OUTPUT_DIR, "pipeline_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    full_train_manifest = meta["manifest"]["train"]
    test_manifest       = meta["manifest"]["test"]

    train_case_ids = list(full_train_manifest.keys())
    _, val_ids = train_test_split(
        train_case_ids, test_size=0.2, random_state=VAL_SPLIT_SEED
    )
    val_manifest = {cid: full_train_manifest[cid] for cid in val_ids}

    val_ds   = IOHDataset(TRAIN_DIR, val_manifest)
    test_ds  = IOHDataset(TEST_DIR,  test_manifest)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"[INFO] Val  windows : {len(val_ds)}")
    print(f"[INFO] Test windows : {len(test_ds)}")

    return val_loader, test_loader


# SECTION 3 — MODEL INFERENCE

def build_model(config):
    """Instantiate a fresh model. Extend this if Transformer needs different args."""
    if config["transformer"]:
        return IOHPredictor(input_dim=config["input_dim"], model_dim_1=32, model_dim_2=64, num_heads=4)
    else:
        return IOHMambaPredictor(input_dim=config["input_dim"], model_dim_1=32, model_dim_2=64)


def run_inference(model, loader, config):
    """
    Run inference on a dataloader and return (probabilities, labels) as numpy arrays.
    Handles vitals_only, drug_only, and no_covariates ablation variants.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for X_seq, X_static, labels in tqdm(loader):
            X_seq    = X_seq.to(DEVICE)
            labels   = labels.float().to(DEVICE)

            if config["vitals_only"]:
                # D2: keep only MAP and HR (channels 0 and 1)
                X_seq = X_seq[:, :, :2]
            elif config["drug_only"]:
                # D6: keep only Propofol CE and Remifentanil CE (channels 2 and 3)
                X_seq = X_seq[:, :, 2:]

            if config["no_covariates"]:
                # D5: zero out covariates — model sees no static information
                X_static = torch.zeros(X_seq.size(0), 5).to(DEVICE)
            else:
                X_static = X_static.to(DEVICE)

            logits = model(X_seq, X_static)
            probs  = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def get_mean_probs_across_seeds(config, loader):
    """
    Load each seed's checkpoint, run inference, and return:
        - per_seed_probs : list of 3 numpy arrays, one per seed
        - labels         : numpy array (identical across seeds, taken from seed 0)
    """
    per_seed_probs = []
    labels_out     = None

    for ckpt_path in config["checkpoints"]:
        assert ckpt_path != "", (
            f"Checkpoint path not filled in for {config['label']}. "
            f"Please fill in MODEL_CONFIGS."
        )
        model = build_model(config)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)

        probs, labels = run_inference(model, loader, config)
        per_seed_probs.append(probs)

        if labels_out is None:
            labels_out = labels

    return per_seed_probs, labels_out


# SECTION 4 — BOOTSTRAP CI

def bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Compute a metric and its 95% CI via stratified bootstrap resampling.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_score)

    bootstrapped = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        y_true_b  = y_true[idx]
        y_score_b = y_score[idx]

        # Skip degenerate bootstrap samples with only one class
        if len(np.unique(y_true_b)) < 2:
            continue
        bootstrapped.append(metric_fn(y_true_b, y_score_b))

    bootstrapped = np.array(bootstrapped)
    ci_lower = np.percentile(bootstrapped, 100 * CI_ALPHA / 2)
    ci_upper = np.percentile(bootstrapped, 100 * (1 - CI_ALPHA / 2))

    return point, ci_lower, ci_upper


# SECTION 5 — THRESHOLD SELECTION (M1 VALIDATION SET)

def select_threshold_youden(y_true, y_score):
    """
    Select the optimal decision threshold using Youden's J statistic.
    Youden's J = Sensitivity + Specificity - 1.
    Maximising J balances sensitivity and specificity optimally.

    Returns
    -------
    float : optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    j_scores    = tpr + specificity - 1
    best_idx    = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    print(f"[INFO] Youden's J optimal threshold: {optimal_threshold:.4f} "
          f"(Sensitivity={tpr[best_idx]:.3f}, Specificity={specificity[best_idx]:.3f})")
    return float(optimal_threshold)


def compute_threshold_metrics(y_true, y_score, threshold):
    """
    Compute Sensitivity, Specificity, PPV, NPV, F1 at a fixed threshold.

    Returns
    -------
    dict with keys: sensitivity, specificity, ppv, npv, f1
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, zero_division=0)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv":         ppv,
        "npv":         npv,
        "f1":          f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

# SECTION 6 — CLINICAL UTILITY METRICS
def compute_lead_time(y_true, y_score, threshold):
    
    y_pred = (y_score >= threshold).astype(int)
    tp_mask = (y_true == 1) & (y_pred == 1)
    n_tp = int(tp_mask.sum())

    if n_tp == 0:
        return {"mean_lead_time_min": 0.0, "std_lead_time_min": 0.0, "n_true_positives": 0}

    LEAD_GAP_MIN    = LEAD_GAP_SAMPLES    / SAMPLES_PER_MIN   # 5.0 min
    PRED_WINDOW_MIN = PRED_WINDOW_SAMPLES / SAMPLES_PER_MIN   # 10.0 min

    # Uniform distribution over [5, 15] minutes → mean = 10 min, std = sqrt((10²)/12) ≈ 2.89 min
    mean_lead = LEAD_GAP_MIN + PRED_WINDOW_MIN / 2.0
    std_lead  = PRED_WINDOW_MIN / np.sqrt(12)

    print(f"[NOTE] Lead time is estimated from pipeline parameters (uniform distribution "
          f"over [{LEAD_GAP_MIN:.0f}, {LEAD_GAP_MIN + PRED_WINDOW_MIN:.0f}] min). "
          f"For exact per-window lead times, store IOH onset offsets during preprocessing.")

    return {
        "mean_lead_time_min": mean_lead,
        "std_lead_time_min":  std_lead,
        "n_true_positives":   n_tp,
    }


def compute_false_alarm_rate(y_true, y_score, threshold, total_surgery_hours):
    """
    Compute false alarm rate as number of false positive predictions per hour of surgery.

    Parameters
    ----------
    y_true              : numpy array of true labels
    y_score             : numpy array of predicted probabilities
    threshold           : decision threshold
    total_surgery_hours : total hours of surgery covered by the test set
                          Computed as: n_windows * step_size_seconds / 3600

    Returns
    -------
    dict with keys: false_alarms_per_hour, n_false_positives, total_hours
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    false_alarms_per_hour = fp / total_surgery_hours if total_surgery_hours > 0 else 0.0

    return {
        "false_alarms_per_hour": false_alarms_per_hour,
        "n_false_positives":     int(fp),
        "total_surgery_hours":   total_surgery_hours,
    }


def estimate_total_surgery_hours(n_test_windows):
    """
    Estimate total surgery hours covered by the test set.

    Each window slides by STEP_SIZE_SAMPLES at SAMPLING_HZ.
    Total time = n_windows * step_size_in_seconds / 3600.

    This is an upper bound — it counts overlapping windows once each.
    The true unique coverage is less due to sliding window overlap,
    but this is the standard convention for false alarm rate reporting.
    """
    step_size_seconds = STEP_SIZE_SAMPLES / SAMPLING_HZ   # 60 seconds = 1 minute
    total_seconds     = n_test_windows * step_size_seconds
    return total_seconds / 3600.0


# SECTION 7 — DELONG TEST

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance using the DeLong method.
    Based on: DeLong et al. (1988) Biometrics 44:837-845.
    Implementation follows Sun & Xu (2014) Academic Radiology.
    """
    order      = (-predictions).argsort()
    label_1_count = int(ground_truth.sum())
    label_0_count = len(ground_truth) - label_1_count

    predictions_sorted_transposed = predictions[np.newaxis, order]
    ground_truth_sorted           = ground_truth[order]

    aucs, delongcov = _fastDeLong(predictions_sorted_transposed, label_1_count)
    return aucs, delongcov


def _fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01) if v01.shape[1] > 1 else np.array([[np.var(v01)]])
    sy = np.cov(v10) if v10.shape[1] > 1 else np.array([[np.var(v10)]])

    delongcov = sx / m + sy / n
    return aucs, delongcov


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_test(y_true, y_score_a, y_score_b):
    """
    Two-sided DeLong test comparing AUROC of two models on the same test set.

    Returns
    -------
    (auc_a, auc_b, z_stat, p_value)
    """
    m = int(y_true.sum())
    n = len(y_true) - m
    order = (-y_true).argsort()

    y_true_sorted = y_true[order]

    score_ab = np.vstack([y_score_a[order], y_score_b[order]])
    aucs, cov = _fastDeLong(score_ab, m)

    auc_a, auc_b = aucs
    var_a  = cov[0, 0]
    var_b  = cov[1, 1]
    cov_ab = cov[0, 1]

    se    = np.sqrt(var_a + var_b - 2 * cov_ab)
    z     = (auc_a - auc_b) / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(auc_a), float(auc_b), float(z), float(p_val)


# SECTION 8 — RESULTS FORMATTING

def print_separator(char="=", width=80):
    print(char * width)


def print_model_results(model_key, config, results):
    print_separator()
    print(f"  {config['label']}")
    print_separator()

    r = results
    print(f"  AUROC : {r['auroc']:.4f}  (95% CI: {r['auroc_ci_lo']:.4f} – {r['auroc_ci_hi']:.4f})")
    print(f"  AUPRC : {r['auprc']:.4f}  (95% CI: {r['auprc_ci_lo']:.4f} – {r['auprc_ci_hi']:.4f})")
    print(f"  Random AUPRC baseline: {RANDOM_AUPRC_BASELINE:.4f}")
    print(f"  AUPRC lift over random: {r['auprc'] / RANDOM_AUPRC_BASELINE:.2f}x")
    print()
    print(f"  Threshold metrics (threshold = {r['threshold']:.4f})")
    print(f"  Sensitivity : {r['sensitivity']:.4f}")
    print(f"  Specificity : {r['specificity']:.4f}")
    print(f"  PPV         : {r['ppv']:.4f}")
    print(f"  NPV         : {r['npv']:.4f}")
    print(f"  F1-Score    : {r['f1']:.4f}")
    print(f"  TP={r['tp']}  TN={r['tn']}  FP={r['fp']}  FN={r['fn']}")
    print()
    print(f"  Clinical utility")
    print(f"  True positive predictions   : {r['n_true_positives']}")
    print(f"  Mean lead time              : {r['mean_lead_time_min']:.1f} ± {r['std_lead_time_min']:.1f} min")
    print(f"  False alarm rate            : {r['false_alarms_per_hour']:.2f} alarms/hour")
    print(f"  Total surgery hours (test)  : {r['total_surgery_hours']:.1f} h")
    print()

    # Per-seed breakdown
    print(f"  Per-seed AUROC / AUPRC")
    for i, (au, ap) in enumerate(zip(r["seed_aurocs"], r["seed_auprcs"])):
        print(f"  Seed {i+1}: AUROC={au:.4f}  AUPRC={ap:.4f}")
    print(f"  Mean AUROC: {np.mean(r['seed_aurocs']):.4f} ± {np.std(r['seed_aurocs']):.4f}")
    print(f"  Mean AUPRC: {np.mean(r['seed_auprcs']):.4f} ± {np.std(r['seed_auprcs']):.4f}")


def print_delong_results(delong_results, n_comparisons):
    print_separator()
    print("  DeLong Test Results")
    print(f"  Bonferroni-corrected significance threshold: "
          f"p < {CI_ALPHA / n_comparisons:.4f} (α={CI_ALPHA}, n={n_comparisons})")
    print_separator()

    for desc, auc_a, auc_b, z, p, p_corrected, significant in delong_results:
        sig_marker = "***" if significant else "n.s."
        print(f"  {desc}")
        print(f"    AUROC A={auc_a:.4f}  AUROC B={auc_b:.4f}  "
              f"z={z:.3f}  p={p:.4f}  p_bonf={p_corrected:.4f}  {sig_marker}")
        print()


# SECTION 9 — MAIN PIPELINE

def main():
    print_separator()
    print("  IOH Prediction - Step 5 Metrics Evaluation")
    print_separator()

    val_loader, test_loader = build_loaders()
    n_test_windows      = sum(1 for _ in test_loader.dataset)
    total_surgery_hours = estimate_total_surgery_hours(n_test_windows)
    print(f"[INFO] Test windows: {n_test_windows}")
    print(f"[INFO] Estimated total surgery hours: {total_surgery_hours:.1f} h\n")

    # Step A: Get M1 validation probabilities for threshold selection
    print("[STEP A] Selecting threshold from M1 validation set using Youden's J...")
    m1_config = MODEL_CONFIGS["M1_Mamba_D1"]

    # Use the first seed checkpoint for threshold selection
    model_m1 = build_model(m1_config["input_dim"])
    model_m1.load_state_dict(
        torch.load(m1_config["checkpoints"][0], map_location=DEVICE, weights_only=True)
    )
    model_m1.to(DEVICE)
    val_probs_m1, val_labels_m1 = run_inference(model_m1, val_loader, m1_config)
    GLOBAL_THRESHOLD = select_threshold_youden(val_labels_m1, val_probs_m1)
    print(f"[INFO] Global threshold fixed at: {GLOBAL_THRESHOLD:.4f}\n")

    # Step B: Evaluate all models on test set
    all_results = {}
    # Store mean probs per model for DeLong test
    mean_probs_store = {}
    test_labels_store = None

    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n[EVALUATING] {config['label']} ...")

        per_seed_probs, test_labels = get_mean_probs_across_seeds(config, test_loader)

        if test_labels_store is None:
            test_labels_store = test_labels

        # Mean probability across seeds — used for bootstrap CI and DeLong
        mean_probs  = np.mean(per_seed_probs, axis=0)
        mean_probs_store[model_key] = mean_probs

        # Per-seed individual metrics
        seed_aurocs = [roc_auc_score(test_labels, p) for p in per_seed_probs]
        seed_auprcs = [average_precision_score(test_labels, p) for p in per_seed_probs]

        # Bootstrap CI on mean probs
        auroc, auroc_lo, auroc_hi = bootstrap_metric(
            test_labels, mean_probs, roc_auc_score
        )
        auprc, auprc_lo, auprc_hi = bootstrap_metric(
            test_labels, mean_probs, average_precision_score
        )

        # Threshold-based metrics
        tm = compute_threshold_metrics(test_labels, mean_probs, GLOBAL_THRESHOLD)

        # Clinical utility
        lt = compute_lead_time(test_labels, mean_probs, GLOBAL_THRESHOLD)
        fa = compute_false_alarm_rate(
            test_labels, mean_probs, GLOBAL_THRESHOLD, total_surgery_hours
        )

        all_results[model_key] = {
            # Discrimination
            "auroc":          auroc,
            "auroc_ci_lo":    auroc_lo,
            "auroc_ci_hi":    auroc_hi,
            "auprc":          auprc,
            "auprc_ci_lo":    auprc_lo,
            "auprc_ci_hi":    auprc_hi,
            # Threshold
            "threshold":      GLOBAL_THRESHOLD,
            **tm,
            # Clinical utility
            "mean_lead_time_min":  lt["mean_lead_time_min"],
            "std_lead_time_min":   lt["std_lead_time_min"],
            "n_true_positives":    lt["n_true_positives"],
            "false_alarms_per_hour": fa["false_alarms_per_hour"],
            "n_false_positives":   fa["n_false_positives"],
            "total_surgery_hours": fa["total_surgery_hours"],
            # Per-seed
            "seed_aurocs": seed_aurocs,
            "seed_auprcs": seed_auprcs,
        }

    # Step C: Print all model results
    print("\n\n")
    print_separator("═")
    print("  FULL RESULTS")
    print_separator("═")
    for model_key, config in MODEL_CONFIGS.items():
        print_model_results(model_key, config, all_results[model_key])

    # Step D: DeLong tests
    print("\n")
    n_comparisons = len(DELONG_PAIRS)
    bonferroni_threshold = CI_ALPHA / n_comparisons

    delong_results = []
    for key_a, key_b, desc in DELONG_PAIRS:
        probs_a = mean_probs_store[key_a]
        probs_b = mean_probs_store[key_b]
        auc_a, auc_b, z, p = delong_test(test_labels_store, probs_a, probs_b)
        p_corrected = min(p * n_comparisons, 1.0)   # Bonferroni correction
        significant = p_corrected < CI_ALPHA
        delong_results.append((desc, auc_a, auc_b, z, p, p_corrected, significant))

    print_delong_results(delong_results, n_comparisons)

    # Step E: Summary table
    print_separator("═")
    print("  SUMMARY TABLE (for paper)")
    print_separator("═")
    header = f"{'Model':<35} {'AUROC':>12} {'AUPRC':>12} {'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    for model_key, config in MODEL_CONFIGS.items():
        r = all_results[model_key]
        row = (
            f"{config['label']:<35} "
            f"{r['auroc']:.4f} ({r['auroc_ci_lo']:.3f}–{r['auroc_ci_hi']:.3f})  "
            f"{r['auprc']:.4f} ({r['auprc_ci_lo']:.3f}–{r['auprc_ci_hi']:.3f})  "
            f"{r['sensitivity']:.3f}  "
            f"{r['specificity']:.3f}  "
            f"{r['ppv']:.3f}  "
            f"{r['npv']:.3f}  "
            f"{r['f1']:.3f}"
        )
        print(row)
    print(f"\nRandom guessing AUPRC baseline: {RANDOM_AUPRC_BASELINE:.4f}")
    print(f"Global threshold (Youden's J on M1 val set): {GLOBAL_THRESHOLD:.4f}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Bonferroni threshold: p < {bonferroni_threshold:.4f} for {n_comparisons} comparisons")

    return all_results


# ENTRY POINT

if __name__ == "__main__":
    results = main()