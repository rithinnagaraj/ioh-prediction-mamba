"""
data_preprocessing.py
=====================
End-to-end pipeline for extracting, preprocessing, windowing, balancing, and
normalizing VitalDB surgical data for **Early-Warning Intraoperative Hypotension
(IOH) Prediction**.

Final output: a PyTorch Dataset & DataLoader that yields batches of
    (X_seq, X_static, Y)
where
    X_seq   : [batch, 900, 4]   — 30-min observation window of 4 time-series
    X_static: [batch, 5]        — 5 static demographic features
    Y       : [batch]           — binary IOH label (0 or 1)

Date   : 2026-03-11
"""

import os
import pickle
import functools
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# VitalDB open-dataset access  (pip install vitaldb)
import vitaldb

# 1. CONFIGURATION CONSTANTS

# Set CASE_IDS to a list of ints to hand-pick cases, otherwise the first
# NUM_CASES cases that have all four tracks will be used.
CASE_IDS: Optional[List[int]] = None
NUM_CASES: int = None  # ignored when CASE_IDS is provided

TRACK_NAMES: List[str] = [
    "Solar8000/ART_MBP",   # Arterial Mean Blood Pressure
    "Solar8000/HR",        # Heart Rate
    "Orchestra/PPF20_CE",  # Propofol effect-site concentration
    "Orchestra/RFTN20_CE", # Remifentanil effect-site concentration
]

DEMOGRAPHIC_COLS: List[str] = ["age", "sex", "height", "weight", "bmi"]

SAMPLING_INTERVAL_SEC: int = 2  # 0.5 Hz (one sample every 2 seconds)

GAP_INTERP_MAX: int = 30    # If < 1 min, then linear interpolation
GAP_FFILL_MAX: int = 150    # If 1–5 min  then forward fill
# If Gaps > GAP_FFILL_MAX (> 5 min) then mark region; windows that overlap are discarded

OBS_WINDOW: int = 900       # 30 min observation
LEAD_GAP: int   = 150       # 5 min gap (ignored)
PRED_WINDOW: int = 300      # 10 min prediction window
STEP_SIZE: int   = 30       # 1 min slide step
TOTAL_SPAN: int  = OBS_WINDOW + LEAD_GAP + PRED_WINDOW  # 1350 samples

IOH_THRESHOLD: float = 65.0   # mmHg (strictly below)
IOH_MIN_CONSECUTIVE: int = 30 # 1 min continuous (at 0.5 Hz)

POS_NEG_RATIO: float = 1 / 3  # 1 positive : 3 negatives

TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "processed_data")
TRAIN_DIR: str = os.path.join(OUTPUT_DIR, "train")
TEST_DIR: str  = os.path.join(OUTPUT_DIR, "test")

BATCH_SIZE: int = 32
NUM_WORKERS: int = 0  # set >0 if on Linux/multi-core

LRU_CACHE_SIZE: int = 8  # keep at most N patient files in memory


# 2. DATA EXTRACTION

def download_clinical_data() -> pd.DataFrame:
    """
    Download the full VitalDB clinical-information table (one row per case).

    The CSV is served at https://api.vitaldb.net/cases and contains 72+
    perioperative columns including age, sex, height, weight, bmi, caseid, etc.

    Returns
    -------
    pd.DataFrame
        Full clinical dataframe indexed by ``caseid``.
    """
    url = "https://api.vitaldb.net/cases"
    print("[INFO] Downloading clinical metadata from VitalDB …")
    df = pd.read_csv(url)
    df = df.set_index("caseid")
    print(f"[INFO] Clinical table loaded: {len(df)} cases, {df.shape[1]} columns")
    return df


def resolve_case_ids(clinical_df: pd.DataFrame) -> List[int]:
    """
    Determine which case IDs to process.

    If ``CASE_IDS`` is set, use those directly.  Otherwise, pick the first
    ``NUM_CASES`` cases from the clinical table.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Full clinical table (indexed by caseid).

    Returns
    -------
    list[int]
        Sorted list of case IDs to process.
    """
    if CASE_IDS is not None:
        ids = sorted(CASE_IDS)
        print(f"[INFO] Using user-specified case IDs: {len(ids)} cases")
        return ids
    else:
        ids = sorted(clinical_df.index.tolist()[:NUM_CASES])
        print(f"[INFO] Selected first {len(ids)} cases from clinical table")
        return ids


def extract_case_tracks(case_id: int) -> Optional[np.ndarray]:
    """
    Load the four time-series tracks for a single case, resampled at 0.5 Hz.

    Uses ``vitaldb.load_case`` which returns a NumPy ndarray of shape
    ``(T, len(TRACK_NAMES))`` where T is the recording length in samples.

    Parameters
    ----------
    case_id : int
        VitalDB case identifier.

    Returns
    -------
    np.ndarray or None
        Shape ``(T, 4)`` array, columns ordered as ``TRACK_NAMES``.
        Returns None if the case cannot be loaded (e.g. missing tracks).
    """
    try:
        # vitaldb.load_case returns a numpy array with shape (T, num_tracks).
        # The `interval` argument controls the resampling period in seconds.
        data = vitaldb.load_case(case_id, TRACK_NAMES, interval=SAMPLING_INTERVAL_SEC)
        if data is None or len(data) == 0:
            return None
        return np.array(data, dtype=np.float32)
    except Exception as e:
        print(f"  [WARN] Could not load case {case_id}: {e}")
        return None


def extract_demographics(clinical_df: pd.DataFrame,
                         case_id: int) -> Dict[str, float]:
    """
    Pull static demographic features for a single case from the clinical table.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Full clinical table indexed by caseid.
    case_id : int
        Target case.

    Returns
    -------
    dict
        Keys: age, sex, height, weight, bmi.  Values may be NaN.
    """
    if case_id not in clinical_df.index:
        # Return all-NaN so downstream imputation still works
        return {c: np.nan for c in DEMOGRAPHIC_COLS}

    row = clinical_df.loc[case_id]
    demo = {}
    for col in DEMOGRAPHIC_COLS:
        val = row.get(col, np.nan)
        # Sex is stored as 'M'/'F' string — convert now
        if col == "sex":
            if isinstance(val, str):
                demo[col] = 0.0 if val.strip().upper() == "M" else 1.0
            else:
                demo[col] = np.nan
        else:
            demo[col] = float(val) if pd.notna(val) else np.nan
    return demo


# 3. PATIENT-LEVEL TRAIN / TEST SPLIT

def split_patients(case_ids: List[int]) -> Tuple[List[int], List[int]]:
    """
    Perform an 80/20 patient-level split BEFORE any windowing or normalization.

    Parameters
    ----------
    case_ids : list[int]
        All case IDs to be included in the study.

    Returns
    -------
    (train_ids, test_ids) : tuple of list[int]
    """
    train_ids, test_ids = train_test_split(
        case_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"[INFO] Patient split — Train: {len(train_ids)}, Test: {len(test_ids)}")
    return sorted(train_ids), sorted(test_ids)


# 4. PREPROCESSING & IMPUTATION

def _classify_nan_gaps(series: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find all contiguous NaN runs in a 1-D array.

    Returns
    -------
    list of (start_idx, length) tuples.
    """
    is_nan = np.isnan(series)
    gaps = []
    i = 0
    n = len(series)
    while i < n:
        if is_nan[i]:
            start = i
            while i < n and is_nan[i]:
                i += 1
            gaps.append((start, i - start))
        else:
            i += 1
    return gaps


def _build_large_gap_mask(series: np.ndarray) -> np.ndarray:
    """
    Build a boolean mask that is True at positions falling inside a
    NaN gap longer than ``GAP_FFILL_MAX`` samples (> 5 min).

    This mask is later used to discard any observation window that overlaps
    these regions.

    Parameters
    ----------
    series : np.ndarray
        1-D time-series (ART_MBP or HR) BEFORE imputation.

    Returns
    -------
    np.ndarray
        Boolean mask (same length as *series*). True = large gap.
    """
    mask = np.zeros(len(series), dtype=bool)
    for start, length in _classify_nan_gaps(series):
        if length > GAP_FFILL_MAX:
            mask[start: start + length] = True
    return mask


def impute_vitals(series: np.ndarray) -> np.ndarray:
    """
    Apply the three-tier imputation strategy to ART_MBP or HR:
        • gaps < 1 min  (< 30 samples)  → linear interpolation
        • gaps 1–5 min  (30–150 samples) → forward-fill
        • gaps > 5 min  (> 150 samples)  → left as NaN (windows discarded later)

    The function works **in-place on a copy** so the original is untouched.

    Parameters
    ----------
    series : np.ndarray
        1-D float32 array with potential NaN gaps.

    Returns
    -------
    np.ndarray
        Imputed copy of the series.
    """
    s = series.copy()
    gaps = _classify_nan_gaps(series)

    for start, length in gaps:
        end = start + length  # exclusive end index

        if length < GAP_INTERP_MAX:
            left_val = s[start - 1] if start > 0 and not np.isnan(s[start - 1]) else None
            right_val = s[end] if end < len(s) and not np.isnan(s[end]) else None

            if left_val is not None and right_val is not None:
                s[start:end] = np.linspace(left_val, right_val, length + 2)[1:-1]
            elif left_val is not None:
                s[start:end] = left_val
            elif right_val is not None:
                s[start:end] = right_val

        elif length <= GAP_FFILL_MAX:
            left_val = s[start - 1] if start > 0 and not np.isnan(s[start - 1]) else None
            if left_val is not None:
                s[start:end] = left_val

    return s


def preprocess_case(tracks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply all preprocessing to a single case's time-series data.

    Steps
    -----
    1. Fill drug concentrations (PPF20_CE, RFTN20_CE) NaN → 0.
    2. Build a large-gap mask from ART_MBP and HR.
    3. Impute ART_MBP and HR using the three-tier strategy.

    Parameters
    ----------
    tracks : np.ndarray
        Shape ``(T, 4)`` — columns [ART_MBP, HR, PPF20_CE, RFTN20_CE].

    Returns
    -------
    (processed_tracks, large_gap_mask) : tuple
        processed_tracks : np.ndarray, shape (T, 4) — imputed data.
        large_gap_mask   : np.ndarray, shape (T,)   — True where a >5 min gap
                           exists in either ART_MBP or HR.
    """
    processed = tracks.copy()

    processed[:, 2] = np.nan_to_num(processed[:, 2], nan=0.0)  # PPF20_CE
    processed[:, 3] = np.nan_to_num(processed[:, 3], nan=0.0)  # RFTN20_CE

    large_gap_mbp = _build_large_gap_mask(tracks[:, 0])  # ART_MBP original
    large_gap_hr  = _build_large_gap_mask(tracks[:, 1])  # HR original
    large_gap_mask = large_gap_mbp | large_gap_hr

    processed[:, 0] = impute_vitals(tracks[:, 0])  # ART_MBP
    processed[:, 1] = impute_vitals(tracks[:, 1])  # HR

    return processed, large_gap_mask


# 5. SLIDING-WINDOW GENERATION

def generate_windows(
    tracks: np.ndarray,
    large_gap_mask: np.ndarray,
    demographics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide an (observation + lead-gap + prediction) window across one case.

    Layout per position ``i``:
        [ obs_window (900) | lead_gap (150) | pred_window (300) ]
        ^                                                        ^
        i                                                   i + 1350

    Discard any window where:
        • The observation span overlaps a large-gap region.
        • The observation span still contains NaN after imputation
          (edge effect from gaps at the very start/end of the recording).
        • ART_MBP drops below ``IOH_THRESHOLD`` at any point during the
          lead-gap region (IOH event in the blind spot → window unusable).

    Parameters
    ----------
    tracks : np.ndarray
        Shape ``(T, 4)`` — preprocessed time-series.
    large_gap_mask : np.ndarray
        Shape ``(T,)`` — boolean, True at positions inside >5 min gaps.
    demographics : np.ndarray
        Shape ``(5,)`` — static features for this patient.

    Returns
    -------
    (X_seq_list, X_static_list, Y_list) : tuple of np.ndarray
        X_seq    : (N_windows, 900, 4)
        X_static : (N_windows, 5)
        Y        : (N_windows,) — binary labels
        May be empty arrays if no valid windows exist.
    """
    T = len(tracks)
    X_seq_list = []
    X_static_list = []
    Y_list = []

    for start in range(0, T - TOTAL_SPAN + 1, STEP_SIZE):
        obs_end   = start + OBS_WINDOW            # end of observation (exclusive)
        pred_start = start + OBS_WINDOW + LEAD_GAP # start of prediction window
        pred_end  = pred_start + PRED_WINDOW       # end of prediction window (excl.)

        if np.any(large_gap_mask[start:obs_end]):
            continue

        obs_data = tracks[start:obs_end]  # (900, 4)
        if np.any(np.isnan(obs_data)):
            continue

        lead_gap_mbp = tracks[obs_end:pred_start, 0]  # ART_MBP in lead gap
        if np.any(lead_gap_mbp < IOH_THRESHOLD):
            continue

        pred_mbp = tracks[pred_start:pred_end, 0]  # ART_MBP in pred window

        # If the prediction window has NaN in ART_MBP, we cannot label — skip
        if np.any(np.isnan(pred_mbp)):
            continue

        label = _label_window(pred_mbp)

        X_seq_list.append(obs_data)
        X_static_list.append(demographics)
        Y_list.append(label)

    if len(X_seq_list) == 0:
        return (np.empty((0, OBS_WINDOW, 4), dtype=np.float32),
                np.empty((0, len(DEMOGRAPHIC_COLS)), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    return (
        np.stack(X_seq_list).astype(np.float32),
        np.stack(X_static_list).astype(np.float32),
        np.array(Y_list, dtype=np.int64),
    )


# 6. LABELING LOGIC

def _label_window(pred_mbp: np.ndarray) -> int:
    """
    Assign Y = 1 (positive for IOH) if ART_MBP drops **strictly below**
    ``IOH_THRESHOLD`` mmHg for at least ``IOH_MIN_CONSECUTIVE`` consecutive
    samples anywhere in the prediction window.

    Parameters
    ----------
    pred_mbp : np.ndarray
        1-D array of ART_MBP values in the prediction window (300 samples).

    Returns
    -------
    int
        1 if IOH event detected, 0 otherwise.
    """
    below = (pred_mbp < IOH_THRESHOLD).astype(np.int8)

    max_run = 0
    current_run = 0
    for val in below:
        if val:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return 1 if max_run >= IOH_MIN_CONSECUTIVE else 0


# 7. DATASET BALANCING

def balance_windows(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Undersample the negative class in the **training set only** to achieve a
    1:3 (positive:negative) ratio.

    Parameters
    ----------
    X_seq, X_static, Y : np.ndarray
        Concatenated windows from all training patients.

    Returns
    -------
    Balanced copies of (X_seq, X_static, Y).
    """
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]

    n_pos = len(pos_idx)
    n_neg_desired = int(n_pos / POS_NEG_RATIO)  # 3× positives

    print(f"[INFO] Balancing — positives: {n_pos}, "
          f"negatives before: {len(neg_idx)}, target: {n_neg_desired}")

    if len(neg_idx) <= n_neg_desired:
        print("[INFO] No undersampling needed (natural ratio ≤ target).")
        return X_seq, X_static, Y

    rng = np.random.default_rng(RANDOM_STATE)
    neg_keep = rng.choice(neg_idx, size=n_neg_desired, replace=False)

    keep_idx = np.sort(np.concatenate([pos_idx, neg_keep]))

    return X_seq[keep_idx], X_static[keep_idx], Y[keep_idx]


# 8. NORMALIZATION

def compute_zscore_stats(
    all_train_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and std from the **training observation windows**.

    Parameters
    ----------
    all_train_seq : np.ndarray
        Shape ``(N_train_windows, 900, 4)``.

    Returns
    -------
    (mean, std) : tuple of np.ndarray
        Each of shape ``(4,)``.
    """
    flat = all_train_seq.reshape(-1, all_train_seq.shape[-1])  # (N*900, 4)
    mean = np.nanmean(flat, axis=0)
    std  = np.nanstd(flat, axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_minmax_stats(
    all_train_static: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature min and max from the **training demographics**.

    Parameters
    ----------
    all_train_static : np.ndarray
        Shape ``(N_train_windows, 5)``.

    Returns
    -------
    (feat_min, feat_max) : tuple of np.ndarray
        Each of shape ``(5,)``.
    """
    feat_min = np.nanmin(all_train_static, axis=0)
    feat_max = np.nanmax(all_train_static, axis=0)
    rng = feat_max - feat_min
    rng[rng < 1e-8] = 1.0
    feat_max = feat_min + rng  # adjusted max
    return feat_min.astype(np.float32), feat_max.astype(np.float32)


def apply_zscore(
    seq: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Z-score standardize time-series: (x - μ) / σ."""
    return ((seq - mean) / std).astype(np.float32)


def apply_minmax(
    static: np.ndarray, feat_min: np.ndarray, feat_max: np.ndarray
) -> np.ndarray:
    """Min-Max normalize demographics to [0, 1]."""
    return ((static - feat_min) / (feat_max - feat_min)).astype(np.float32)


# 9. PERSISTENCE  (one .pt file per patient)

def save_patient_file(
    case_id: int,
    X_seq: np.ndarray,
    X_static: np.ndarray,
    Y: np.ndarray,
    output_dir: str,
) -> int:
    """
    Save all windows for a single patient into one ``.pt`` file.

    File content is a dict::

        {
            'X_seq':    Tensor of shape (N_windows, 900, 4),
            'X_static': Tensor of shape (N_windows, 5),
            'Y':        Tensor of shape (N_windows,),
        }

    Parameters
    ----------
    case_id : int
        Patient / case identifier.
    X_seq, X_static, Y : np.ndarray
        Arrays for this patient's windows.
    output_dir : str
        Directory (train or test) to write to.

    Returns
    -------
    int
        Number of windows saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(Y)
    if n == 0:
        return 0

    payload = {
        "X_seq":    torch.tensor(X_seq, dtype=torch.float32),
        "X_static": torch.tensor(X_static, dtype=torch.float32),
        "Y":        torch.tensor(Y, dtype=torch.long),
    }
    path = os.path.join(output_dir, f"case_{case_id}.pt")
    torch.save(payload, path)
    return n


def save_manifest_and_scalers(
    manifest: Dict[int, int],
    zscore_mean: np.ndarray,
    zscore_std: np.ndarray,
    minmax_min: np.ndarray,
    minmax_max: np.ndarray,
    output_dir: str,
) -> None:
    """
    Persist the per-patient window counts (manifest) and scaler parameters
    so that the Dataset can reconstruct the global index and new data can be
    transformed identically.

    Saved as ``pipeline_meta.pkl`` in *output_dir*.

    Parameters
    ----------
    manifest : dict
        {case_id: n_windows, …} for this split (train or test).
    zscore_mean, zscore_std : np.ndarray
        Z-score parameters fitted on training time-series.
    minmax_min, minmax_max : np.ndarray
        Min-Max parameters fitted on training demographics.
    output_dir : str
        Root ``processed_data/`` directory.
    """
    meta = {
        "manifest": manifest,
        "zscore_mean": zscore_mean,
        "zscore_std": zscore_std,
        "minmax_min": minmax_min,
        "minmax_max": minmax_max,
    }
    path = os.path.join(output_dir, "pipeline_meta.pkl")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[INFO] Saved pipeline metadata → {path}")


# 10. PYTORCH DATASET & DATALOADER

class IOHDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for IOH prediction windows.

    Each global index maps to a ``(case_id, window_index)`` pair.
    Patient files are loaded on demand and cached with an LRU cache to
    avoid reloading the same file for consecutive windows of the same patient.

    Yields
    ------
    (X_seq, X_static, Y) where:
        X_seq   : Tensor [900, 4]
        X_static: Tensor [5]
        Y       : Tensor scalar (0 or 1)
    """

    def __init__(self, data_dir: str, manifest: Dict[int, int]):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing per-patient ``.pt`` files.
        manifest : dict
            {case_id: n_windows} — number of windows per patient file.
        """
        super().__init__()
        self.data_dir = data_dir

        # Build the master index: list of (case_id, window_index)
        self.index_map: List[Tuple[int, int]] = []
        for case_id in sorted(manifest.keys()):
            n_windows = manifest[case_id]
            for w in range(n_windows):
                self.index_map.append((case_id, w))

        # Wrap the file loader in an LRU cache keyed by case_id
        self._load_case_file = functools.lru_cache(maxsize=LRU_CACHE_SIZE)(
            self._load_case_file_impl
        )

    def _load_case_file_impl(self, case_id: int) -> Dict[str, torch.Tensor]:
        """
        Load a single patient's ``.pt`` file from disk.

        This method is wrapped by ``functools.lru_cache`` so recently-accessed
        files stay in memory.
        """
        path = os.path.join(self.data_dir, f"case_{case_id}.pt")
        return torch.load(path, weights_only=True)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case_id, window_idx = self.index_map[idx]
        data = self._load_case_file(case_id)

        X_seq    = data["X_seq"][window_idx]     # (900, 4)
        X_static = data["X_static"][window_idx]  # (5,)
        Y        = data["Y"][window_idx]         # scalar

        return X_seq, X_static, Y


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_manifest: Dict[int, int],
    test_manifest: Dict[int, int],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from the saved per-patient files.

    Parameters
    ----------
    train_dir, test_dir : str
        Directories containing per-patient .pt files.
    train_manifest, test_manifest : dict
        {case_id: n_windows} for each split.
    batch_size : int
    num_workers : int

    Returns
    -------
    (train_loader, test_loader) : tuple of DataLoader
    """
    train_ds = IOHDataset(train_dir, train_manifest)
    test_ds  = IOHDataset(test_dir, test_manifest)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"[INFO] DataLoaders created — "
          f"Train: {len(train_ds)} windows ({len(train_loader)} batches), "
          f"Test: {len(test_ds)} windows ({len(test_loader)} batches)")

    return train_loader, test_loader

# 11. MAIN PIPELINE

def main() -> Tuple[DataLoader, DataLoader]:
    """
    Run the full preprocessing pipeline end-to-end:

    1. Download clinical metadata.
    2. Resolve case IDs.
    3. Patient-level train/test split.
    4. For each case: extract tracks → preprocess → generate windows.
    5. Impute missing demographics (training medians).
    6. Balance training windows.
    7. Fit normalization on training set; apply to both.
    8. Save per-patient .pt files + manifest/scalers.
    9. Build and return DataLoaders.

    Returns
    -------
    (train_loader, test_loader) : tuple of DataLoader
    """
    print("=" * 72)
    print("  VitalDB IOH Prediction — Data Preprocessing Pipeline")
    print("=" * 72)

    # Step 1: Clinical metadata
    clinical_df = download_clinical_data()

    # Step 2: Resolve case IDs
    case_ids = resolve_case_ids(clinical_df)

    # Step 3: Patient-level split (BEFORE windowing/normalization)
    train_ids, test_ids = split_patients(case_ids)

    # Step 4 & 5: Extract, preprocess, and window each case

    # Storage: {case_id: (X_seq, X_static, Y)}
    train_windows: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    test_windows:  Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Also collect raw demographics for training-set median imputation
    train_demographics_raw: List[Dict[str, float]] = []

    # First pass: extract demographics for all training patients
    # (needed to compute medians before we impute any patient)
    for cid in train_ids:
        demo = extract_demographics(clinical_df, cid)
        train_demographics_raw.append(demo)

    # Compute training-set medians for each demographic feature
    demo_df = pd.DataFrame(train_demographics_raw)
    demo_medians: Dict[str, float] = {}
    for col in DEMOGRAPHIC_COLS:
        if col == "sex":
            # Sex is already 0/1; median doesn't make clinical sense.
            # Use mode (most frequent) instead.
            mode_val = demo_df[col].mode()
            demo_medians[col] = float(mode_val.iloc[0]) if len(mode_val) > 0 else 0.0
        else:
            demo_medians[col] = float(demo_df[col].median())
    print(f"[INFO] Training demographic medians: {demo_medians}")

    def _impute_demographics(demo: Dict[str, float]) -> np.ndarray:
        """Replace NaN demographics with training medians and return array."""
        arr = np.array([
            demo[col] if not np.isnan(demo[col]) else demo_medians[col]
            for col in DEMOGRAPHIC_COLS
        ], dtype=np.float32)
        return arr

    # Second pass: full extraction + windowing per case
    def _process_split(case_id_list: List[int],
                       storage: Dict[int, Tuple],
                       split_name: str):
        """Extract, preprocess, and window all cases in a split."""
        for cid in tqdm(case_id_list, desc=f"Processing {split_name}"):
            # 4a. Extract tracks
            tracks = extract_case_tracks(cid)
            if tracks is None:
                continue

            # 4b. Extract & impute demographics
            demo_raw = extract_demographics(clinical_df, cid)
            demo_arr = _impute_demographics(demo_raw)

            # 4c. Preprocess (impute vitals, fill drugs, build gap mask)
            processed_tracks, large_gap_mask = preprocess_case(tracks)

            # 4d. Generate sliding windows + labels
            X_seq, X_static, Y = generate_windows(
                processed_tracks, large_gap_mask, demo_arr
            )

            if len(Y) > 0:
                storage[cid] = (X_seq, X_static, Y)

    _process_split(train_ids, train_windows, "train")
    _process_split(test_ids,  test_windows,  "test")

    # Concatenate all training windows for global statistics
    if len(train_windows) == 0:
        raise RuntimeError("No valid training windows produced! "
                           "Check case IDs and data availability.")

    all_train_seq    = np.concatenate([v[0] for v in train_windows.values()], axis=0)
    all_train_static = np.concatenate([v[1] for v in train_windows.values()], axis=0)
    all_train_Y      = np.concatenate([v[2] for v in train_windows.values()], axis=0)

    n_pos_before = int(all_train_Y.sum())
    n_neg_before = len(all_train_Y) - n_pos_before
    print(f"[INFO] Training windows BEFORE balancing — "
          f"Pos: {n_pos_before}, Neg: {n_neg_before}, "
          f"Total: {len(all_train_Y)}")

    # Step 6: Balance the training set
    all_train_seq, all_train_static, all_train_Y = balance_windows(
        all_train_seq, all_train_static, all_train_Y
    )

    n_pos_after = int(all_train_Y.sum())
    n_neg_after = len(all_train_Y) - n_pos_after
    print(f"[INFO] Training windows AFTER balancing  — "
          f"Pos: {n_pos_after}, Neg: {n_neg_after}, "
          f"Total: {len(all_train_Y)}")

    # Step 7: Fit normalization on training set
    zscore_mean, zscore_std = compute_zscore_stats(all_train_seq)
    minmax_min, minmax_max  = compute_minmax_stats(all_train_static)

    print(f"[INFO] Z-score params — mean: {zscore_mean}, std: {zscore_std}")
    print(f"[INFO] MinMax params  — min:  {minmax_min},  max: {minmax_max}")

    # ── Redistribute balanced training windows back to per-patient groups ──

    # Rebuild per-patient mapping
    balanced_train_per_patient: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    sorted_train_cids = sorted(train_windows.keys())
    case_offset = {}  # case_id → (start_idx, end_idx) in the pre-balance array
    offset = 0
    for cid in sorted_train_cids:
        n = len(train_windows[cid][2])
        case_offset[cid] = (offset, offset + n)
        offset += n

    static_to_cid: Dict[tuple, int] = {}
    for cid in sorted_train_cids:
        static_vec = tuple(train_windows[cid][1][0].tolist())
        static_to_cid[static_vec] = cid

    # Assign each balanced window back to its patient
    for cid in sorted_train_cids:
        balanced_train_per_patient[cid] = ([], [], [])  # type: ignore

    for i in range(len(all_train_Y)):
        svec = tuple(all_train_static[i].tolist())
        cid = static_to_cid.get(svec)
        if cid is None:
            continue
        balanced_train_per_patient[cid][0].append(all_train_seq[i])     # type: ignore
        balanced_train_per_patient[cid][1].append(all_train_static[i])  # type: ignore
        balanced_train_per_patient[cid][2].append(all_train_Y[i])       # type: ignore

    # Convert lists → numpy arrays
    for cid in list(balanced_train_per_patient.keys()):
        seqs, stats, ys = balanced_train_per_patient[cid]
        if len(ys) == 0:
            del balanced_train_per_patient[cid]
            continue
        balanced_train_per_patient[cid] = (
            np.stack(seqs).astype(np.float32),
            np.stack(stats).astype(np.float32),
            np.array(ys, dtype=np.int64),
        )

    # Step 8: Normalize and save per-patient .pt files
    train_manifest: Dict[int, int] = {}
    test_manifest:  Dict[int, int] = {}

    # Training set
    for cid, (seq, stat, y) in balanced_train_per_patient.items():
        seq_norm  = apply_zscore(seq, zscore_mean, zscore_std)
        stat_norm = apply_minmax(stat, minmax_min, minmax_max)
        n_saved = save_patient_file(cid, seq_norm, stat_norm, y, TRAIN_DIR)
        if n_saved > 0:
            train_manifest[cid] = n_saved

    # Test set (no balancing, just normalize)
    for cid, (seq, stat, y) in test_windows.items():
        seq_norm  = apply_zscore(seq, zscore_mean, zscore_std)
        stat_norm = apply_minmax(stat, minmax_min, minmax_max)
        n_saved = save_patient_file(cid, seq_norm, stat_norm, y, TEST_DIR)
        if n_saved > 0:
            test_manifest[cid] = n_saved

    # Save manifest + scaler metadata
    save_manifest_and_scalers(
        manifest={"train": train_manifest, "test": test_manifest},
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
        minmax_min=minmax_min,
        minmax_max=minmax_max,
        output_dir=OUTPUT_DIR,
    )

    # Step 9: Build DataLoaders
    train_loader, test_loader = create_dataloaders(
        TRAIN_DIR, TEST_DIR, train_manifest, test_manifest
    )

    # Summary statistics
    total_train = sum(train_manifest.values())
    total_test  = sum(test_manifest.values())

    # Count labels in test set
    test_Y_all = []
    for cid in test_manifest:
        test_Y_all.append(test_windows[cid][2])
    if test_Y_all:
        test_Y_cat = np.concatenate(test_Y_all)
        test_pos = int(test_Y_cat.sum())
        test_neg = len(test_Y_cat) - test_pos
    else:
        test_pos = test_neg = 0

    print("\n" + "=" * 72)
    print("  Pipeline Complete — Summary")
    print("=" * 72)
    print(f"  Train patients : {len(train_manifest)}")
    print(f"  Train windows  : {total_train}  "
          f"(Pos: {n_pos_after}, Neg: {n_neg_after})")
    print(f"  Test patients  : {len(test_manifest)}")
    print(f"  Test windows   : {total_test}  "
          f"(Pos: {test_pos}, Neg: {test_neg})")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  X_seq shape    : [batch, {OBS_WINDOW}, {len(TRACK_NAMES)}]")
    print(f"  X_static shape : [batch, {len(DEMOGRAPHIC_COLS)}]")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print("=" * 72)

    return train_loader, test_loader

# ENTRY POINT
if __name__ == "__main__":
    train_loader, test_loader = main()

    print("\nFetching one training batch")
    for X_seq_batch, X_static_batch, Y_batch in train_loader:
        print(f"  X_seq: {X_seq_batch.shape}")    # [B, 900, 4]
        print(f"  X_static: {X_static_batch.shape}")  # [B, 5]
        print(f"  Y: {Y_batch.shape}")          # [B]
        print(f"  Y values: {Y_batch.tolist()[:10]} …")
        break