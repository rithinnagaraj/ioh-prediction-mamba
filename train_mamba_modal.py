import modal

# ---------------------------------------------------------------------------
# IMAGE
# Base image: pre-built Modal image deployed under
#   App  : mamba-notebook-env
#   Fn   : pre_build_mamba
#   Image: im-zxcQBsKTPGH84LbRtKGmDI
# Additional runtime deps are layered on top via pip_install.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_name("im-zxcQBsKTPGH84LbRtKGmDI")
    .pip_install(
        "scikit-learn",
        "ipywidgets",
        "jupyterlab",
    )
)

# ---------------------------------------------------------------------------
# VOLUME  –  persists /mnt/iohdata across runs (data in, checkpoints out)
# ---------------------------------------------------------------------------
iohdata_volume = modal.Volume.from_name("iohdata", create_if_missing=False)

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = modal.App("train-mamba-ioh", image=image)

# ---------------------------------------------------------------------------
# REMOTE TRAINING FUNCTION
# ---------------------------------------------------------------------------
@app.function(
    # GPU – L4; CPU / RAM left at Modal defaults (lowest reservation)
    gpu="L4",
    # Persist the data volume at the same path the script expects
    volumes={"/mnt/iohdata": iohdata_volume},
    # Ship model_mamba.py from the local working directory into the container
    mounts=[
        modal.Mount.from_local_file(
            local_path="model_mamba.py",
            remote_path="/root/model_mamba.py",
        )
    ],
    # Allow up to 24 h for the full multi-seed run
    timeout=86_400,
)
def train_mamba():
    import os
    import random
    import pickle
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, roc_auc_score
    import functools
    from tqdm.auto import tqdm
    from typing import List, Tuple, Dict

    # The mount places model_mamba.py at /root/model_mamba.py; /root is on sys.path
    from model_mamba import IOHMambaPredictor

    # -----------------------------------------------------------------------
    # PATHS & HYPERPARAMETERS
    # -----------------------------------------------------------------------
    TRAIN_DIR  = "/mnt/iohdata/processed_data/train"
    TEST_DIR   = "/mnt/iohdata/processed_data/test"
    OUTPUT_DIR = "/mnt/iohdata/processed_data"

    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-4
    EPOCHS        = 50
    PATIENCE      = 5
    LRU_CACHE_SIZE = 8

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {DEVICE}")

    # -----------------------------------------------------------------------
    # DATASET
    # -----------------------------------------------------------------------
    class IOHDataset(Dataset):
        def __init__(self, data_dir: str, manifest: Dict[int, int]):
            super().__init__()
            self.data_dir = data_dir
            self.index_map: List[Tuple[int, int]] = []
            for case_id in sorted(manifest.keys()):
                for w in range(manifest[case_id]):
                    self.index_map.append((case_id, w))

            self._load_case_file = functools.lru_cache(maxsize=LRU_CACHE_SIZE)(
                self._load_case_file_impl
            )

        def _load_case_file_impl(self, case_id: int) -> Dict[str, torch.Tensor]:
            path = os.path.join(self.data_dir, f"case_{case_id}.pt")
            return torch.load(path, weights_only=True)

        def __len__(self) -> int:
            return len(self.index_map)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            case_id, window_idx = self.index_map[idx]
            data = self._load_case_file(case_id)
            return data["X_seq"][window_idx], data["X_static"][window_idx], data["Y"][window_idx]

    # -----------------------------------------------------------------------
    # SEED HELPER
    # -----------------------------------------------------------------------
    def set_seed(seed: int, deterministic: bool = True):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark     = not deterministic

    # -----------------------------------------------------------------------
    # SINGLE-SEED TRAINING RUN
    # -----------------------------------------------------------------------
    def run_seed(seed: int):
        print(f"\n{'='*52}")
        print(f"  SEED {seed}  –  Training on {DEVICE}")
        print(f"{'='*52}")

        # Load metadata
        meta_path = os.path.join(OUTPUT_DIR, "pipeline_meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        full_train_manifest = meta["manifest"]["train"]
        test_manifest       = meta["manifest"]["test"]

        # Patient-level 80/20 split
        train_case_ids = list(full_train_manifest.keys())
        actual_train_ids, val_ids = train_test_split(
            train_case_ids, test_size=0.2, random_state=seed
        )
        train_manifest = {cid: full_train_manifest[cid] for cid in actual_train_ids}
        val_manifest   = {cid: full_train_manifest[cid] for cid in val_ids}

        print(f"Patients -> Train: {len(actual_train_ids)} | Val: {len(val_ids)} | Test: {len(test_manifest)}")

        # DataLoaders
        train_loader = DataLoader(
            IOHDataset(TRAIN_DIR, train_manifest),
            batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True,
        )
        val_loader = DataLoader(
            IOHDataset(TRAIN_DIR, val_manifest),
            batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
        )
        test_loader = DataLoader(
            IOHDataset(TEST_DIR, test_manifest),
            batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
        )

        # Model / loss / optimiser
        model = IOHMambaPredictor(input_dim=4, model_dim_1=32, model_dim_2=64).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

        best_val_auprc    = 0.0
        epochs_no_improve = 0
        min_delta         = 0.001
        ckpt_path         = f"/mnt/iohdata/PM_47k_D4_contaminated_{seed}.pth"

        for epoch in range(EPOCHS):
            epoch_start = time.time()
            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)

            # ---- Train ----
            model.train()
            train_loss = 0.0
            for X_seq, X_static, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
                X_seq, X_static = X_seq.to(DEVICE), X_static.to(DEVICE)
                labels = labels.float().to(DEVICE)

                optimizer.zero_grad()
                logits = model(X_seq, X_static)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * X_seq.size(0)

            train_loss /= len(train_loader.dataset)

            # ---- Validate ----
            model.eval()
            val_loss, all_preds, all_labels = 0.0, [], []
            with torch.no_grad():
                for X_seq, X_static, labels in val_loader:
                    X_seq, X_static = X_seq.to(DEVICE), X_static.to(DEVICE)
                    labels = labels.float().to(DEVICE)
                    logits = model(X_seq, X_static)
                    val_loss += criterion(logits, labels).item() * X_seq.size(0)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss  /= len(val_loader.dataset)
            val_auprc  = average_precision_score(all_labels, all_preds)
            val_auroc  = roc_auc_score(all_labels, all_preds)

            scheduler.step(val_auprc)

            duration  = time.time() - epoch_start
            peak_vram = (torch.cuda.max_memory_allocated(DEVICE) / 1024**2
                         if DEVICE.type == "cuda" else 0.0)

            print(f"Epoch [{epoch+1}/{EPOCHS}] | {duration:.2f}s | Peak VRAM: {peak_vram:.1f} MB")
            print(f"             | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| Val AUPRC: {val_auprc:.4f} | Val AUROC: {val_auroc:.4f}")

            # Early stopping / checkpointing
            if val_auprc > (best_val_auprc + min_delta):
                best_val_auprc    = val_auprc
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                print("  -> Validation AUPRC improved. Model saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break

        # ---- Test (blind vault) ----
        print("\n========== Evaluating on Holdout Test Set ==========")
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for X_seq, X_static, labels in tqdm(test_loader, desc="Testing"):
                X_seq, X_static = X_seq.to(DEVICE), X_static.to(DEVICE)
                labels = labels.float().to(DEVICE)
                probs  = torch.sigmoid(model(X_seq, X_static))
                test_preds.extend(probs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_auprc = average_precision_score(test_labels, test_preds)
        test_auroc = roc_auc_score(test_labels, test_preds)
        print(f"FINAL TEST METRICS | AUPRC: {test_auprc:.4f} | AUROC: {test_auroc:.4f}")
        print("====================================================")

        # Commit checkpoint to the volume so it persists after the container exits
        iohdata_volume.commit()

        return test_auprc, test_auroc

    # -----------------------------------------------------------------------
    # MULTI-SEED ENTRY POINT
    # -----------------------------------------------------------------------
    SEEDS  = [42, 123, 7]
    auprcs, aurocs = [], []

    for seed in SEEDS:
        set_seed(seed)
        auprc, auroc = run_seed(seed)
        auprcs.append(auprc)
        aurocs.append(auroc)
        print(f"Seed {seed} -> AUPRC: {auprc:.4f} | AUROC: {auroc:.4f}")

    print(f"\nFinal Results across {len(SEEDS)} seeds:")
    print(f"AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")
    print(f"AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")


# ---------------------------------------------------------------------------
# LOCAL ENTRYPOINT  –  invoked by `modal run --detach train_mamba_modal.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    train_mamba.remote()