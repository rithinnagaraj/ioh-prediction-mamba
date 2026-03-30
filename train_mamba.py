import os
import random
import pickle
import time  # <-- Added for timing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import functools
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Optional

from model_mamba import IOHMambaPredictor

TRAIN_DIR = "processed_data/train"
TEST_DIR = "processed_data/test"
OUTPUT_DIR = "processed_data"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 5  # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class IOHDataset(Dataset):
    def __init__(self, data_dir: str, manifest: Dict[int, int]):
        super().__init__()
        self.data_dir = data_dir
        
        # 1. Calculate total windows to pre-allocate exact memory
        total_windows = sum(manifest.values())
        print(f"Pre-allocating RAM for {total_windows} windows...")

        # 2. Pre-allocate massive empty tensors (Zero memory spike!)
        self.X_seq = torch.empty((total_windows, 900, 4), dtype=torch.float32)
        self.X_static = torch.empty((total_windows, 5), dtype=torch.float32)
        self.Y = torch.empty((total_windows,), dtype=torch.long)

        # 3. Fill the tensors directly from disk
        current_idx = 0
        for case_id in tqdm(sorted(manifest.keys())):
            n_windows = manifest[case_id]
            if n_windows == 0:
                continue
                
            path = os.path.join(data_dir, f"case_{case_id}.pt")
            data = torch.load(path, weights_only=True)
            assert data["X_seq"].shape[0] == n_windows, \
                f"case_{case_id}: manifest says {n_windows} windows but file has {data['X_seq'].shape[0]}"
            
            # Slot the data into the pre-allocated block
            end_idx = current_idx + n_windows
            self.X_seq[current_idx:end_idx] = data["X_seq"]
            self.X_static[current_idx:end_idx] = data["X_static"]
            self.Y[current_idx:end_idx] = data["Y"]
            
            current_idx = end_idx

        print(f"Dataset successfully loaded into RAM. Size: {self.X_seq.element_size() * self.X_seq.nelement() / 1e9:.2f} GB")

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        return self.X_seq[idx], self.X_static[idx], self.Y[idx]

def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(seed):

    meta_path = os.path.join(OUTPUT_DIR, "pipeline_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    full_train_manifest = meta["manifest"]["train"]
    test_manifest = meta["manifest"]["test"]

    # Extract patient IDs (keys) from the training manifest
    train_case_ids = list(full_train_manifest.keys())

    # Patient-Level Validation Split (80% Train, 20% Val)
    actual_train_ids, val_ids = train_test_split(train_case_ids, test_size=0.2, random_state=seed)

    # Rebuild the manifests for Train and Val
    train_manifest = {cid: full_train_manifest[cid] for cid in actual_train_ids}
    val_manifest = {cid: full_train_manifest[cid] for cid in val_ids}

    print(f"Patients -> Train: {len(actual_train_ids)} | Val: {len(val_ids)} | Test: {len(test_manifest.keys())}")

    train_ds = IOHDataset(TRAIN_DIR, train_manifest)
    val_ds = IOHDataset(TRAIN_DIR, val_manifest)  # Val uses TRAIN_DIR because it was split from train
    test_ds = IOHDataset(TEST_DIR, test_manifest)

    # Pin memory speeds up CPU-to-GPU data transfers
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    model = IOHMambaPredictor(input_dim=4, model_dim_1=32, model_dim_2=64)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_val_auprc = 0.0
    epochs_no_improve = 0

    min_delta = 0.001

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        model.train()
        train_loss = 0.0
        
        for X_seq, X_static, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training"):
            X_seq, X_static = X_seq.to(DEVICE), X_static.to(DEVICE)
            labels = labels.float().to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_seq, X_static)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * X_seq.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for X_seq, X_static, labels in val_loader:
                X_seq, X_static = X_seq.to(DEVICE), X_static.to(DEVICE)
                labels = labels.float().to(DEVICE)

                logits = model(X_seq, X_static)
                loss = criterion(logits, labels)
                val_loss += loss.item() * X_seq.size(0)

                probs = torch.sigmoid(logits)
                all_val_preds.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        
        # Calculate Metrics
        val_auprc = average_precision_score(all_val_labels, all_val_preds)
        val_auroc = roc_auc_score(all_val_labels, all_val_preds)

        scheduler.step(val_auprc)

        epoch_duration = time.time() - epoch_start_time
        if DEVICE.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Time: {epoch_duration:.2f}s | Peak VRAM: {peak_vram_mb:.1f} MB")
        print(f"             | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f} | Val AUROC: {val_auroc:.4f}")

        if val_auprc > (best_val_auprc + min_delta):
            best_val_auprc = val_auprc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"PM_47k_D1_denovo_{seed}.pth")
            print("  -> Validation AUPRC improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    print("\n========== Evaluating on Holdout Test Set ==========")
    # Load the best weights discovered during validation
    model.load_state_dict(torch.load(f"PM_47k_D1_denovo_{seed}.pth", weights_only=True))
    model.eval()
    
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for X_seq, X_static, labels in tqdm(test_loader, desc=f"Testing: "):
            X_seq, X_static = X_seq.to(DEVICE), X_static.to(DEVICE)
            labels = labels.float().to(DEVICE)

            logits = model(X_seq, X_static)
            probs = torch.sigmoid(logits)
            
            all_test_preds.extend(probs.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    test_auprc = average_precision_score(all_test_labels, all_test_preds)
    test_auroc = roc_auc_score(all_test_labels, all_test_preds)

    print(f"FINAL TEST METRICS | AUPRC: {test_auprc:.4f} | AUROC: {test_auroc:.4f}")
    print("====================================================")
    return test_auprc, test_auroc

if __name__ == "__main__":
    SEEDS = [42, 123, 7]
    AUPRCS = []
    AUROCS = []
    
    for seed in SEEDS:
        set_seed(seed)
        test_auprc, test_auroc = main(seed)
        AUPRCS.append(test_auprc)
        AUROCS.append(test_auroc)
        print(f"Seed {seed} -> AUPRC: {test_auprc:.4f} | AUROC: {test_auroc:.4f}")
    
    print(f"\nFinal Results across {len(SEEDS)} seeds:")
    print(f"AUPRC: {np.mean(AUPRCS):.4f} +/- {np.std(AUPRCS):.4f}")
    print(f"AUROC: {np.mean(AUROCS):.4f} +/- {np.std(AUROCS):.4f}")