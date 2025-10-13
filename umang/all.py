# ========================================
# HYPERPARAMETER TUNING FOR PRICE PREDICTION
# ========================================
import os
import sys
import logging
import builtins

# Configure logging: write INFO logs to umang/log.txt and also log to stdout
log_path = os.path.join(os.path.dirname(__file__), "log.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler (append)
file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Stream handler so prints still appear on console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(stream_handler)

# Preserve original print in case some prints target other files
_orig_print = builtins.print

def _print_to_log(*args, **kwargs):
    # If a file parameter is provided and it's not stdout/stderr, use original print
    file = kwargs.get('file', None)
    if file not in (None, sys.stdout, sys.stderr):
        _orig_print(*args, **kwargs)
        return

    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(str(a) for a in args)

    # Send to logger (this writes to file and the stream handler)
    # Logger does not append an extra newline, so we don't include end here.
    logger.info(msg)

    # If a custom end is requested (not the default '\n'), emulate it on stdout
    if end != '\n':
        try:
            sys.stdout.write(end)
            if kwargs.get('flush', False):
                sys.stdout.flush()
        except Exception:
            pass

# Override built-in print for this process so existing print(...) calls are captured
builtins.print = _print_to_log

# Start message (will be routed through logger/print override)
print("Starting hyperparameter tuning...")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import gc # Garbage Collector
import itertools

# ======================
# 1. HYPERPARAMETER GRID
# ======================
# Define the search space for the hyperparameters you want to tune
hyperparameter_grid = {
    'LEARNING_RATE': [1e-4, 3e-4, 5e-4],
    'WEIGHT_DECAY': [1e-5, 5e-5, 1e-4],
    'DROPOUT_RATE': [0.2, 0.3, 0.4]
}

# General settings
N_SPLITS = 3
EPOCHS = 50 # You might want to reduce epochs for faster tuning
BATCH_SIZE = 128

# ======================
# 2. DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ======================
# 3. LOAD ALL DATA
# ======================
print("Loading all image and text embeddings...")
all_img_emb = np.load("../clip_embeddings/train_image_embeddings_clip.npy")
all_text_emb = np.load("../embeddings/all_text_emb.npy")

X_all = np.concatenate([all_img_emb, all_text_emb], axis=1)
all_df = pd.read_csv("../clean_data/train_cleaned.csv")
y_all = all_df["price"].values

del all_img_emb, all_text_emb, all_df
gc.collect()
print(f"Final combined feature matrix shape: {X_all.shape}")
print(f"Final combined target array shape: {y_all.shape}")

use_log = True
if use_log:
    print("Applying log1p transformation to target prices.")
    y_all = np.log1p(y_all)

# ======================
# 4. DATASET CLASS
# ======================
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# 5. MODEL ARCHITECTURE (MODIFIED)
# ======================
class WideDeepMLP(nn.Module):
    # Added dropout_rate as a parameter
    def __init__(self, input_dim, dropout_rate=0.2):
        super().__init__()
        self.deep1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate) # Use parameter
        )
        self.deep2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate) # Use parameter
        )
        self.deep3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.res_proj1 = nn.Linear(1024, 512)
        self.res_proj2 = nn.Linear(512, 256)
        self.wide = nn.Linear(input_dim, 64)
        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.out = nn.Linear(256 + 64, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        d1 = self.deep1(x)
        d2 = self.deep2(d1) + self.res_proj1(d1)
        d3 = self.deep3(d2) + self.res_proj2(d2)
        attn_weights = self.attn(d3)
        d3 = d3 * attn_weights
        wide_feat = self.wide(x)
        combined = torch.cat([d3, wide_feat], dim=1)
        return self.softplus(self.out(combined)).squeeze(1)

# ======================
# 6. SMAPE METRIC
# ======================
def smape(y_true, y_pred):
    if use_log:
        y_true = torch.expm1(y_true)
        y_pred = torch.expm1(y_pred)
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + 1e-8
    return 100 * torch.mean(numerator / denominator)

# ======================
# 7. HYPERPARAMETER TUNING LOOP
# ======================
best_avg_smape = float("inf")
best_params = {}
trial_count = 0

# Create all combinations of hyperparameters
keys, values = zip(*hyperparameter_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
total_trials = len(param_combinations)

for params in param_combinations:
    trial_count += 1
    print("="*60)
    print(f"🚀 STARTING TRIAL {trial_count}/{total_trials}")
    print(f"🧪 Testing params: {params}")
    print("="*60)
    
    fold_best_smapes = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        
        train_ds = PriceDataset(X_train, y_train)
        val_ds = PriceDataset(X_val, y_val)
        
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model and optimizer with current trial's hyperparameters
        input_dim = X_train.shape[1]
        model = WideDeepMLP(input_dim=input_dim, dropout_rate=params['DROPOUT_RATE']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        loss_fn = nn.L1Loss()
        
        fold_best_smape = float("inf")
        
        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    all_preds.append(preds)
                    all_targets.append(y_batch)
            
            all_preds_tensor = torch.cat(all_preds)
            all_targets_tensor = torch.cat(all_targets)
            val_smape = smape(all_targets_tensor, all_preds_tensor).item()
            
            scheduler.step()
            
            if val_smape < fold_best_smape:
                fold_best_smape = val_smape
                # torch.save(model.state_dict(), f"best_model.pt") # Saving is disabled during tuning
        
        print(f"Fold {fold+1} Best Val SMAPE: {fold_best_smape:.4f}")
        fold_best_smapes.append(fold_best_smape)

    # --- After all folds for one trial are done ---
    avg_smape_for_trial = np.mean(fold_best_smapes)
    print("-"*60)
    print(f"📊 Average SMAPE for trial {trial_count} (params: {params}): {avg_smape_for_trial:.4f}")
    
    if avg_smape_for_trial < best_avg_smape:
        best_avg_smape = avg_smape_for_trial
        best_params = params
        print(f"🏆 NEW BEST! Average SMAPE: {best_avg_smape:.4f}")

# --- After all trials are done ---
print("\n\n" + "="*60)
print("🎉 HYPERPARAMETER TUNING COMPLETE 🎉")
print(f"Best Average Validation SMAPE: {best_avg_smape:.4f}")
print(f"Best Hyperparameters: {best_params}")
print("="*60)