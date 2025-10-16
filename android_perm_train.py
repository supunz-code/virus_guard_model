# android_perm_train.py
# Train a PyTorch MLP on srimeenakshiks/Android-Malware-Dataset (permissions -> malware/benign).
# Split = 70% train / 20% val / 10% test (stratified)

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
import joblib

# -------------------------
# Config
# -------------------------
HF_REPO   = "srimeenakshiks/Android-Malware-Dataset"
LABEL     = "Result"      # Force label column
BATCH_SIZE = 1024
EPOCHS     = 25
LR         = 1e-3
DROPOUT    = 0.30
SEED       = 42

# Common non-feature columns we don't want as inputs
META_DROP = [
    "id","ID","sha256","hash","pkg","package","apk","app_name",
    "column_name","column_type","null","key","default","extra"
]

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Feature selection + arrays
# -------------------------
def select_permission_feature_cols(all_cols, label_col):
    """
    Keep columns that *look like* Android permission or package flags.
    """
    def is_perm_like(c: str) -> bool:
        if c in META_DROP or c == label_col:
            return False
        lc = c.lower()
        return (
            "permission" in lc
            or c.startswith(("android.", "com.", "org.", "me.", "net."))
        )
    return [c for c in all_cols if is_perm_like(c)]

def ds_to_arrays(ds, label_col):
    cols = ds.column_names
    if label_col not in cols:
        raise RuntimeError(f"Label column '{label_col}' not found. Columns: {cols[:40]}")

    feat_cols = select_permission_feature_cols(cols, label_col)
    if not feat_cols:
        raise RuntimeError("No feature columns selected — check dataset schema/filters.")

    # Build numeric DataFrame
    X = pd.DataFrame({c: ds[c] for c in feat_cols})
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0).astype("float32")

    # Labels: ensure {0,1}
    y_raw = np.array(ds[label_col])
    if y_raw.dtype.kind in {"U","S","O"}:
        mapping = {"benign":0,"Benign":0,"benign_app":0,
                   "malware":1,"Malware":1,"malicious":1}
        y = np.array([mapping.get(str(v), v) for v in y_raw], dtype="float32")
    else:
        y = y_raw.astype("float32")

    if not np.isin(np.unique(y), [0.0, 1.0]).all():
        y = (y >= 0.5).astype("float32")

    return X, y, feat_cols

# -------------------------
# Data utils
# -------------------------
def make_loader(X, y, bs=1024, shuffle=False):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=shuffle)

def ratio(y):
    y = np.asarray(y).ravel()
    pos = (y == 1).mean() if len(y) else 0.0
    return f"pos={pos:.3f}, neg={1-pos:.3f}, n={len(y)}"

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, d, p=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128,1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)

    print("Loading dataset from Hugging Face…")
    ds = load_dataset(HF_REPO)
    d = ds["train"]
    print(f"Rows: {len(d)}")

    label_col = LABEL
    print("Label column:", label_col)

    X_df, y, feat_cols = ds_to_arrays(d, label_col)
    print(f"Selected {len(feat_cols)} feature columns.")
    print("Sample features:", feat_cols[:10])

    # -------- Stratified 70/20/10 split --------
    # Step 1: hold out Test = 10%
    X_temp, Xte_df, y_temp, yte = train_test_split(
        X_df, y, test_size=0.10, random_state=SEED, stratify=y
    )
    # Step 2: split remaining 90% into Train (70%) and Val (20%)
    # Train fraction within 90% is 7/9; val is 2/9.
    Xtr_df, Xva_df, ytr, yva = train_test_split(
        X_temp, y_temp, test_size=2/9, random_state=SEED, stratify=y_temp
    )

    print("Class balance:")
    print("  Train:", ratio(ytr))
    print("  Val  :", ratio(yva))
    print("  Test :", ratio(yte))

    # -------- Numpy + Scaling --------
    Xtr = np.asarray(Xtr_df, dtype="float32")
    Xva = np.asarray(Xva_df, dtype="float32")
    Xte = np.asarray(Xte_df, dtype="float32")
    ytr = ytr.astype("float32")
    yva = yva.astype("float32")
    yte = yte.astype("float32")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)
    joblib.dump(scaler, "android_perm_scaler.joblib")

    # -------- DataLoaders --------
    train_loader = make_loader(Xtr, ytr, bs=BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(Xva, yva, bs=max(1024, BATCH_SIZE))
    test_loader  = make_loader(Xte, yte, bs=max(1024, BATCH_SIZE))

    # -------- Model / Optimizer --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = MLP(d=Xtr.shape[1], p=DROPOUT).to(device)
    crit  = nn.BCELoss()
    opt   = optim.Adam(model.parameters(), lr=LR)

    # -------- Training --------
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item()

        # Validation AUC
        model.eval()
        with torch.no_grad():
            v_probs, v_gts = [], []
            for xb, yb in val_loader:
                p = model(xb.to(device)).cpu().numpy().ravel()
                v_probs.append(p)
                v_gts.append(yb.numpy().ravel())
        v_probs = np.concatenate(v_probs) if v_probs else np.array([])
        v_gts   = np.concatenate(v_gts) if v_gts else np.array([])
        try:
            v_auc = roc_auc_score(v_gts, v_probs) if len(v_gts) else float("nan")
        except Exception:
            v_auc = float("nan")

        print(f"Epoch {ep:02d}/{EPOCHS} | train_loss={running/len(train_loader):.4f} | val_auc={v_auc:.4f}")

    # Save weights
    torch.save(model.state_dict(), "android_perm_mlp.pt")

    # -------- Test --------
    model.eval()
    with torch.no_grad():
        t_probs, t_gts = [], []
        for xb, yb in test_loader:
            p = model(xb.to(device)).cpu().numpy().ravel()
            t_probs.append(p)
            t_gts.append(yb.numpy().ravel())
    t_probs = np.concatenate(t_probs)
    t_gts   = np.concatenate(t_gts)
    t_pred  = (t_probs >= 0.5).astype("int32")

    print("\n=== TEST RESULTS ===")
    print("AUC      :", roc_auc_score(t_gts, t_probs))
    print("Accuracy :", accuracy_score(t_gts, t_pred))
    print("F1       :", f1_score(t_gts, t_pred))
    print(classification_report(t_gts, t_pred, digits=4))

    print("\nSaved:")
    print(" - android_perm_mlp.pt (weights)")
    print(" - android_perm_scaler.joblib (scaler)")

if __name__ == "__main__":
    main()
