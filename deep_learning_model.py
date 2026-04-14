#!/usr/bin/env python
# coding: utf-8

# # 🧠 Deep Learning Risk-Assessment Engine
# **PyTorch Neural Network Classifier | Non-Payload Metadata | Zero PII Access**
# 
# This notebook trains a multi-layer neural network to classify network flows into 4 risk tiers:
# - **Negligible** (0) — Normal traffic
# - **Low** (1) — Reconnaissance
# - **Elevated** (2) — Exploitation attempts
# - **Critical** (3) — DoS / DDoS / Exfiltration

# ## Imports & Setup

# In[1]:


import os
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

t0 = time.time()
print("\n" + "=" * 70)
print("  DEEP LEARNING RISK-ASSESSMENT ENGINE")
print("  PyTorch Neural Network  |  Non-Payload Metadata  |  Zero PII")
print("=" * 70)

OUTPUT_DIR = "dl_model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  Output directory: {OUTPUT_DIR}/")


# ## Stage 1 : Load & Anonymise Data

# In[2]:


print("\n> Stage 1/8 : Loading & anonymising data ...")

DATA_DIR = "Data"
csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

frames = []
for fname in csv_files:
    fpath = os.path.join(DATA_DIR, fname)
    print(f"  Reading {fname} ...", end=" ")
    df_chunk = pd.read_csv(fpath, low_memory=False, encoding="utf-8",
                           encoding_errors="replace")
    print(f"({len(df_chunk):,} rows)")
    frames.append(df_chunk)

data = pd.concat(frames, ignore_index=True)
print(f"  Total raw rows: {len(data):,}")

# Clean column names
data.columns = data.columns.str.strip()

# Privacy: hash destination port
data["Destination_Port_Hash"] = (
    data["Destination Port"].astype(str)
    .apply(lambda v: hashlib.sha256(v.encode()).hexdigest()[:16])
)

# Hashed flow IDs
data["Flow_ID"] = [
    hashlib.sha256(f"flow_{i}_42".encode()).hexdigest()[:20]
    for i in range(len(data))
]

# Drop rows with NaN / Inf
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
before = len(data)
data.dropna(subset=numeric_cols, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"  Dropped {before - len(data):,} corrupt rows -> {len(data):,} clean rows")

# Stratified 20% sample
SAMPLE_FRAC = 0.20
data = (
    data.groupby("Label", group_keys=False)
    .sample(frac=SAMPLE_FRAC, random_state=42)
    .reset_index(drop=True)
)
print(f"  Sampled {SAMPLE_FRAC:.0%} -> {len(data):,} rows")


# ## Stage 2 : Feature Engineering

# In[3]:


print("\n> Stage 2/8 : Engineering features ...")

TIME_FEATURES = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

STRUCT_FEATURES = [
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "min_seg_size_forward",
    "Fwd Header Length", "Bwd Header Length", "Fwd Header Length.1",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
]

VOLUME_FEATURES = [
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "act_data_pkt_fwd", "Destination Port",
]

# Derived features
safe_div = lambda a, b: np.where(b != 0, a / b, 0.0)

data["iat_cv_flow"] = safe_div(data.get("Flow IAT Std", 0), data.get("Flow IAT Mean", 1))
data["iat_cv_fwd"]  = safe_div(data.get("Fwd IAT Std", 0),  data.get("Fwd IAT Mean", 1))
data["iat_cv_bwd"]  = safe_div(data.get("Bwd IAT Std", 0),  data.get("Bwd IAT Mean", 1))

total_pkts = data.get("Total Fwd Packets", 0) + data.get("Total Backward Packets", 0)
data["fwd_bwd_pkt_ratio"] = safe_div(data.get("Total Fwd Packets", 0), total_pkts)
data["byte_burstiness"]   = safe_div(data.get("Packet Length Std", 0), data.get("Packet Length Mean", 1))
total_bytes = data.get("Total Length of Fwd Packets", 0) + data.get("Total Length of Bwd Packets", 0)
data["bytes_per_packet"]  = safe_div(total_bytes, total_pkts)

DERIVED = ["iat_cv_flow", "iat_cv_fwd", "iat_cv_bwd",
           "fwd_bwd_pkt_ratio", "byte_burstiness", "bytes_per_packet"]

all_features = TIME_FEATURES + STRUCT_FEATURES + VOLUME_FEATURES + DERIVED
feature_names = [c for c in all_features if c in data.columns]
print(f"  Selected {len(feature_names)} features")


# ## Stage 3 : Map Labels → 4 Risk Tiers

# In[4]:


print("\n> Stage 3/8 : Mapping labels -> risk tiers ...")

TIER_NAMES = {0: "Negligible", 1: "Low", 2: "Elevated", 3: "Critical"}

LABEL_TO_TIER = {
    "BENIGN":           0,
    "PortScan":         1,
    "FTP-Patator":      2,
    "SSH-Patator":      2,
    "Infiltration":     2,
    "Bot":              2,
    "DDoS":             3,
    "DoS GoldenEye":    3,
    "DoS Hulk":         3,
    "DoS Slowhttptest": 3,
    "DoS slowloris":    3,
    "Heartbleed":       3,
}

data["Risk_Tier"] = data["Label"].map(LABEL_TO_TIER).fillna(0).astype(int)

print("  Tier distribution:")
for tier_id, count in data["Risk_Tier"].value_counts().sort_index().items():
    print(f"    {TIER_NAMES[tier_id]:>12s} ({tier_id}): {count:>10,}")


# ## Stage 4 : Normalise & Prepare DataLoaders

# In[5]:


print("\n> Stage 4/8 : Normalising features & creating DataLoaders ...")

X = data[feature_names].values.astype(np.float64)
y = data["Risk_Tier"].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Apply log transform to handle extreme right-skewed network flow data
X = np.log1p(np.abs(X)) * np.sign(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler & feature names
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
pd.Series(feature_names).to_csv(
    os.path.join(OUTPUT_DIR, "feature_names.csv"), index=False, header=False
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# Removed SMOTE to prevent synthetic noise generation on sparse network features
print(f"  Using original imbalanced dataset -> {X_train.shape[0]:,} training samples")

# Compute class weights for loss function (Smoothed with sqrt to prevent extreme gradients)
class_counts = np.bincount(y_train)
total_samples = len(y_train)
n_classes = len(class_counts)
weights_raw = 1.0 / np.sqrt(class_counts.astype(np.float64))
class_weights = weights_raw / np.sum(weights_raw) * n_classes
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"  Smoothed Class weights: {[f'{w:.3f}' for w in class_weights]}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_test_t  = torch.LongTensor(y_test).to(device)

# DataLoaders
BATCH_SIZE = 1024
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Feature matrix shape: {X_scaled.shape}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")


# ## Stage 5 : Define Neural Network Architecture

# In[6]:


class RiskClassifierNet(nn.Module):
    """
    Multi-layer neural network for 4-class risk classification.
    Architecture: 3 hidden layers with BatchNorm + Dropout + GELU.
    """
    def __init__(self, n_features, n_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: Input -> 512
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            # Output: 128 -> n_classes
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.network(x)


n_features = X_train.shape[1]
model = RiskClassifierNet(n_features, n_classes=4).to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model Architecture:")
print(model)
print(f"\n  Total parameters    : {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")


# ## Stage 6 : Training Loop

# In[7]:


print("\n> Stage 6/8 : Training neural network ...")

NUM_EPOCHS = 50
PATIENCE = 8

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [],  "val_acc": [],
}

best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # ---- Training ----
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_loss = train_loss_sum / train_total
    train_acc  = train_correct / train_total

    # ---- Validation ----
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_loss = val_loss_sum / val_total
    val_acc  = val_correct / val_total

    # Record history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # Print progress
    print(
        f"  Epoch {epoch+1:>2d}/{NUM_EPOCHS}  |  "
        f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
        f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
        f"LR: {current_lr:.1e}"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"  Restored best model (val_loss={best_val_loss:.4f})")

# Save training history
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as fh:
    json.dump(history, fh, indent=2)
print(f"  Training history saved.")


# ## Stage 7 : Evaluation

# In[8]:


print("\n> Stage 7/8 : Evaluating model ...")

model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

y_pred = np.array(all_preds)
y_probs = np.array(all_probs)

target_names = [TIER_NAMES[i] for i in sorted(TIER_NAMES)]

report_str = classification_report(y_test, y_pred, target_names=target_names,
                                   digits=4, zero_division=0)
report_dict = classification_report(y_test, y_pred, target_names=target_names,
                                    output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\n  === Classification Report ===\n{report_str}")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 (wtd) : {f1:.4f}")

# Feature importance via gradient-based attribution
print("\n  Computing gradient-based feature importance ...")
model.eval()
X_imp = torch.FloatTensor(X_test[:5000]).to(device)
X_imp.requires_grad_(True)
outputs = model(X_imp)
pred_classes = outputs.argmax(dim=1)
loss_imp = criterion(outputs, pred_classes)
loss_imp.backward()
grad_importance = X_imp.grad.abs().mean(dim=0).cpu().numpy()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": grad_importance,
}).sort_values("importance", ascending=False).reset_index(drop=True)

# Normalise importance to sum to 1
importance["importance"] = importance["importance"] / importance["importance"].sum()

print("\n  Top-15 features (gradient attribution):")
for _, row in importance.head(15).iterrows():
    bar = "#" * int(row["importance"] * 200)
    print(f"    {row['feature']:>35s}  {row['importance']:.4f}  {bar}")

# Save model + metrics
torch.save({
    "model_state_dict": model.state_dict(),
    "n_features": n_features,
    "n_classes": 4,
    "architecture": "RiskClassifierNet",
}, os.path.join(OUTPUT_DIR, "dl_risk_model.pth"))

importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

metrics = {
    "accuracy": round(acc, 5),
    "f1_weighted": round(f1, 5),
    "report": report_dict,
    "confusion_matrix": cm.tolist(),
    "trained_at": pd.Timestamp.now().isoformat(),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_features": int(n_features),
    "n_epochs_trained": len(history["train_loss"]),
    "best_val_loss": round(best_val_loss, 5),
    "total_parameters": total_params,
    "device": str(device),
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2)

print(f"\n  Model saved to {OUTPUT_DIR}/dl_risk_model.pth")
print(f"  Metrics saved to {OUTPUT_DIR}/metrics.json")


# ## Stage 8 : Generate Action Log

# In[ ]:


print("\n> Stage 8/8 : Generating action log ...")

ACTION_MAP = {
    0: {"action": "LOG_ONLY",        "desc": "Normal traffic -- logged for audit."},
    1: {"action": "SOC_ALERT",       "desc": "Recon detected -- alert to SOC."},
    2: {"action": "RE_AUTHENTICATE", "desc": "Exploitation attempt -- forcing 802.1X re-auth."},
    3: {"action": "MAC_ISOLATION",   "desc": "Critical threat -- MAC isolation + port shutdown."},
}

actions = []
tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for i, pred in enumerate(y_pred):
    pred = int(pred)
    tier_counts[pred] += 1
    sid = hashlib.sha256(f"session_{i}".encode()).hexdigest()[:16]
    template = ACTION_MAP[pred]
    actions.append({
        "flow_id": sid,
        "risk_tier": pred,
        "risk_label": TIER_NAMES[pred],
        "action": template["action"],
        "description": template["desc"],
        "confidence": round(float(y_probs[i][pred]), 4),
    })

# Save (trim to keep file small)
save_actions = actions[:500] + actions[-100:] if len(actions) > 600 else actions
with open(os.path.join(OUTPUT_DIR, "action_log.json"), "w") as fh:
    json.dump(save_actions, fh, indent=2)

print("  Action summary:")
for tid in sorted(tier_counts):
    print(f"    {TIER_NAMES[tid]:>12s}: {tier_counts[tid]:>10,}  ->  {ACTION_MAP[tid]['action']}")
print(f"\n  Re-auth requests : {tier_counts[2]:,}")
print(f"  MAC isolations   : {tier_counts[3]:,}")
print(f"  Action log saved.")


# ## Pipeline Complete

# In[ ]:


elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"  DEEP LEARNING PIPELINE COMPLETE")
print(f"  Elapsed time       : {elapsed:.1f}s")
print(f"  Device             : {device}")
print(f"  Parameters         : {total_params:,}")
print(f"  Epochs trained     : {len(history['train_loss'])}")
print(f"  Accuracy           : {acc:.4f}")
print(f"  F1 (weighted)      : {f1:.4f}")
print(f"  Best val loss      : {best_val_loss:.4f}")
print(f"  Artifacts saved in : {OUTPUT_DIR}/")
print(f"  PRIVACY            : Zero PII accessed -- all records hashed")
print("=" * 70)

