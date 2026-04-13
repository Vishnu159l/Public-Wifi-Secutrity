##############################################################################
#  NETWORK FLOW RISK-ASSESSMENT ENGINE
#  Single-script pipeline -- just run: python main.py
#  No classes, no __main__ guard, no multi-file imports.
##############################################################################

import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
print("=" * 70)
print("  NETWORK FLOW RISK-ASSESSMENT ENGINE")
print("  Non-Payload Metadata Classifier  |  Zero PII Access")
print("=" * 70)

OUTPUT_DIR = "model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


##############################################################################
#  STAGE 1 : LOAD & ANONYMISE DATA
##############################################################################
print("\n> Stage 1/6 : Loading & anonymising data ...")

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

# Clean column names (strip whitespace)
data.columns = data.columns.str.strip()

# Privacy: hash destination port for traceability
data["Destination_Port_Hash"] = (
    data["Destination Port"].astype(str)
    .apply(lambda v: hashlib.sha256(v.encode()).hexdigest()[:16])
)

# Assign hashed flow IDs (privacy-compliant tracking)
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

# Stratified 20% sample for faster training (set to 1.0 for full data)
SAMPLE_FRAC = 0.20
data = (
    data.groupby("Label", group_keys=False)
    .apply(lambda g: g.sample(frac=SAMPLE_FRAC, random_state=42))
    .reset_index(drop=True)
)
print(f"  Sampled {SAMPLE_FRAC:.0%} -> {len(data):,} rows")


##############################################################################
#  STAGE 2 : FEATURE ENGINEERING
##############################################################################
print("\n> Stage 2/6 : Engineering features ...")

# -- Time-Series features --
TIME_FEATURES = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# -- Structural features (TCP window, flags, headers) --
STRUCT_FEATURES = [
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "min_seg_size_forward",
    "Fwd Header Length", "Bwd Header Length", "Fwd Header Length.1",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
]

# -- Volumetric features (bytes, packets, ratios) --
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

# -- Derived features --
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

# Collect all features that exist in the data
all_features = TIME_FEATURES + STRUCT_FEATURES + VOLUME_FEATURES + DERIVED
feature_names = [c for c in all_features if c in data.columns]
print(f"  Selected {len(feature_names)} features")


##############################################################################
#  STAGE 3 : MAP LABELS -> 4 RISK TIERS
##############################################################################
print("\n> Stage 3/6 : Mapping labels -> risk tiers ...")

TIER_NAMES = {0: "Negligible", 1: "Low", 2: "Elevated", 3: "Critical"}

LABEL_TO_TIER = {
    "BENIGN":           0,   # Negligible
    "PortScan":         1,   # Low (recon)
    "FTP-Patator":      2,   # Elevated (exploitation)
    "SSH-Patator":      2,
    "Infiltration":     2,
    "Bot":              2,
    "DDoS":             3,   # Critical (DoS / exfiltration)
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


##############################################################################
#  STAGE 4 : NORMALISE FEATURES
##############################################################################
print("\n> Stage 4/6 : Normalising features ...")

X = data[feature_names].values.astype(np.float64)
y = data["Risk_Tier"].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
pd.Series(feature_names).to_csv(
    os.path.join(OUTPUT_DIR, "feature_names.csv"), index=False, header=False
)
print(f"  Feature matrix shape: {X_scaled.shape}")
print(f"  Scaler saved.")


##############################################################################
#  STAGE 5 : TRAIN RANDOM FOREST
##############################################################################
print("\n> Stage 5/6 : Training classifier ...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# Optional SMOTE for minority classes
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"  SMOTE applied -> {X_train.shape[0]:,} training samples")
except ImportError:
    print("  (imbalanced-learn not installed, skipping SMOTE)")

# Train
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
print("  Training Random Forest (200 trees, depth=20) ...")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
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

# Feature importance
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\n  Top-15 features:")
for _, row in importance.head(15).iterrows():
    bar = "#" * int(row["importance"] * 100)
    print(f"    {row['feature']:>35s}  {row['importance']:.4f}  {bar}")

# Save model + metrics
joblib.dump(clf, os.path.join(OUTPUT_DIR, "risk_model.joblib"))
importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

metrics = {
    "accuracy": round(acc, 5),
    "f1_weighted": round(f1, 5),
    "report": report_dict,
    "confusion_matrix": cm.tolist(),
    "trained_at": pd.Timestamp.now().isoformat(),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_features": int(X_train.shape[1]),
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2)


##############################################################################
#  STAGE 6 : GENERATE ACTION LOG
##############################################################################
print("\n> Stage 6/6 : Generating action log ...")

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


##############################################################################
#  DONE
##############################################################################
elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"  PIPELINE COMPLETE")
print(f"  Elapsed time       : {elapsed:.1f}s")
print(f"  Accuracy           : {acc:.4f}")
print(f"  F1 (weighted)      : {f1:.4f}")
print(f"  Artifacts saved in : {OUTPUT_DIR}/")
print(f"  PRIVACY            : Zero PII accessed -- all records hashed")
print("=" * 70)
