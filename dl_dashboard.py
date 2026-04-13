"""
dl_dashboard.py — Streamlit Deep Learning Risk-Assessment Dashboard.

Launch:  streamlit run dl_dashboard.py
"""

import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
import torch.nn as nn

TIER_NAMES = {0: "Negligible", 1: "Low", 2: "Elevated", 3: "Critical"}

# ═══════════════════════════════════════════════════════════════════════
#  Model Definition (must match notebook)
# ═══════════════════════════════════════════════════════════════════════
class RiskClassifierNet(nn.Module):
    def __init__(self, n_features, n_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════
ARTIFACT_DIR = "dl_model_artifacts"
TIER_COLORS = {
    "Negligible": "#2ecc71",
    "Low":        "#f1c40f",
    "Elevated":   "#e67e22",
    "Critical":   "#e74c3c",
}

st.set_page_config(
    page_title="Deep Learning Risk Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0a0a1a 0%, #11112b 50%, #0d1f3c 100%); }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080818 0%, #101028 100%);
        border-right: 1px solid rgba(100, 60, 255, 0.2);
    }
    
    .stMetric {
        background: rgba(100, 60, 255, 0.05);
        border: 1px solid rgba(100, 60, 255, 0.15);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    h1 { 
        background: linear-gradient(90deg, #a855f7, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    h2, h3 { color: #c9d1d9 !important; }
    
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    .badge-negligible { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
    .badge-low { background: rgba(241,196,15,0.15); color: #f1c40f; border: 1px solid rgba(241,196,15,0.3); }
    .badge-elevated { background: rgba(230,126,34,0.15); color: #e67e22; border: 1px solid rgba(230,126,34,0.3); }
    .badge-critical { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }

    .glass-card {
        background: rgba(100, 60, 255, 0.03);
        border: 1px solid rgba(100, 60, 255, 0.12);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        margin-bottom: 16px;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(100, 60, 255, 0.03);
        border: 1px solid rgba(100, 60, 255, 0.12);
        border-radius: 12px;
    }

    .nn-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 11px;
        background: rgba(168, 85, 247, 0.15);
        color: #a855f7;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Load artifacts
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    device = torch.device("cpu")

    checkpoint = torch.load(
        os.path.join(ARTIFACT_DIR, "dl_risk_model.pth"),
        map_location=device, weights_only=False,
    )
    n_features = checkpoint["n_features"]
    n_classes = checkpoint["n_classes"]

    model = RiskClassifierNet(n_features, n_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    importance = pd.read_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"))

    with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    feature_names = pd.read_csv(
        os.path.join(ARTIFACT_DIR, "feature_names.csv"), header=None
    )[0].tolist()

    action_log = []
    action_path = os.path.join(ARTIFACT_DIR, "action_log.json")
    if os.path.exists(action_path):
        with open(action_path) as f:
            action_log = json.load(f)

    history = {}
    hist_path = os.path.join(ARTIFACT_DIR, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    return model, scaler, importance, metrics, feature_names, action_log, history


try:
    model, scaler, importance, metrics, feature_names, action_log, history = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(
        f"⚠ Could not load DL model artifacts. "
        f"Run the `deep_learning_model.ipynb` notebook first.\n\n{e}"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🧠 DL Risk Engine")
    st.markdown("---")
    st.markdown("### Risk Tier Legend")
    for name, color in TIER_COLORS.items():
        st.markdown(
            f'<span class="risk-badge badge-{name.lower()}">{name}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown(
        f'<span class="nn-badge">PyTorch</span>'
        f'<span class="nn-badge">Neural Network</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"- **Accuracy**: `{metrics['accuracy']:.4f}`")
    st.markdown(f"- **F1 Score**: `{metrics['f1_weighted']:.4f}`")
    st.markdown(f"- **Parameters**: `{metrics.get('total_parameters', 'N/A'):,}`")
    st.markdown(f"- **Epochs**: `{metrics.get('n_epochs_trained', 'N/A')}`")
    st.markdown(f"- **Train samples**: `{metrics['n_train']:,}`")
    st.markdown(f"- **Test samples**: `{metrics['n_test']:,}`")
    st.markdown(f"- **Features**: `{metrics['n_features']}`")
    st.markdown(f"- **Device**: `{metrics.get('device', 'cpu')}`")
    st.markdown(f"- **Trained**: `{metrics['trained_at'][:19]}`")
    st.markdown("---")
    st.markdown(
        "🔒 **Privacy**: Zero PII access.  \n"
        "All flows identified by SHA-256 hash."
    )

# ═══════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 Deep Learning Risk-Assessment Engine")
st.markdown(
    "*PyTorch neural network classifier with automated response logic  •  "
    "Non-payload metadata analysis  •  Zero PII access*"
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  KPI Cards
# ═══════════════════════════════════════════════════════════════════════
cm = np.array(metrics["confusion_matrix"])
tier_totals = cm.sum(axis=1)

col1, col2, col3, col4 = st.columns(4)
kpi_data = [
    ("Negligible", tier_totals[0] if len(tier_totals) > 0 else 0, "#2ecc71", "✅"),
    ("Low",        tier_totals[1] if len(tier_totals) > 1 else 0, "#f1c40f", "⚠️"),
    ("Elevated",   tier_totals[2] if len(tier_totals) > 2 else 0, "#e67e22", "🔶"),
    ("Critical",   tier_totals[3] if len(tier_totals) > 3 else 0, "#e74c3c", "🛑"),
]
for col, (name, count, color, icon) in zip([col1, col2, col3, col4], kpi_data):
    with col:
        st.metric(f"{icon} {name}", f"{int(count):,}", delta=None)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  Row 1: Training Curves (unique to DL dashboard)
# ═══════════════════════════════════════════════════════════════════════
if history:
    st.markdown("### 📉 Training Curves")
    tc1, tc2 = st.columns(2)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    with tc1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["train_loss"],
            name="Train Loss", mode="lines+markers",
            line=dict(color="#a855f7", width=2),
            marker=dict(size=5),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            name="Val Loss", mode="lines+markers",
            line=dict(color="#06b6d4", width=2),
            marker=dict(size=5),
        ))
        fig_loss.update_layout(
            title="Loss per Epoch",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Loss", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=40, b=40),
            height=350,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with tc2:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=history["train_acc"],
            name="Train Accuracy", mode="lines+markers",
            line=dict(color="#a855f7", width=2),
            marker=dict(size=5),
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=history["val_acc"],
            name="Val Accuracy", mode="lines+markers",
            line=dict(color="#06b6d4", width=2),
            marker=dict(size=5),
        ))
        fig_acc.update_layout(
            title="Accuracy per Epoch",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Accuracy", range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=40, b=40),
            height=350,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  Row 2: Risk Distribution + Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown("### 📊 Risk Tier Distribution")
    tier_labels = [TIER_NAMES[i] for i in range(len(tier_totals))]
    colors = [TIER_COLORS[n] for n in tier_labels]
    fig_donut = go.Figure(go.Pie(
        labels=tier_labels,
        values=tier_totals.tolist(),
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0a0a1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with r1c2:
    st.markdown("### 🔥 Confusion Matrix")
    tier_labels_short = [TIER_NAMES[i] for i in range(cm.shape[0])]
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=tier_labels_short,
        y=tier_labels_short,
        color_continuous_scale=["#080818", "#1a1a3e", "#a855f7", "#e74c3c"],
        text_auto=True,
    )
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  Row 3: Feature Importance + Per-Class Metrics
# ═══════════════════════════════════════════════════════════════════════
r2c1, r2c2 = st.columns([3, 2])

with r2c1:
    st.markdown("### 🏆 Top-20 Feature Importance (Gradient Attribution)")
    top20 = importance.head(20).iloc[::-1]
    fig_imp = go.Figure(go.Bar(
        y=top20["feature"],
        x=top20["importance"],
        orientation="h",
        marker=dict(
            color=top20["importance"],
            colorscale=[[0, "#11112b"], [0.5, "#a855f7"], [1, "#06b6d4"]],
        ),
        text=top20["importance"].round(4),
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=60, t=20, b=40),
        height=550,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with r2c2:
    st.markdown("### 📈 Per-Class Performance")
    report = metrics.get("report", {})
    perf_data = []
    for tier_name in ["Negligible", "Low", "Elevated", "Critical"]:
        if tier_name.lower() in {k.lower() for k in report}:
            key = next(k for k in report if k.lower() == tier_name.lower())
            r = report[key]
            perf_data.append({
                "Tier": tier_name,
                "Precision": round(r.get("precision", 0), 4),
                "Recall": round(r.get("recall", 0), 4),
                "F1-Score": round(r.get("f1-score", 0), 4),
                "Support": int(r.get("support", 0)),
            })
    if perf_data:
        perf_df = pd.DataFrame(perf_data)

        fig_perf = go.Figure()
        for metric, color in [("Precision", "#06b6d4"), ("Recall", "#a855f7"), ("F1-Score", "#2ecc71")]:
            fig_perf.add_trace(go.Bar(
                name=metric,
                x=perf_df["Tier"],
                y=perf_df[metric],
                marker_color=color,
                text=perf_df[metric],
                textposition="outside",
                textfont=dict(size=11),
            ))
        fig_perf.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            yaxis=dict(range=[0, 1.15], gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=20, b=40),
            height=350,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        st.dataframe(
            perf_df.style.format({
                "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  Row 4: Network Architecture Visualization
# ═══════════════════════════════════════════════════════════════════════
st.markdown("### 🏗️ Neural Network Architecture")

arch_cols = st.columns(5)
layers = [
    ("Input", f"{metrics['n_features']}", "#6366f1"),
    ("Hidden 1", "256 → BN → ReLU → Drop(0.3)", "#a855f7"),
    ("Hidden 2", "128 → BN → ReLU → Drop(0.3)", "#8b5cf6"),
    ("Hidden 3", "64 → BN → ReLU → Drop(0.2)", "#7c3aed"),
    ("Output", "4 classes", "#06b6d4"),
]
for col, (name, desc, color) in zip(arch_cols, layers):
    with col:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; min-height:100px; border-color:{color}40;">
            <h4 style="color:{color}; margin:0;">{name}</h4>
            <p style="color:#8b949e; font-size:12px; margin:8px 0 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  Row 5: Action Logic Summary + Action Log
# ═══════════════════════════════════════════════════════════════════════
st.markdown("### ⚡ Automated Response Actions")

a1, a2, a3, a4 = st.columns(4)
action_info = [
    ("✅ LOG ONLY",       "Negligible traffic",       "badge-negligible"),
    ("⚠️ SOC ALERT",      "Reconnaissance detected",  "badge-low"),
    ("🔶 RE-AUTH",        "Force 802.1X re-auth",     "badge-elevated"),
    ("🛑 MAC ISOLATION",  "Block + port shutdown",    "badge-critical"),
]
for col, (title, desc, badge) in zip([a1, a2, a3, a4], action_info):
    with col:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; min-height:120px;">
            <h4 style="margin:0;">{title}</h4>
            <p style="color:#8b949e; font-size:13px; margin:8px 0 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

if action_log:
    with st.expander("📋 Action Log (sample)", expanded=False):
        action_df = pd.DataFrame(action_log[:200])
        display_cols = ["flow_id", "risk_label", "action", "description", "confidence"]
        display_cols = [c for c in display_cols if c in action_df.columns]
        st.dataframe(action_df[display_cols], use_container_width=True, hide_index=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  Row 6: Live Classifier Simulator
# ═══════════════════════════════════════════════════════════════════════
st.markdown("### 🧪 Live Classification Simulator")
st.markdown("*Enter flow metadata values to predict the risk tier using the neural network.*")

with st.expander("▸ Open Simulator", expanded=False):
    sim_cols = st.columns(4)
    input_vals = {}
    for i, feat in enumerate(feature_names[:20]):
        col_idx = i % 4
        with sim_cols[col_idx]:
            input_vals[feat] = st.number_input(
                feat[:30], value=0.0, format="%.2f", key=f"sim_{feat}"
            )

    if st.button("🔍 Classify Flow", type="primary"):
        input_vec = np.array([[input_vals.get(f, 0.0) for f in feature_names]])
        input_scaled = scaler.transform(input_vec)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            output = model(input_tensor)
            proba = torch.softmax(output, dim=1).numpy()[0]
            pred = int(output.argmax(dim=1).item())

        pred_name = TIER_NAMES.get(pred, "Unknown")
        pred_color = TIER_COLORS.get(pred_name, "#fff")

        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <h2 style="color:{pred_color}; margin:0;">Risk Tier: {pred_name}</h2>
            <p style="color:#8b949e;">Confidence: {proba[pred]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        fig_proba = go.Figure(go.Bar(
            x=[TIER_NAMES[i] for i in range(len(proba))],
            y=proba,
            marker_color=[TIER_COLORS[TIER_NAMES[i]] for i in range(len(proba))],
            text=[f"{p:.1%}" for p in proba],
            textposition="outside",
        ))
        fig_proba.update_layout(
            title="Prediction Probabilities",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            yaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.05)"),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_proba, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#484f58; font-size:12px;">'
    '🧠 Deep Learning Risk-Assessment Engine  •  '
    'PyTorch Neural Network  •  '
    'Non-Payload Metadata Analysis  •  '
    'Zero PII Access  •  Privacy-Compliant'
    '</p>',
    unsafe_allow_html=True,
)
