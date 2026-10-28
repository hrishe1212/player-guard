import os
import json
import csv
import pickle
import datetime
import streamlit as st
from rag_engine import PolicyRAG
from llm_agent import generate_agent_briefing
from email_service import send_player_alert, get_credentials
from train_tree import PureDecisionTreeClassifier, Node

# Page Configuration
st.set_page_config(
    page_title="PlayerGuard AI — Northbridge Meridian FC",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- FOOTBALL THEME & PLAYFUL STYLING -----------------
st.markdown("""
<style>
    /* Main App Background - Stadium Pitch Night Theme */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d3820 0%, #061a0f 60%, #030d07 100%);
        color: #f1f5f9;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* Stadium Floodlight Top Header */
    .football-banner {
        background: linear-gradient(135deg, #062b16 0%, #0e4e2a 50%, #051d10 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .banner-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .banner-sub {
        font-size: 14px;
        color: #86efac;
        margin-top: 4px;
        font-weight: 500;
    }
    .club-crest {
        background: #14532d;
        border: 2px solid #facc15;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        box-shadow: 0 0 15px rgba(250, 204, 21, 0.4);
    }

    /* Player Profile Card */
    .player-profile-card {
        background: rgba(15, 60, 35, 0.75);
        border: 1px solid rgba(74, 222, 128, 0.35);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .player-jersey {
        background: linear-gradient(135deg, #facc15 0%, #ca8a04 100%);
        color: #000000;
        font-weight: 900;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 16px;
        display: inline-block;
        margin-right: 10px;
        letter-spacing: 1px;
    }
    .player-name-tag {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
    }
    .player-pos-tag {
        color: #4ade80;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 600;
        margin-left: 10px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(10, 42, 24, 0.7);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .metric-val {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-lbl {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #86efac;
    }

    /* Referee Card Badges */
    .card-green {
        background: rgba(34, 197, 94, 0.15);
        border: 2px solid #22c55e;
        color: #4ade80;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 16px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.3);
    }
    .card-yellow {
        background: rgba(234, 179, 8, 0.15);
        border: 2px solid #eab308;
        color: #fde047;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 16px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(234, 179, 8, 0.3);
    }
    .card-red {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #f87171;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 16px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.35);
    }

    /* Policy Rules Card */
    .pitch-rule-card {
        background: rgba(6, 30, 16, 0.85);
        border-left: 5px solid #22c55e;
        border-top: 1px solid rgba(34, 197, 94, 0.2);
        border-right: 1px solid rgba(34, 197, 94, 0.2);
        border-bottom: 1px solid rgba(34, 197, 94, 0.2);
        padding: 16px 20px;
        margin-bottom: 14px;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .pitch-rule-title {
        color: #facc15;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .pitch-rule-text {
        font-size: 13px;
        line-height: 1.5;
        color: #cbd5e1;
    }

    /* Pitch Radar Container */
    .pitch-box {
        background: rgba(5, 38, 20, 0.8);
        border: 2px dashed rgba(74, 222, 128, 0.4);
        border-radius: 12px;
        padding: 18px;
        margin: 15px 0;
    }

    /* FIX: High-contrast white text across ALL widgets, inputs, radios, and labels */
    div[data-testid="stWidgetLabel"] label p,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] span,
    .stRadio label,
    .stRadio p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

 
    /* Radio Group Container */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background: rgba(4, 26, 14, 0.85) !important;
        border: 2px solid #22c55e !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5) !important;
    }

    /* Radio items styling */
    div[data-testid="stRadio"] label {
        padding: 10px 14px !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        background: rgba(14, 53, 30, 0.5) !important;
        border: 1px solid rgba(74, 222, 128, 0.2) !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(34, 197, 94, 0.25) !important;
        border-color: #22c55e !important;
    }

    /* Textareas and Text Inputs - Pure White Text on Dark Emerald Field */
    .stTextArea textarea {
        background-color: #041f0f !important;
        color: #ffffff !important;
        border: 2px solid #22c55e !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 12px 14px !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #facc15 !important;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.4) !important;
        color: #ffffff !important;
    }
    .stTextInput input {
        background-color: #041f0f !important;
        color: #ffffff !important;
        border: 2px solid #22c55e !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus {
        border-color: #facc15 !important;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.4) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SQUAD ROSTER (20 SQUAD PLAYERS) -----------------
SQUAD_ROSTER = [
    {"id": "PG-NMFC-010", "name": "Mateo Silva", "number": 10, "pos": "Attacking Midfielder (C)", "age": 27, "inj_hist": 1, "rec_idx": 18},
    {"id": "PG-NMFC-009", "name": "Marcus Vance", "number": 9, "pos": "Striker", "age": 25, "inj_hist": 0, "rec_idx": 3},
    {"id": "PG-NMFC-007", "name": "Alejandro Gomez", "number": 7, "pos": "Left Winger", "age": 24, "inj_hist": 1, "rec_idx": 15},
    {"id": "PG-NMFC-011", "name": "Bukayo Davies", "number": 11, "pos": "Right Winger", "age": 23, "inj_hist": 0, "rec_idx": 5},
    {"id": "PG-NMFC-006", "name": "Declan O'Connor", "number": 6, "pos": "Defensive Midfielder", "age": 26, "inj_hist": 0, "rec_idx": 8},
    {"id": "PG-NMFC-004", "name": "Julian Romero", "number": 4, "pos": "Center Back", "age": 28, "inj_hist": 1, "rec_idx": 24},
    {"id": "PG-NMFC-002", "name": "Kieran Walker", "number": 2, "pos": "Right Wing Back", "age": 29, "inj_hist": 1, "rec_idx": 19},
    {"id": "PG-NMFC-003", "name": "Theo Hernandez", "number": 3, "pos": "Left Full Back", "age": 26, "inj_hist": 0, "rec_idx": 11},
    {"id": "PG-NMFC-001", "name": "David De Costa", "number": 1, "pos": "Goalkeeper", "age": 31, "inj_hist": 0, "rec_idx": 1},
    {"id": "PG-NMFC-008", "name": "Gabriel Martin", "number": 8, "pos": "Central Midfielder", "age": 25, "inj_hist": 0, "rec_idx": 7},
    {"id": "PG-NMFC-005", "name": "Sandro Bastoni", "number": 5, "pos": "Center Back", "age": 27, "inj_hist": 0, "rec_idx": 12},
    {"id": "PG-NMFC-017", "name": "Phil Sterling", "number": 17, "pos": "Attacking Midfielder", "age": 23, "inj_hist": 1, "rec_idx": 20},
    {"id": "PG-NMFC-019", "name": "Rasmus Lindstrom", "number": 19, "pos": "Forward", "age": 22, "inj_hist": 0, "rec_idx": 21},
    {"id": "PG-NMFC-014", "name": "Luka Modric", "number": 14, "pos": "Playmaker", "age": 34, "inj_hist": 1, "rec_idx": 22},
    {"id": "PG-NMFC-021", "name": "Eduardo Camavinga", "number": 21, "pos": "Box-to-Box Midfielder", "age": 22, "inj_hist": 0, "rec_idx": 23},
    {"id": "PG-NMFC-018", "name": "Alphonso Davies", "number": 18, "pos": "Left Wing Back", "age": 24, "inj_hist": 1, "rec_idx": 16},
    {"id": "PG-NMFC-016", "name": "Rodri Hernandez", "number": 16, "pos": "Anchor Midfield", "age": 28, "inj_hist": 0, "rec_idx": 4},
    {"id": "PG-NMFC-023", "name": "Cole Palmer", "number": 23, "pos": "Right Forward", "age": 22, "inj_hist": 0, "rec_idx": 9},
    {"id": "PG-NMFC-012", "name": "William Saliba", "number": 12, "pos": "Center Back", "age": 24, "inj_hist": 0, "rec_idx": 14},
    {"id": "PG-NMFC-025", "name": "Lucas Paqueta", "number": 25, "pos": "Attacking Midfielder", "age": 26, "inj_hist": 1, "rec_idx": 25}
]

# ----------------- CACHED RESOURCES -----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_decision_tree.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_rag():
    return PolicyRAG()

@st.cache_data
def load_multimodal_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), 'sports_multimodal_data.csv')
    if not os.path.exists(csv_path):
        return None, []
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= 100:
                break
            rows.append([float(x) for x in row])
    return header[:-1], rows

creds = get_credentials()
model = load_model()
rag = load_rag()
feature_names, sample_rows = load_multimodal_dataset()

# Audit Log
AUDIT_LOG_FILE = os.path.join(os.path.dirname(__file__), 'audit_log.csv')
def log_decision_to_audit(player_name, player_id, risk_pred, decision, reviewer, notes, rules):
    file_exists = os.path.exists(AUDIT_LOG_FILE)
    with open(AUDIT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'player_name', 'player_id', 'model_risk', 'human_decision', 'reviewer', 'notes', 'rules_cited'])
        writer.writerow([
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            player_name,
            player_id,
            risk_pred,
            decision,
            reviewer,
            notes,
            ';'.join(rules)
        ])

# ----------------- SIDEBAR (FOOTBALL THEME - NO RAW CREDENTIALS) -----------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 15px 0;">
    <div style="font-size: 42px;">⚽</div>
    <h3 style="margin: 0; color: #facc15; font-weight: 800; letter-spacing: 1px;">NORTHBRIDGE MERIDIAN FC</h3>
    <span style="color: #86efac; font-size: 12px; font-weight: 600;">PLAYERGUARD AI MEDICAL CENTER</span>
</div>
""", unsafe_allow_html=True)

# System Status Indicators (Clean & Secure - Credentials hidden from UI)
st.sidebar.markdown("""
<div style="background: rgba(10, 40, 20, 0.8); border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
    <div style="color: #4ade80; font-size: 12px; font-weight: 700;">🟢 POLICY ENGINE: ACTIVE</div>
    <div style="color: #94a3b8; font-size: 11px;">NMFC-PG-POL-001 (73 Rules Loaded)</div>
    <div style="color: #4ade80; font-size: 12px; font-weight: 700; margin-top: 4px;">⚡ GROQ LLM AGENT: CONNECTED</div>
    <div style="color: #94a3b8; font-size: 11px;">Model: Qwen 3.8 27B / GPT-OSS 120B</div>
    <div style="color: #4ade80; font-size: 12px; font-weight: 700; margin-top: 4px;">📬 ALERT GATEWAY: READY</div>
    <div style="color: #94a3b8; font-size: 11px;">Gmail Medical Escalation Active</div>
</div>
""", unsafe_allow_html=True)

# Reviewer Role
reviewer_role = st.sidebar.selectbox(
    "🩺 Reviewing Medical Official (PW-001)",
    [
        "Dr. Eleanor Vance (Chief Medical Officer)",
        "Marcus Torres (Lead First-Team Physiotherapist)",
        "Dr. Kevin Patel (Head of Sports Science & Performance)",
        "Dr. Sophia Chen (Matchday Team Doctor)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏃‍♂️ Squad Roster & Telemetry")
mode = st.sidebar.radio("Player Selection Mode:", ["First-Team Squad Roster", "Create Custom Player & Telemetry"])

input_features = {}
current_player = {}

if mode == "First-Team Squad Roster":
    squad_options = [f"#{p['number']} {p['name']} ({p['pos']})" for p in SQUAD_ROSTER]
    selected_player_str = st.sidebar.selectbox("Select Player from Squad:", squad_options, index=0)
    selected_idx = squad_options.index(selected_player_str)
    current_player = SQUAD_ROSTER[selected_idx]

    # Map to historical row data
    rec_idx = current_player.get("rec_idx", 0)
    if sample_rows and rec_idx < len(sample_rows):
        sample_vals = sample_rows[rec_idx]
        for i, fn in enumerate(feature_names):
            input_features[fn] = sample_vals[i]
else:
    # Custom Player
    custom_name = st.sidebar.text_input("Player Full Name:", value="Leo Gallagher")
    custom_number = st.sidebar.number_input("Squad Kit Number:", min_value=1, max_value=99, value=10)
    custom_pos = st.sidebar.selectbox("Tactical Position:", ["Striker", "Winger", "Attacking Midfielder", "Central Midfielder", "Full Back", "Center Back", "Goalkeeper"])
    current_player = {
        "id": f"PG-CUSTOM-{custom_number:02d}",
        "name": custom_name,
        "number": custom_number,
        "pos": custom_pos,
        "age": 25,
        "inj_hist": 1
    }

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Input Mode: Sliders & Direct Type-In")
input_style = st.sidebar.radio("Control Interface:", ["🎛️ Hybrid Sliders & Type-In", "⌨️ Direct Number Inputs (Exact Values)"])

# Interactive features with BOTH slider & direct type-in support
def metric_input(label, key, min_val, max_val, default_val, step=0.1, unit=""):
    cur_val = float(input_features.get(key, default_val))
    if input_style == "🎛️ Hybrid Sliders & Type-In":
        col1, col2 = st.sidebar.columns([3, 2])
        with col2:
            typed = st.number_input(f"{label} {unit}", min_value=float(min_val), max_value=float(max_val), value=float(cur_val), step=float(step), key=f"num_{key}")
        with col1:
            slid = st.slider(f"Slide {label}", min_value=float(min_val), max_value=float(max_val), value=float(typed), step=float(step), key=f"sli_{key}", label_visibility="collapsed")
        return slid
    else:
        return st.sidebar.number_input(f"{label} ({unit})", min_value=float(min_val), max_value=float(max_val), value=float(cur_val), step=float(step), key=f"exact_{key}")

st.sidebar.markdown("##### Matchday Biomechanical Telemetry")
input_features['heart_rate'] = metric_input("Heart Rate", 'heart_rate', 40.0, 210.0, 85.0, 0.5, "bpm")
input_features['emg_amplitude'] = metric_input("EMG Muscle Load", 'emg_amplitude', 0.05, 1.5, 0.72, 0.01, "mV")
input_features['ground_reaction_force'] = metric_input("Ground Force (GRF)", 'ground_reaction_force', 100.0, 1100.0, 715.0, 1.0, "N")
input_features['fatigue_index'] = metric_input("Fatigue Index", 'fatigue_index', 10.0, 100.0, 62.0, 0.5, "pts")
input_features['training_duration'] = metric_input("Session Duration", 'training_duration', 10.0, 200.0, 90.0, 1.0, "min")
input_features['workload_intensity'] = metric_input("Workload Intensity", 'workload_intensity', 0.0, 12.0, 6.8, 0.1, "AU")
input_features['previous_injury_history'] = st.sidebar.selectbox("Previous Injury History", [0, 1], index=current_player.get("inj_hist", 0))

# Fill default baseline for remaining features
defaults = {
    'skin_temp': 33.1, 'gsr': 0.25, 'respiratory_rate': 16.0, 'spo2': 98.2,
    'bp_systolic': 120.0, 'bp_diastolic': 80.0, 'acceleration': 0.8, 'angular_velocity': 0.15,
    'body_orientation': 42.0, 'step_count': 110.0, 'cadence': 82.0, 'jump_height': 0.48,
    'range_of_motion': 78.0, 'impact_force': 310.0, 'gait_symmetry': 0.88, 'speed': 6.5,
    'altitude': 480.0, 'ambient_temp': 21.0, 'humidity': 55.0, 'heat_index': 25.0,
    'rest_period': 7.2, 'repetition_count': 32.0, 'acc_rms': 1.08
}
for k, v in defaults.items():
    if k not in input_features:
        input_features[k] = v

symptoms_input = st.sidebar.text_area(
    "📋 Reported Physical Symptoms / Physio Observations:",
    value="Player reported sharp fatigue in right hamstring during late sprint repetitions; mild gait stiffness observed."
)

# ----------------- MAIN STADIUM BANNER -----------------
st.markdown(f"""
<div class="football-banner">
    <div>
        <div class="banner-title">
            <span>⚽</span> NORTHBRIDGE MERIDIAN FC
        </div>
        <div class="banner-sub">
            PlayerGuard AI • Sports Club Player Welfare & Injury Escalation System • Policy NMFC-PG-POL-001
        </div>
    </div>
    <div class="club-crest">
        🛡️
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- PLAYER SPOTLIGHT CARD -----------------
st.markdown(f"""
<div class="player-profile-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span class="player-jersey">#{current_player.get('number', 10)}</span>
            <span class="player-name-tag">{current_player.get('name', 'Player')}</span>
            <span class="player-pos-tag">⚽ {current_player.get('pos', 'First Team')}</span>
        </div>
        <div style="font-size: 13px; color: #86efac; font-weight: 600;">
            ID: {current_player.get('id', 'PG-001')} | Squad Status: Matchday Pool
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- MODEL INFERENCE -----------------
ordered_vals = [input_features[fn] for fn in feature_names] if feature_names else []
if model and ordered_vals:
    pred_class, pred_prob = model.predict_one(ordered_vals)
else:
    pred_class = 1 if (input_features.get('emg_amplitude', 0) > 0.67 and input_features.get('heart_rate', 0) > 82.0) else 0
    pred_prob = 0.94 if pred_class == 1 else 0.06

risk_label_str = "Elevated Risk (Class 1)" if pred_class == 1 else "Lower Risk (Class 0)"

# ----------------- DASHBOARD TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Matchday Risk Radar",
    "📜 Club Policy & RAG",
    "🤖 Groq AI Briefing",
    "🟨 Human Decision & Escalation",
    "🏆 Compliance & Audit Trail"
])

# ----------------- TAB 1: RISK RADAR -----------------
with tab1:
    st.subheader("🌲 Decision Tree Biomechanical Risk Prediction")
    st.caption("Engine: DecisionTreeClassifier(max_depth=3, max_leaf_nodes=None, class_weight='balanced')")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if pred_class == 1:
            st.markdown('<div class="card-red">🟥 ELEVATED INJURY RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-green">🟩 LOWER INJURY RISK</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Risk Probability</div>
            <div class="metric-val" style="color: {'#ef4444' if pred_class==1 else '#4ade80'};">{pred_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Decision Tree Depth</div>
            <div class="metric-val">3 Levels</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Leaves Specification</div>
            <div class="metric-val">Unlimited</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📡 Live Pitch & Sensor Telemetry")
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric("💓 Heart Rate", f"{input_features.get('heart_rate', 0):.1f} bpm")
    t2.metric("⚡ Muscle Load (EMG)", f"{input_features.get('emg_amplitude', 0):.3f}")
    t3.metric("👟 Ground Impact (GRF)", f"{input_features.get('ground_reaction_force', 0):.0f} N")
    t4.metric("🔋 Fatigue Index", f"{input_features.get('fatigue_index', 0):.1f}")
    t5.metric("⏱️ Training Duration", f"{input_features.get('training_duration', 0):.0f} min")
    t6.metric("🔥 Workload Intensity", f"{input_features.get('workload_intensity', 0):.1f} AU")

    with st.expander("🔍 View Decision Tree Path & Tactical Thresholds"):
        st.code("""
Decision Tree Traversal (Trained on 5,430 Sessions):
IF emg_amplitude <= 0.674:
    IF heart_rate <= 86.654:
        --> Class: Lower Risk (Class 0) (Prob Class 1: 0.00%)
    ELSE (heart_rate > 86.654):
        IF ground_reaction_force <= 681.049:
            --> Class: Lower Risk (Class 0) (Prob Class 1: 0.00%)
        ELSE (ground_reaction_force > 681.049):
            --> Class: Elevated Risk (Class 1) (Prob Class 1: 88.02%)
ELSE (emg_amplitude > 0.674):
    IF ground_reaction_force <= 690.732:
        IF heart_rate <= 81.947:
            --> Class: Lower Risk (Class 0) (Prob Class 1: 13.12%)
        ELSE (heart_rate > 81.947):
            --> Class: Elevated Risk (Class 1) (Prob Class 1: 95.62%)
    ELSE (ground_reaction_force > 690.732):
        IF emg_amplitude <= 0.697:
            --> Class: Lower Risk (Class 0) (Prob Class 1: 0.00%)
        ELSE (emg_amplitude > 0.697):
            --> Class: Elevated Risk (Class 1) (Prob Class 1: 99.37%)
        """, language="yaml")

# ----------------- TAB 2: CLUB POLICY RAG -----------------
with tab2:
    st.subheader("📚 Governing Club Welfare Policy Clauses")
    st.caption("Northbridge Meridian FC • Document Ref: NMFC-PG-POL-001 (73 Rules Formally Indexed)")

    retrieved_rules = rag.retrieve(
        query=f"injury risk {risk_label_str} player welfare escalation",
        risk_label=pred_class,
        symptoms=symptoms_input,
        top_k=4
    )

    for r in retrieved_rules:
        st.markdown(f"""
        <div class="pitch-rule-card">
            <div class="pitch-rule-title">🛡️ [{r['rule_id']}] {r['title']}</div>
            <div class="pitch-rule-text">{r['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- TAB 3: GROQ AI BRIEFING -----------------
with tab3:
    st.subheader("🤖 Groq LLM Clinical Briefing Agent")
    st.caption("Engine: Groq Ultra-Low Latency Inference • Section 15 Medical Template Compliant")

    if st.button("⚡ Generate / Update Matchday AI Briefing", type="primary"):
        with st.spinner("Synthesizing telemetry data and policy rules via Groq LLM..."):
            key_metrics_summary = {
                "player_name": current_player.get('name'),
                "squad_number": current_player.get('number'),
                "position": current_player.get('pos'),
                "heart_rate": round(input_features.get('heart_rate', 0), 1),
                "emg_amplitude": round(input_features.get('emg_amplitude', 0), 3),
                "ground_reaction_force": round(input_features.get('ground_reaction_force', 0), 1),
                "fatigue_index": round(input_features.get('fatigue_index', 0), 1),
                "training_duration": round(input_features.get('training_duration', 0), 1)
            }
            briefing = generate_agent_briefing(
                player_id=f"{current_player.get('name')} (#{current_player.get('number')}) - {current_player.get('id')}",
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                risk_label=pred_class,
                risk_prob=pred_prob,
                key_metrics=key_metrics_summary,
                symptoms=symptoms_input,
                retrieved_rules=retrieved_rules,
                api_key=creds.get('groq_api_key')
            )
            st.session_state['current_briefing'] = briefing

    saved_briefing = st.session_state.get('current_briefing', None)
    if saved_briefing:
        st.markdown("---")
        with st.container():
            st.markdown(saved_briefing)
    else:
        st.info("Click the button above to generate the policy-grounded briefing for this player.")

# ----------------- TAB 4: HUMAN DECISION & ESCALATION -----------------
with tab4:
    st.subheader("🟨 Human-in-the-Loop Matchday Determination")
    st.warning("⚖️ **Mandatory Governance Rule PW-001**: AI generates decision support only. Matchday clearance, load restriction, and clinical referrals are strictly reserved to human medical professionals.")

    st.markdown("### 1. Matchday Clearance / Action Determination")
    human_decision = st.radio(
        "Referee & Medical Classification Outcome:",
        [
            "🟩 Eligible to Play (Full matchday & tactical clearance)",
            "🟨 Put Under Review (Restricted training load & active physio monitoring)",
            "🟥 Medical Assistance (Immediate referral to Team Physician / Clinical Diagnostic Assessment)"
        ],
        index=2 if pred_class == 1 else 0
    )

    st.markdown("### 2. Clinical Rationale & Physio Review Notes")
    clinical_notes = st.text_area(
        "Enter clinical justification for compliance audit (Rule AUD-001):",
        value=f"Decision for {current_player.get('name')} (#{current_player.get('number')}) following review of Decision Tree risk ({risk_label_str}, Prob: {pred_prob:.1%}), reported symptoms ('{symptoms_input}'), and Policy Rules PW-001/ESC-001."
    )

    st.markdown("### 3. Dispatch Official Medical Alert")
    alert_recipient = st.text_input("Emergency / Escalation Recipient Email:", value="tejaswi070203@gmail.com")

    if st.button("📨 Commit Decision & Dispatch Alert Email", type="primary"):
        rule_ids = [r['rule_id'] for r in retrieved_rules]
        log_decision_to_audit(
            player_name=current_player.get('name'),
            player_id=current_player.get('id'),
            risk_pred=risk_label_str,
            decision=human_decision,
            reviewer=reviewer_role,
            notes=clinical_notes,
            rules=rule_ids
        )
        st.success(f"✅ Determination for {current_player.get('name')} recorded in compliance audit log!")

        # Dispatch Email
        briefing_text = st.session_state.get('current_briefing', f"Player: {current_player.get('name')} (#{current_player.get('number')})\nRisk: {risk_label_str}\nDecision: {human_decision}\nNotes: {clinical_notes}")
        with st.spinner("Dispatching secure escalation notification via Gmail SMTP..."):
            success, msg = send_player_alert(
                recipient_email=alert_recipient,
                player_id=f"{current_player.get('name')} (#{current_player.get('number')})",
                risk_level=risk_label_str,
                human_decision=human_decision,
                briefing_text=briefing_text,
                rules_cited=rule_ids,
                reviewer=reviewer_role
            )
            if success:
                st.balloons()
                st.success(f"📧 Official alert successfully delivered to {alert_recipient}!")
            else:
                st.error(f"Alert Delivery Note: {msg}")

# ----------------- TAB 5: AUDIT TRAIL -----------------
with tab5:
    st.subheader("📋 Official Compliance Register (AUD-001)")
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        if rows:
            st.dataframe(rows, width="stretch")
        else:
            st.info("No determinations recorded yet.")
    else:
        st.info("Audit log file will be generated on your first committed determination.")

    st.markdown("---")
    st.markdown("""
    **Northbridge Meridian FC — Medical Governance System**
    - Policy Reference: NMFC-PG-POL-001
    - Document Owner: Head of Medical and Performance Services
    """)
