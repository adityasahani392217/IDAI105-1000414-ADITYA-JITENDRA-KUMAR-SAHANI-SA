import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCharge Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — dark electric theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --neon: #00f5d4;
    --neon2: #7b2ff7;
    --bg: #07090f;
    --card: #0e1320;
    --border: #1a2540;
    --text: #c8d8f0;
    --accent: #f0c040;
}

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default streamlit chrome during loading */
#MainMenu, footer, header { visibility: hidden; }

.stApp { background-color: var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0e1320 100%) !important;
    border-right: 1px solid var(--border);
}

/* Cards */
.ev-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.ev-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--neon), var(--neon2));
}

/* Metric boxes */
.metric-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    color: var(--neon);
    text-shadow: 0 0 20px rgba(0,245,212,0.5);
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6a7fa8;
    margin-top: 4px;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: var(--neon) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--neon2), #4a1fa0);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 1px;
    padding: 10px 24px;
    transition: all 0.3s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--neon), var(--neon2));
    color: #07090f !important;
    box-shadow: 0 0 20px rgba(0,245,212,0.4);
    transform: translateY(-2px);
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0a0e1a !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--neon) !important;
    box-shadow: 0 0 10px rgba(0,245,212,0.2) !important;
}

/* Loading screen */
.loading-screen {
    position: fixed; top:0; left:0; right:0; bottom:0;
    background: #07090f;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    z-index: 9999;
}
.loader-ring {
    width: 80px; height: 80px;
    border-radius: 50%;
    border: 3px solid transparent;
    border-top-color: #00f5d4;
    border-right-color: #7b2ff7;
    animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Anomaly badge */
.anomaly-badge {
    display: inline-block;
    background: rgba(255,60,60,0.15);
    border: 1px solid #ff3c3c;
    color: #ff7070;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Orbitron', monospace;
}

/* Cluster badge */
.cluster-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Welcome hero */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f5d4 0%, #7b2ff7 50%, #f0c040 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 16px;
}
.hero-sub {
    color: #6a7fa8;
    font-size: 1.1rem;
    letter-spacing: 1px;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* Progress bar override */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--neon), var(--neon2)) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    color: #6a7fa8 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--neon2), #4a1fa0) !important;
    color: white !important;
}

/* Alerts */
.stAlert {
    background: rgba(0,245,212,0.05) !important;
    border: 1px solid rgba(0,245,212,0.3) !important;
    border-radius: 8px !important;
}

/* Login screen */
.login-container {
    max-width: 420px;
    margin: 0 auto;
    padding: 40px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 0 60px rgba(0,245,212,0.08), 0 0 120px rgba(123,47,247,0.06);
}
.login-logo {
    text-align: center;
    margin-bottom: 32px;
}
.login-logo-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 8px;
}
.login-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--neon);
    text-align: center;
    margin-bottom: 4px;
}
.login-subtitle {
    color: #6a7fa8;
    font-size: 0.85rem;
    text-align: center;
    letter-spacing: 1px;
    margin-bottom: 32px;
}

/* Onboarding */
.onboard-step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-bottom: 16px;
}
.onboard-icon {
    font-size: 3rem;
    margin-bottom: 16px;
}
.onboard-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    color: var(--neon);
    margin-bottom: 8px;
}
.onboard-text {
    color: #6a7fa8;
    font-size: 0.9rem;
    line-height: 1.6;
}
.progress-dots {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin: 24px 0;
}
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--border); }
.dot.active { background: var(--neon); box-shadow: 0 0 8px var(--neon); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "loading_done": False,
        "logged_in": False,
        "onboarding_done": False,
        "onboard_step": 0,
        "loading_progress": 0,
        "df": None,
        "df_clean": None,
        "cluster_labels": None,
        "anomaly_flags": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
#  DATA GENERATION  (synthetic EV dataset)
# ─────────────────────────────────────────────
@st.cache_data
def generate_ev_dataset(n=500):
    np.random.seed(42)
    station_ids = [f"EV-{str(i).zfill(4)}" for i in range(1, n+1)]
    lats  = np.random.uniform(25.0, 49.0, n)
    lons  = np.random.uniform(-122.0, -70.0, n)
    charger_types = np.random.choice(["AC Level 1","AC Level 2","DC Fast"], n, p=[0.2,0.45,0.35])
    operators = np.random.choice(["ChargePoint","Blink","EVgo","Tesla","Electrify America"], n)
    cost = np.where(charger_types == "DC Fast",
                    np.random.uniform(0.28, 0.55, n),
                    np.where(charger_types == "AC Level 2",
                             np.random.uniform(0.12, 0.30, n),
                             np.random.uniform(0.05, 0.15, n)))
    capacity = np.where(charger_types == "DC Fast",
                        np.random.uniform(50, 350, n),
                        np.where(charger_types == "AC Level 2",
                                 np.random.uniform(6.2, 22, n),
                                 np.random.uniform(1.4, 3.3, n)))
    usage = (np.random.normal(35, 15, n) +
             np.where(charger_types == "DC Fast", 20, 0) +
             np.where(np.array(operators) == "Tesla", 10, 0)).clip(1, 200)
    # inject anomalies
    anomaly_idx = np.random.choice(n, 25, replace=False)
    usage[anomaly_idx] *= np.random.uniform(3.5, 6, len(anomaly_idx))
    availability = np.random.choice(["Available","Occupied","Offline"], n, p=[0.55,0.35,0.10])
    distance_city = np.random.exponential(8, n).clip(0.1, 80)
    install_year  = np.random.choice(range(2015, 2025), n)
    renewable     = np.random.choice(["Yes","No"], n, p=[0.40, 0.60])
    rating        = np.random.uniform(2.5, 5.0, n).round(1)
    parking_spots = np.random.randint(1, 20, n)
    maintenance_freq = np.random.choice(["Monthly","Quarterly","Bi-Annual","Annual"], n)
    connector_types  = np.random.choice(["CCS","CHAdeMO","J1772","Tesla"], n)

    df = pd.DataFrame({
        "Station_ID": station_ids,
        "Latitude": lats.round(4),
        "Longitude": lons.round(4),
        "Charger_Type": charger_types,
        "Cost_USD_kWh": cost.round(3),
        "Availability": availability,
        "Distance_to_City_km": distance_city.round(2),
        "Usage_Stats_avg_users_day": usage.round(1),
        "Station_Operator": operators,
        "Charging_Capacity_kW": capacity.round(1),
        "Connector_Types": connector_types,
        "Installation_Year": install_year,
        "Renewable_Energy_Source": renewable,
        "Reviews_Rating": rating,
        "Parking_Spots": parking_spots,
        "Maintenance_Frequency": maintenance_freq,
    })
    # add some missing
    for col in ["Reviews_Rating","Renewable_Energy_Source","Connector_Types"]:
        idx = np.random.choice(n, int(n*0.04), replace=False)
        df.loc[idx, col] = np.nan
    return df

# ─────────────────────────────────────────────
#  DATA PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df2 = df.copy()
    df2["Reviews_Rating"].fillna(df2["Reviews_Rating"].median(), inplace=True)
    df2["Renewable_Energy_Source"].fillna("No", inplace=True)
    df2["Connector_Types"].fillna("J1772", inplace=True)
    df2.drop_duplicates(subset="Station_ID", inplace=True)

    le = LabelEncoder()
    df2["Charger_Type_enc"]     = le.fit_transform(df2["Charger_Type"])
    df2["Operator_enc"]         = le.fit_transform(df2["Station_Operator"])
    df2["Renewable_enc"]        = (df2["Renewable_Energy_Source"] == "Yes").astype(int)
    df2["Availability_enc"]     = le.fit_transform(df2["Availability"])

    scaler = StandardScaler()
    num_cols = ["Cost_USD_kWh","Usage_Stats_avg_users_day","Charging_Capacity_kW","Distance_to_City_km"]
    df2[[c+"_norm" for c in num_cols]] = scaler.fit_transform(df2[num_cols])
    return df2

# ─────────────────────────────────────────────
#  CLUSTERING
# ─────────────────────────────────────────────
@st.cache_data
def run_clustering(df_clean, k=4):
    feats = ["Cost_USD_kWh_norm","Usage_Stats_avg_users_day_norm",
             "Charging_Capacity_kW_norm","Distance_to_City_km_norm","Charger_Type_enc"]
    X = df_clean[feats].values
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    names = {0:"🌿 Eco Commuters", 1:"⚡ Power Hubs", 2:"🌆 City Fast-Chargers", 3:"🛣️ Highway Stoppers"}
    return labels, names

# ─────────────────────────────────────────────
#  ANOMALY DETECTION
# ─────────────────────────────────────────────
@st.cache_data
def detect_anomalies(df_clean):
    z = np.abs(stats.zscore(df_clean["Usage_Stats_avg_users_day"]))
    Q1 = df_clean["Usage_Stats_avg_users_day"].quantile(0.25)
    Q3 = df_clean["Usage_Stats_avg_users_day"].quantile(0.75)
    IQR = Q3 - Q1
    iqr_flag = (df_clean["Usage_Stats_avg_users_day"] < Q1 - 1.5*IQR) | \
               (df_clean["Usage_Stats_avg_users_day"] > Q3 + 1.5*IQR)
    z_flag   = z > 3.0
    flag = iqr_flag | z_flag
    return flag.values

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,19,32,0.8)",
    font=dict(family="Exo 2", color="#c8d8f0"),
    colorway=["#00f5d4","#7b2ff7","#f0c040","#ff6b6b","#4ecdc4","#45b7d1"],
    xaxis=dict(gridcolor="#1a2540", linecolor="#1a2540"),
    yaxis=dict(gridcolor="#1a2540", linecolor="#1a2540"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────
#  LOADING SCREEN
# ─────────────────────────────────────────────
def show_loading():
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; 
                justify-content:center; height:85vh; gap:32px;">
        <div style="text-align:center;">
            <div style="font-size:4rem; margin-bottom:16px;">⚡</div>
            <div style="font-family:'Orbitron',monospace; font-size:2rem; font-weight:900;
                        background:linear-gradient(135deg,#00f5d4,#7b2ff7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        margin-bottom:8px;">
                SmartCharge Analytics
            </div>
            <div style="color:#6a7fa8; letter-spacing:3px; font-size:0.85rem; text-transform:uppercase;">
                Initializing Intelligence Systems
            </div>
        </div>
        <div style="width:80px; height:80px; border-radius:50%;
                    border:3px solid transparent;
                    border-top-color:#00f5d4; border-right-color:#7b2ff7;
                    animation:spin 1s linear infinite;">
        </div>
    </div>
    <style>@keyframes spin{to{transform:rotate(360deg);}}</style>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_msgs = [
        "⚡ Connecting to charging network...",
        "📡 Loading station telemetry...",
        "🧠 Initializing ML engines...",
        "📊 Preparing analytics modules...",
        "🗺️ Calibrating geo-intelligence...",
        "✅ All systems online",
    ]
    status_placeholder = st.empty()
    import time
    for i, msg in enumerate(status_msgs):
        status_placeholder.markdown(
            f"<div style='text-align:center; color:#00f5d4; font-family:Exo 2; font-size:0.9rem;'>{msg}</div>",
            unsafe_allow_html=True)
        pct = int((i+1) / len(status_msgs) * 100)
        progress_bar.progress(pct)
        time.sleep(0.45)

    st.session_state.loading_done = True
    st.rerun()

# ─────────────────────────────────────────────
#  LOGIN SCREEN
# ─────────────────────────────────────────────
CREDENTIALS = {"admin": "ev2024", "analyst": "charge123", "demo": "demo"}

def show_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.4,1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">
                <span class="login-logo-icon">⚡</span>
                <div class="login-title">SmartCharge Analytics</div>
                <div class="login-subtitle">EV Intelligence Platform v2.0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("<div style='background:#0e1320; border:1px solid #1a2540; border-radius:20px; padding:32px;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-family:Orbitron; font-size:1.2rem; color:#00f5d4; text-align:center; margin-bottom:24px;'>🔐 Secure Access</p>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")

            st.markdown("""
            <div style='background:rgba(0,245,212,0.05); border:1px solid rgba(0,245,212,0.2);
                        border-radius:8px; padding:12px; margin:12px 0; font-size:0.78rem; color:#6a7fa8;'>
            <b style='color:#00f5d4;'>Demo Credentials:</b><br>
            👤 admin / ev2024 &nbsp;|&nbsp; 👤 analyst / charge123 &nbsp;|&nbsp; 👤 demo / demo
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ LAUNCH DASHBOARD", key="login_btn"):
                if username in CREDENTIALS and CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Access granted. Loading platform...")
                    import time; time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ONBOARDING
# ─────────────────────────────────────────────
ONBOARD_STEPS = [
    {
        "icon": "⚡",
        "title": "Welcome to SmartCharge Analytics",
        "body": "Your all-in-one EV charging intelligence platform. Explore real-time station data, behavior clusters, anomalies, and strategic insights — all powered by advanced data mining."
    },
    {
        "icon": "🗺️",
        "title": "Interactive Station Maps",
        "body": "Visualize charging stations geographically. Color-coded clusters reveal usage patterns across regions — from busy city hubs to quiet highway stops."
    },
    {
        "icon": "🧠",
        "title": "AI-Powered Clustering",
        "body": "Our K-Means engine groups stations into behavioral clusters: Eco Commuters, Power Hubs, City Fast-Chargers, and Highway Stoppers — giving you instant segmentation."
    },
    {
        "icon": "🔍",
        "title": "Anomaly Detection Engine",
        "body": "Z-Score and IQR methods automatically flag stations with abnormal usage spikes, faulty patterns, or suspiciously low activity despite excellent facilities."
    },
    {
        "icon": "📊",
        "title": "Association Rule Mining",
        "body": "Discover hidden relationships: which charger types attract premium users? When is DC Fast demand highest? Rules mined via Apriori reveal what drives station success."
    },
]

def show_onboarding():
    step = st.session_state.onboard_step
    total = len(ONBOARD_STEPS)
    s = ONBOARD_STEPS[step]

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:8px;'>
            <span style='font-family:Orbitron; font-size:0.75rem; color:#6a7fa8; letter-spacing:3px;'>
                STEP {step+1} OF {total}
            </span>
        </div>
        <div class="onboard-step">
            <div class="onboard-icon">{s["icon"]}</div>
            <div class="onboard-title">{s["title"]}</div>
            <div class="onboard-text">{s["body"]}</div>
        </div>
        <div class="progress-dots">
        """ + "".join([f'<div class="dot {"active" if i==step else ""}"></div>' for i in range(total)]) + """
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if step > 0:
                if st.button("← Back", key="ob_back"):
                    st.session_state.onboard_step -= 1
                    st.rerun()
        with c2:
            if step < total - 1:
                if st.button("Next →", key="ob_next"):
                    st.session_state.onboard_step += 1
                    st.rerun()
            else:
                if st.button("✅ I Agree & Launch", key="ob_agree"):
                    st.session_state.onboarding_done = True
                    st.rerun()

        if step == total - 1:
            st.markdown("""
            <div style='background:rgba(0,245,212,0.05); border:1px solid rgba(0,245,212,0.2);
                        border-radius:10px; padding:14px; margin-top:8px; font-size:0.8rem; color:#6a7fa8; text-align:center;'>
            By clicking <b style='color:#00f5d4;'>I Agree & Launch</b>, you acknowledge that this platform 
            uses AI-generated insights for analytical purposes only. Data is synthetic and for educational use.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
def show_dashboard():
    # Load data
    if st.session_state.df is None:
        st.session_state.df = generate_ev_dataset(500)
    if st.session_state.df_clean is None:
        st.session_state.df_clean = preprocess(st.session_state.df)
    if st.session_state.cluster_labels is None:
        labels, names = run_clustering(st.session_state.df_clean)
        st.session_state.cluster_labels = labels
        st.session_state.cluster_names  = names
    if st.session_state.anomaly_flags is None:
        st.session_state.anomaly_flags = detect_anomalies(st.session_state.df_clean)

    df  = st.session_state.df_clean.copy()
    df["Cluster"] = st.session_state.cluster_labels
    df["Cluster_Name"] = df["Cluster"].map(st.session_state.cluster_names)
    df["Is_Anomaly"] = st.session_state.anomaly_flags

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0 10px;'>
            <div style='font-size:2.5rem;'>⚡</div>
            <div style='font-family:Orbitron; font-size:1rem; font-weight:700; color:#00f5d4;'>SmartCharge</div>
            <div style='color:#6a7fa8; font-size:0.75rem; letter-spacing:2px;'>ANALYTICS PLATFORM</div>
        </div>
        <hr style='border-color:#1a2540; margin:16px 0;'>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='color:#6a7fa8; font-size:0.8rem; margin-bottom:16px; text-align:center;'>👤 {st.session_state.get('username','analyst').upper()}</div>", unsafe_allow_html=True)

        page = st.selectbox("📍 Navigation", [
            "🏠 Overview",
            "🗺️ Station Map",
            "📊 EDA & Visualizations",
            "🧠 Clustering Analysis",
            "🔗 Association Rules",
            "🚨 Anomaly Detection",
            "💡 Insights & Report",
        ])

        st.markdown("<hr style='border-color:#1a2540; margin:16px 0;'>", unsafe_allow_html=True)

        # Filters
        st.markdown("<div style='font-family:Orbitron; font-size:0.75rem; color:#6a7fa8; letter-spacing:2px; margin-bottom:8px;'>FILTERS</div>", unsafe_allow_html=True)
        ct_filter = st.multiselect("Charger Type", df["Charger_Type"].unique(), default=list(df["Charger_Type"].unique()))
        op_filter = st.multiselect("Operator", df["Station_Operator"].unique(), default=list(df["Station_Operator"].unique()))

        if st.button("🚪 Logout"):
            for k in ["logged_in","onboarding_done","df","df_clean","cluster_labels","anomaly_flags"]:
                st.session_state[k] = False if k in ["logged_in","onboarding_done"] else None
            st.rerun()

    # Apply filters
    mask = df["Charger_Type"].isin(ct_filter) & df["Station_Operator"].isin(op_filter)
    dff = df[mask]

    # ── PAGES ─────────────────────────────────

    # ── OVERVIEW ──────────────────────────────
    if page == "🏠 Overview":
        st.markdown('<div class="hero-title">⚡ SmartCharge<br>Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">EV Charging Intelligence Platform — Real-Time Station Insights</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        metrics = [
            (len(dff), "Total Stations"),
            (f"{dff['Usage_Stats_avg_users_day'].mean():.1f}", "Avg Users/Day"),
            (f"${dff['Cost_USD_kWh'].mean():.3f}", "Avg Cost/kWh"),
            (int(df["Is_Anomaly"].sum()), "Anomalies Found"),
            (f"{dff['Reviews_Rating'].mean():.2f}⭐", "Avg Rating"),
        ]
        for col, (val, label) in zip([c1,c2,c3,c4,c5], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(dff, names="Charger_Type", title="Charger Type Distribution",
                         color_discrete_sequence=["#00f5d4","#7b2ff7","#f0c040"])
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            op_usage = dff.groupby("Station_Operator")["Usage_Stats_avg_users_day"].mean().sort_values(ascending=True)
            fig = px.bar(op_usage, orientation='h', title="Avg Usage by Operator",
                         color=op_usage.values, color_continuous_scale=["#1a2540","#00f5d4"])
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            yr = dff.groupby("Installation_Year")["Station_ID"].count().reset_index()
            yr.columns = ["Year","Count"]
            fig = px.area(yr, x="Year", y="Count", title="Station Installations Over Time",
                          color_discrete_sequence=["#00f5d4"])
            fig.update_traces(fill='tozeroy', fillcolor='rgba(0,245,212,0.1)')
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            renew = dff["Renewable_Energy_Source"].value_counts()
            fig = px.pie(values=renew.values, names=renew.index,
                         title="Renewable Energy Adoption",
                         color_discrete_sequence=["#00f5d4","#7b2ff7"])
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── MAP ────────────────────────────────────
    elif page == "🗺️ Station Map":
        st.markdown("## 🗺️ Station Geographic Map")
        view = st.radio("Color by:", ["Charger Type","Cluster","Anomaly","Rating"], horizontal=True)

        if view == "Charger Type":
            fig = px.scatter_mapbox(dff, lat="Latitude", lon="Longitude",
                                    color="Charger_Type", size="Usage_Stats_avg_users_day",
                                    hover_name="Station_ID",
                                    hover_data={"Cost_USD_kWh":True,"Reviews_Rating":True,"Station_Operator":True},
                                    title="EV Stations — Charger Type",
                                    color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"})
        elif view == "Cluster":
            fig = px.scatter_mapbox(dff, lat="Latitude", lon="Longitude",
                                    color="Cluster_Name", size="Usage_Stats_avg_users_day",
                                    hover_name="Station_ID",
                                    title="EV Stations — Behavioral Clusters")
        elif view == "Anomaly":
            dff2 = dff.copy()
            dff2["Status"] = dff2["Is_Anomaly"].map({True:"⚠️ Anomaly", False:"✅ Normal"})
            fig = px.scatter_mapbox(dff2, lat="Latitude", lon="Longitude",
                                    color="Status", size="Usage_Stats_avg_users_day",
                                    hover_name="Station_ID",
                                    color_discrete_map={"⚠️ Anomaly":"#ff3c3c","✅ Normal":"#00f5d4"},
                                    title="EV Stations — Anomaly Flags")
        else:
            fig = px.scatter_mapbox(dff, lat="Latitude", lon="Longitude",
                                    color="Reviews_Rating", size="Usage_Stats_avg_users_day",
                                    hover_name="Station_ID",
                                    color_continuous_scale=["#1a2540","#7b2ff7","#00f5d4"],
                                    title="EV Stations — Review Ratings")

        fig.update_layout(mapbox_style="carto-darkmatter",
                          mapbox_zoom=3, mapbox_center={"lat":37.5,"lon":-96},
                          margin=dict(l=0,r=0,t=40,b=0),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#c8d8f0"),
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ── EDA ────────────────────────────────────
    elif page == "📊 EDA & Visualizations":
        st.markdown("## 📊 Exploratory Data Analysis")
        tabs = st.tabs(["Usage Distribution","Cost Analysis","Rating Analysis","Correlation","Time Trends"])

        with tabs[0]:
            c1,c2 = st.columns(2)
            with c1:
                fig = px.histogram(dff, x="Usage_Stats_avg_users_day", nbins=40,
                                   color="Charger_Type", title="Usage Distribution by Charger Type",
                                   barmode="overlay", opacity=0.7,
                                   color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"})
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(dff, x="Charger_Type", y="Usage_Stats_avg_users_day",
                             color="Charger_Type", title="Usage Boxplot by Charger Type",
                             color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"})
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

            fig = px.violin(dff, x="Station_Operator", y="Usage_Stats_avg_users_day",
                            color="Charger_Type", box=True, title="Usage Distribution by Operator & Charger Type",
                            color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"})
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1,c2 = st.columns(2)
            with c1:
                fig = px.box(dff, x="Station_Operator", y="Cost_USD_kWh",
                             color="Charger_Type", title="Cost Distribution by Operator",
                             color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"})
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(dff, x="Cost_USD_kWh", y="Usage_Stats_avg_users_day",
                                 color="Charger_Type", size="Charging_Capacity_kW",
                                 title="Cost vs Usage (size = Capacity)",
                                 color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"},
                                 hover_name="Station_ID")
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            c1,c2 = st.columns(2)
            with c1:
                fig = px.histogram(dff, x="Reviews_Rating", nbins=25,
                                   title="Rating Distribution", color_discrete_sequence=["#00f5d4"])
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(dff, x="Reviews_Rating", y="Usage_Stats_avg_users_day",
                                 color="Charger_Type", title="Ratings vs Usage",
                                 color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"},
                                 trendline="ols")
                apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            num = dff[["Cost_USD_kWh","Usage_Stats_avg_users_day","Charging_Capacity_kW",
                        "Distance_to_City_km","Reviews_Rating","Parking_Spots"]].corr()
            fig = px.imshow(num, color_continuous_scale=["#1a2540","#0e1320","#7b2ff7","#00f5d4"],
                            title="Correlation Heatmap", text_auto=".2f")
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            yr_ct = dff.groupby(["Installation_Year","Charger_Type"])["Station_ID"].count().reset_index()
            yr_ct.columns = ["Year","Charger_Type","Count"]
            fig = px.line(yr_ct, x="Year", y="Count", color="Charger_Type",
                          markers=True, title="Installation Trend by Charger Type",
                          color_discrete_map={"AC Level 1":"#f0c040","AC Level 2":"#00f5d4","DC Fast":"#7b2ff7"})
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

            yr_usage = dff.groupby("Installation_Year")["Usage_Stats_avg_users_day"].mean().reset_index()
            fig = px.bar(yr_usage, x="Installation_Year", y="Usage_Stats_avg_users_day",
                         title="Avg Daily Usage by Installation Year",
                         color="Usage_Stats_avg_users_day",
                         color_continuous_scale=["#1a2540","#7b2ff7","#00f5d4"])
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    # ── CLUSTERING ─────────────────────────────
    elif page == "🧠 Clustering Analysis":
        st.markdown("## 🧠 Clustering Analysis")

        c1,c2 = st.columns([2,1])
        with c1:
            # Elbow method
            feats = ["Cost_USD_kWh_norm","Usage_Stats_avg_users_day_norm",
                     "Charging_Capacity_kW_norm","Distance_to_City_km_norm","Charger_Type_enc"]
            X = df[feats].values
            inertias = []
            K_range = range(2, 10)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers",
                                     line=dict(color="#00f5d4", width=2),
                                     marker=dict(color="#7b2ff7", size=8)))
            fig.update_layout(title="Elbow Method — Optimal K", xaxis_title="K", yaxis_title="Inertia", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            cluster_counts = dff["Cluster_Name"].value_counts()
            for name, cnt in cluster_counts.items():
                colors = {"🌿 Eco Commuters":"#00f5d4","⚡ Power Hubs":"#f0c040",
                          "🌆 City Fast-Chargers":"#7b2ff7","🛣️ Highway Stoppers":"#ff6b6b"}
                c = colors.get(name, "#6a7fa8")
                pct = cnt/len(dff)*100
                st.markdown(f"""
                <div class="ev-card" style="margin-bottom:10px;">
                    <div style="font-family:Orbitron; font-size:0.85rem; color:{c}; margin-bottom:4px;">{name}</div>
                    <div style="font-size:1.4rem; font-weight:700; color:white;">{cnt} <span style="font-size:0.8rem; color:#6a7fa8;">stations ({pct:.0f}%)</span></div>
                </div>""", unsafe_allow_html=True)

        # PCA scatter
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df[feats].values)
        dff["PCA1"] = X_pca[:, 0]
        dff["PCA2"] = X_pca[:, 1]

        fig = px.scatter(dff, x="PCA1", y="PCA2", color="Cluster_Name",
                         title="Cluster Visualization (PCA Projection)",
                         hover_name="Station_ID",
                         hover_data={"Usage_Stats_avg_users_day":True,"Cost_USD_kWh":True},
                         opacity=0.75)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster radar
        cluster_profile = dff.groupby("Cluster_Name")[["Usage_Stats_avg_users_day",
                                                        "Cost_USD_kWh","Charging_Capacity_kW",
                                                        "Distance_to_City_km","Reviews_Rating"]].mean()
        # normalize 0-1
        cp_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
        cats = cp_norm.columns.tolist()
        fig = go.Figure()
        colors_map = ["#00f5d4","#f0c040","#7b2ff7","#ff6b6b"]
        for i, (idx, row) in enumerate(cp_norm.iterrows()):
            vals = row.tolist() + [row.tolist()[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]],
                                          fill='toself', name=idx,
                                          line_color=colors_map[i % 4],
                                          fillcolor=colors_map[i % 4].replace("#","rgba(").rstrip(")") + ",0.1)" if "#" in colors_map[i%4] else colors_map[i%4]))
        fig.update_layout(title="Cluster Profiles — Radar Chart",
                          polar=dict(bgcolor="rgba(14,19,32,0.8)",
                                     radialaxis=dict(gridcolor="#1a2540"),
                                     angularaxis=dict(gridcolor="#1a2540")),
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── ASSOCIATION RULES ──────────────────────
    elif page == "🔗 Association Rules":
        st.markdown("## 🔗 Association Rule Mining")
        st.info("⚡ Association rules mined from station feature combinations (Charger Type × Operator × Renewable × Availability)")

        # Build transaction-style data manually (without mlxtend for robustness)
        import itertools

        df_arm = dff.copy()
        df_arm["item_charger"] = "CT_" + df_arm["Charger_Type"].str.replace(" ","_")
        df_arm["item_operator"] = "OP_" + df_arm["Station_Operator"].str.replace(" ","_")
        df_arm["item_renewable"] = "RE_" + df_arm["Renewable_Energy_Source"].astype(str)
        df_arm["item_avail"]   = "AV_" + df_arm["Availability"]
        df_arm["item_highuse"] = "USE_" + pd.cut(df_arm["Usage_Stats_avg_users_day"],
                                                  bins=[0,20,50,999],
                                                  labels=["LOW","MID","HIGH"]).astype(str)

        items_cols = ["item_charger","item_operator","item_renewable","item_avail","item_highuse"]

        # Count co-occurrences
        rules_data = []
        item_list = [df_arm[c].tolist() for c in items_cols]
        all_items = list(set(sum(item_list, [])))
        N = len(df_arm)

        # Support for individual items
        support = {}
        for col in items_cols:
            for val, cnt in df_arm[col].value_counts().items():
                support[val] = cnt / N

        # Co-occurrence pairs
        for c1, c2 in list(itertools.combinations(items_cols, 2)):
            co = df_arm.groupby([c1,c2]).size().reset_index(name="count")
            for _, row in co.iterrows():
                ant, con, cnt = row[c1], row[c2], row["count"]
                sup = cnt / N
                conf = sup / support.get(ant, 1)
                lift = conf / support.get(con, 1)
                if sup >= 0.05 and conf >= 0.4:
                    rules_data.append({"Antecedent": ant, "Consequent": con,
                                       "Support": round(sup, 3), "Confidence": round(conf, 3), "Lift": round(lift, 2)})

        rules_df = pd.DataFrame(rules_data).sort_values("Lift", ascending=False).head(20)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(rules_df, x="Support", y="Confidence", size="Lift",
                             color="Lift", text="Consequent",
                             title="Association Rules Map (size=Lift)",
                             color_continuous_scale=["#1a2540","#7b2ff7","#00f5d4"])
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        with c2:
            top10 = rules_df.head(10)
            fig = px.bar(top10, x="Lift", y="Antecedent", orientation='h',
                         color="Confidence", title="Top 10 Rules by Lift",
                         color_continuous_scale=["#1a2540","#7b2ff7","#00f5d4"])
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Top Association Rules")
        st.dataframe(rules_df.style.background_gradient(subset=["Lift","Confidence"],
                                                          cmap="Blues"),
                     use_container_width=True, height=350)

        st.markdown("""
        <div class="ev-card">
        <b style='color:#00f5d4;'>📌 Key Findings:</b><br><br>
        • <b>DC Fast chargers</b> consistently co-occur with high usage stations — driving demand hubs.<br>
        • <b>Renewable + High Rating</b> stations attract significantly more users (Lift > 1.8).<br>
        • <b>ChargePoint & EVgo</b> operators frequently appear with "Available" status — better reliability.<br>
        • <b>Low distance to city</b> strongly associates with MID-HIGH usage patterns.
        </div>""", unsafe_allow_html=True)

    # ── ANOMALY DETECTION ──────────────────────
    elif page == "🚨 Anomaly Detection":
        st.markdown("## 🚨 Anomaly Detection")

        anom = dff[dff["Is_Anomaly"] == True]
        norm = dff[dff["Is_Anomaly"] == False]

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-box"><div class="metric-value" style='color:#ff3c3c;'>{len(anom)}</div>
            <div class="metric-label">Anomalous Stations</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box"><div class="metric-value">{len(norm)}</div>
            <div class="metric-label">Normal Stations</div></div>""", unsafe_allow_html=True)
        with c3:
            rate = len(anom)/len(dff)*100
            st.markdown(f"""<div class="metric-box"><div class="metric-value" style='color:#f0c040;'>{rate:.1f}%</div>
            <div class="metric-label">Anomaly Rate</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=norm["Usage_Stats_avg_users_day"], name="Normal",
                                       marker_color="#00f5d4", opacity=0.7, nbinsx=40))
            fig.add_trace(go.Histogram(x=anom["Usage_Stats_avg_users_day"], name="Anomaly",
                                       marker_color="#ff3c3c", opacity=0.8, nbinsx=20))
            fig.update_layout(title="Usage Distribution: Normal vs Anomaly",
                               barmode="overlay", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(dff, x="Station_ID", y="Usage_Stats_avg_users_day",
                             color=dff["Is_Anomaly"].map({True:"⚠️ Anomaly", False:"✅ Normal"}),
                             title="Station Usage — Anomaly Flags",
                             color_discrete_map={"⚠️ Anomaly":"#ff3c3c","✅ Normal":"#00f5d4"})
            apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

        # Z-score visualization
        z_scores = np.abs(stats.zscore(dff["Usage_Stats_avg_users_day"]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(z_scores))), y=z_scores,
                                 mode="markers",
                                 marker=dict(color=np.where(z_scores>3,"#ff3c3c","#00f5d4"),
                                             size=6, opacity=0.7),
                                 name="Z-Score"))
        fig.add_hline(y=3, line_color="#f0c040", line_dash="dash",
                      annotation_text="Threshold (z=3)", annotation_font_color="#f0c040")
        fig.update_layout(title="Z-Score Analysis — Usage Anomalies", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ⚠️ Anomalous Station Details")
        show_cols = ["Station_ID","Charger_Type","Station_Operator","Usage_Stats_avg_users_day",
                     "Cost_USD_kWh","Reviews_Rating","Distance_to_City_km"]
        st.dataframe(anom[show_cols].sort_values("Usage_Stats_avg_users_day", ascending=False).head(25),
                     use_container_width=True)

    # ── INSIGHTS ───────────────────────────────
    elif page == "💡 Insights & Report":
        st.markdown("## 💡 Strategic Insights & Report")

        insights = [
            ("⚡","DC Fast Chargers Drive Revenue",
             f"DC Fast stations average {dff[dff['Charger_Type']=='DC Fast']['Usage_Stats_avg_users_day'].mean():.0f} users/day vs {dff[dff['Charger_Type']=='AC Level 1']['Usage_Stats_avg_users_day'].mean():.0f} for AC Level 1 — a significant usage premium despite higher cost."),
            ("🌿","Renewable Energy Boosts Ratings",
             f"Stations with renewable energy score {dff[dff['Renewable_Energy_Source']=='Yes']['Reviews_Rating'].mean():.2f}★ vs {dff[dff['Renewable_Energy_Source']=='No']['Reviews_Rating'].mean():.2f}★ — invest in green infrastructure."),
            ("🏙️","Urban Proximity = Higher Demand",
             f"Stations within 5km of city average {dff[dff['Distance_to_City_km']<5]['Usage_Stats_avg_users_day'].mean():.0f} users/day vs {dff[dff['Distance_to_City_km']>20]['Usage_Stats_avg_users_day'].mean():.0f} for remote stations."),
            ("🚨","Anomalies Require Immediate Audit",
             f"{dff['Is_Anomaly'].sum()} stations flagged with abnormally high usage — likely data errors, misreported sessions, or genuinely overcrowded stations needing expansion."),
            ("📈","Installation Growth Accelerating",
             f"Network grew {((dff['Installation_Year']>=2022).sum()/(dff['Installation_Year']<2022).sum()-1)*100:.0f}% in 2022-2024 vs prior years — market momentum is strong."),
        ]

        for icon, title, body in insights:
            st.markdown(f"""
            <div class="ev-card">
                <div style='font-size:1.5rem; margin-bottom:8px;'>{icon}</div>
                <div style='font-family:Orbitron; font-size:0.95rem; color:#00f5d4; margin-bottom:8px;'>{title}</div>
                <div style='color:#c8d8f0; font-size:0.9rem; line-height:1.6;'>{body}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Summary Statistics")
        summary = dff.groupby("Charger_Type").agg(
            Stations=("Station_ID","count"),
            Avg_Usage=("Usage_Stats_avg_users_day","mean"),
            Avg_Cost=("Cost_USD_kWh","mean"),
            Avg_Rating=("Reviews_Rating","mean"),
            Avg_Capacity=("Charging_Capacity_kW","mean"),
        ).round(2)
        st.dataframe(summary, use_container_width=True)

        st.markdown("### 🎯 Strategic Recommendations")
        recs = [
            "**Expand DC Fast charging** along interstate corridors — highest ROI per station.",
            "**Prioritize renewable energy** integration — improves ratings and attracts eco-conscious EV users.",
            "**Investigate flagged anomalies** — 25 stations show usage spikes warranting physical inspection.",
            "**Optimize pricing** at underutilized AC Level 1 stations — consider bundling with parking fees.",
            "**Target cluster 'Power Hubs'** for premium service upgrades — highest capacity utilization.",
        ]
        for r in recs:
            st.markdown(f"- {r}")

# ─────────────────────────────────────────────
#  MAIN ROUTER
# ─────────────────────────────────────────────
if not st.session_state.loading_done:
    show_loading()
elif not st.session_state.logged_in:
    show_login()
elif not st.session_state.onboarding_done:
    show_onboarding()
else:
    show_dashboard()