"""
SmartCharging Analytics Dashboard
Scenario 2 — Data Mining Summative Assessment
Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1e88e5; text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING & CACHING ────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset; fall back to synthetic demo data if no file."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # --- SYNTHETIC DEMO DATA ---
        np.random.seed(42)
        n = 500
        charger_types = ["AC Level 1", "AC Level 2", "DC Fast"]
        operators = ["ChargePoint", "Tesla", "EVgo", "Blink", "Shell", "BP Pulse"]
        renewable = ["Yes", "No"]
        availability = ["Available", "In Use", "Offline"]
        maint = ["Weekly", "Monthly", "Quarterly", "Annually"]

        df = pd.DataFrame({
            "Station ID": [f"STA{i:04d}" for i in range(n)],
            "Latitude": np.random.uniform(25, 55, n),
            "Longitude": np.random.uniform(-120, 20, n),
            "Address": [f"Address {i}" for i in range(n)],
            "Charger Type": np.random.choice(charger_types, n,
                                              p=[0.2, 0.45, 0.35]),
            "Cost (USD/kWh)": np.round(np.random.uniform(0.10, 0.55, n), 3),
            "Availability": np.random.choice(availability, n,
                                              p=[0.55, 0.35, 0.10]),
            "Distance to City (km)": np.round(np.random.exponential(25, n), 1),
            "Usage Stats (avg users/day)": np.round(
                np.clip(np.random.normal(18, 8, n), 1, 60), 1),
            "Station Operator": np.random.choice(operators, n),
            "Charging Capacity (kW)": np.random.choice(
                [7.2, 11, 22, 50, 150, 350], n, p=[0.1,0.15,0.2,0.25,0.2,0.1]),
            "Connector Types": np.random.choice(
                ["CCS", "CHAdeMO", "Type 2", "Tesla"], n),
            "Installation Year": np.random.choice(range(2015, 2024), n),
            "Renewable Energy Source": np.random.choice(renewable, n,
                                                         p=[0.4, 0.6]),
            "Reviews (Rating)": np.round(
                np.clip(np.random.normal(3.8, 0.7, n), 1, 5), 1),
            "Parking Spots": np.random.randint(2, 20, n),
            "Maintenance Frequency": np.random.choice(maint, n),
        })
        # Inject realistic anomalies
        anomaly_idx = np.random.choice(n, 25, replace=False)
        df.loc[anomaly_idx, "Usage Stats (avg users/day)"] = (
            np.random.uniform(55, 80, 25))
        df.loc[anomaly_idx[:10], "Cost (USD/kWh)"] = (
            np.random.uniform(0.60, 0.90, 10))
        df.loc[anomaly_idx[:10], "Reviews (Rating)"] = (
            np.random.uniform(1, 1.8, 10))

    return df


@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(
        subset="Station ID" if "Station ID" in df.columns else df.columns[0])
    num_cols = ["Reviews (Rating)", "Charging Capacity (kW)",
                "Cost (USD/kWh)", "Usage Stats (avg users/day)",
                "Distance to City (km)", "Parking Spots"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    cat_cols = ["Renewable Energy Source", "Connector Types",
                "Charger Type", "Station Operator",
                "Availability", "Maintenance Frequency"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

    if "Renewable Energy Source" in df.columns:
        df["Renewable_Enc"] = (df["Renewable Energy Source"]
                               .str.lower().map({"yes": 1, "no": 0}).fillna(0))
    if "Availability" in df.columns:
        df["Avail_Enc"] = (df["Availability"].str.lower()
                           .map({"available": 1}).fillna(0))
    if "Charger Type" in df.columns:
        df["Charger_Enc"] = (df["Charger Type"]
                             .map({"AC Level 1": 1, "AC Level 2": 2, "DC Fast": 3})
                             .fillna(1))
    if "Maintenance Frequency" in df.columns:
        maint_map = {"Weekly":4,"Monthly":3,"Quarterly":2,"Annually":1,"Rarely":0}
        df["Maint_Enc"] = df["Maintenance Frequency"].map(maint_map).fillna(1)

    scaler = StandardScaler()
    scale_cols = ["Cost (USD/kWh)", "Usage Stats (avg users/day)",
                  "Charging Capacity (kW)", "Distance to City (km)",
                  "Reviews (Rating)"]
    existing = [c for c in scale_cols if c in df.columns]
    scaled = scaler.fit_transform(df[existing])
    for i, c in enumerate(existing):
        df[c + "_sc"] = scaled[:, i]
    return df


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/electric-car.png", width=80)
    st.title("⚡ SmartCharging")
    st.caption("EV Analytics Dashboard")

    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Dataset (CSV)",
                                 type=["csv"])
    st.markdown("---")
    st.markdown("**Filters**")

    df_raw = load_data(uploaded)
    df = preprocess(df_raw.copy())

    charger_filter = st.multiselect(
        "Charger Type",
        options=df["Charger Type"].unique().tolist() if "Charger Type" in df.columns else [],
        default=df["Charger Type"].unique().tolist() if "Charger Type" in df.columns else []
    )
    renewable_filter = st.multiselect(
        "Renewable Energy",
        options=df["Renewable Energy Source"].unique().tolist() if "Renewable Energy Source" in df.columns else [],
        default=df["Renewable Energy Source"].unique().tolist() if "Renewable Energy Source" in df.columns else []
    )
    n_clusters = st.slider("K-Means Clusters", 2, 8, 4)

    if charger_filter and "Charger Type" in df.columns:
        df = df[df["Charger Type"].isin(charger_filter)]
    if renewable_filter and "Renewable Energy Source" in df.columns:
        df = df[df["Renewable Energy Source"].isin(renewable_filter)]

    st.markdown("---")
    st.info(f"📊 **{len(df):,}** stations loaded")


# ── HEADER ────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ SmartCharging Analytics Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Uncovering EV Behaviour Patterns | '
            'SmartEnergy Data Lab</div>', unsafe_allow_html=True)

# ── KPI METRICS ───────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🔌 Total Stations", f"{len(df):,}")
col2.metric("📈 Avg Users/Day",
            f"{df['Usage Stats (avg users/day)'].mean():.1f}")
col3.metric("💰 Avg Cost/kWh",
            f"${df['Cost (USD/kWh)'].mean():.3f}")
col4.metric("⭐ Avg Rating",
            f"{df['Reviews (Rating)'].mean():.2f}/5")
if "Renewable Energy Source" in df.columns:
    pct = (df["Renewable Energy Source"].str.lower() == "yes").mean() * 100
    col5.metric("🌿 Renewable %", f"{pct:.1f}%")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────
tabs = st.tabs(["📊 EDA", "🗂️ Clustering", "🔗 Association Rules",
                "🚨 Anomaly Detection", "🗺️ Geo Map", "📋 Insights"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Usage Stats (avg users/day)",
                           nbins=30, color_discrete_sequence=["#1e88e5"],
                           title="Distribution: Avg Users/Day",
                           marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "Charger Type" in df.columns:
            avg_u = (df.groupby("Charger Type")
                       ["Usage Stats (avg users/day)"].mean()
                       .reset_index())
            fig = px.bar(avg_u, x="Charger Type",
                         y="Usage Stats (avg users/day)",
                         color="Charger Type",
                         title="Avg Daily Users by Charger Type",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "Installation Year" in df.columns:
            yearly = (df.groupby("Installation Year")
                        ["Usage Stats (avg users/day)"].mean().reset_index())
            fig = px.line(yearly, x="Installation Year",
                          y="Usage Stats (avg users/day)",
                          markers=True,
                          title="Avg Users/Day vs Installation Year",
                          color_discrete_sequence=["#43a047"])
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if "Station Operator" in df.columns:
            top_ops = df["Station Operator"].value_counts().head(8).index
            sub = df[df["Station Operator"].isin(top_ops)]
            fig = px.box(sub, x="Station Operator", y="Cost (USD/kWh)",
                         color="Station Operator",
                         title="Cost by Operator (Top 8)",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    num_f = ["Cost (USD/kWh)", "Usage Stats (avg users/day)",
             "Charging Capacity (kW)", "Distance to City (km)",
             "Reviews (Rating)", "Parking Spots",
             "Renewable_Enc", "Charger_Enc", "Avail_Enc"]
    existing = [c for c in num_f if c in df.columns]
    corr_matrix = df[existing].corr()
    fig = px.imshow(corr_matrix, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    # Reviews vs Usage
    st.subheader("Reviews vs. Usage (scatter)")
    fig = px.scatter(df, x="Reviews (Rating)",
                     y="Usage Stats (avg users/day)",
                     color="Charger Type" if "Charger Type" in df.columns else None,
                     hover_data=["Station ID"] if "Station ID" in df.columns else None,
                     opacity=0.7,
                     title="Reviews (Rating) vs Average Daily Users")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("K-Means Clustering Analysis")

    feat_cols = [c + "_sc" for c in
                 ["Usage Stats (avg users/day)", "Charging Capacity (kW)",
                  "Cost (USD/kWh)", "Distance to City (km)"]
                 if c + "_sc" in df.columns]

    if feat_cols and len(df) >= n_clusters:
        X = df[feat_cols].fillna(0)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Cluster"] = km.fit_predict(X).astype(str)

        # Elbow chart
        inertias = []
        for k in range(2, 10):
            inertias.append(KMeans(n_clusters=k, random_state=42,
                                   n_init=10).fit(X).inertia_)
        fig_elbow = px.line(x=range(2,10), y=inertias,
                            markers=True,
                            labels={"x":"K","y":"Inertia"},
                            title="Elbow Method — Optimal K",
                            color_discrete_sequence=["#e53935"])
        st.plotly_chart(fig_elbow, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df,
                             x="Usage Stats (avg users/day)",
                             y="Charging Capacity (kW)",
                             color="Cluster",
                             title="Clusters: Usage vs Capacity",
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df,
                             x="Distance to City (km)",
                             y="Cost (USD/kWh)",
                             color="Cluster",
                             title="Clusters: Distance vs Cost",
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # Cluster profile table
        st.subheader("Cluster Profiles")
        profile_cols = ["Usage Stats (avg users/day)", "Charging Capacity (kW)",
                        "Cost (USD/kWh)", "Distance to City (km)",
                        "Reviews (Rating)"]
        existing_p = [c for c in profile_cols if c in df.columns]
        profile = df.groupby("Cluster")[existing_p].mean().round(2)
        profile.columns = [c.replace(" (avg users/day)","")
                           .replace(" (USD/kWh)","")
                           .replace(" (kW)","")
                           .replace(" to City (km)","")
                           for c in profile.columns]
        st.dataframe(profile, use_container_width=True)
    else:
        st.warning("Not enough data for clustering with selected filters.")


# ════════════════════════════════════════════════════════════════
# TAB 3 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Association Rule Mining")
    min_sup = st.slider("Min Support", 0.05, 0.40, 0.15, 0.01)
    min_conf = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)

    df["Usage_Lev"] = pd.cut(df["Usage Stats (avg users/day)"],
                              bins=3,
                              labels=["Low_Usage","Med_Usage","High_Usage"])
    df["Cost_Lev"]  = pd.cut(df["Cost (USD/kWh)"],
                              bins=3,
                              labels=["Low_Cost","Med_Cost","High_Cost"])
    df["Cap_Lev"]   = pd.cut(df["Charging Capacity (kW)"],
                              bins=3,
                              labels=["Low_Cap","Med_Cap","High_Cap"])
    df["Dist_Lev"]  = pd.cut(df["Distance to City (km)"],
                              bins=3,
                              labels=["Near_City","Mid_Dist","Far_Rural"])

    basket_c = ["Charger Type","Usage_Lev","Cost_Lev","Cap_Lev",
                "Dist_Lev","Renewable Energy Source"]
    basket_c = [c for c in basket_c if c in df.columns]
    transactions = df[basket_c].astype(str).values.tolist()

    te  = TransactionEncoder()
    arr = te.fit_transform(transactions)
    bdf = pd.DataFrame(arr, columns=te.columns_)

    try:
        freq = apriori(bdf, min_support=min_sup,
                       use_colnames=True, verbose=0)
        rules = association_rules(freq, metric="confidence",
                                  min_threshold=min_conf)
        rules = rules.sort_values("lift", ascending=False)

        st.success(f"✅ Found **{len(rules)}** rules from "
                   f"**{len(freq)}** frequent itemsets")

        # Display rules table
        rules_disp = rules.copy()
        rules_disp["antecedents"] = rules_disp["antecedents"].apply(
            lambda x: ", ".join(list(x)))
        rules_disp["consequents"] = rules_disp["consequents"].apply(
            lambda x: ", ".join(list(x)))
        st.dataframe(
            rules_disp[["antecedents","consequents",
                         "support","confidence","lift"]]
            .round(3).head(20),
            use_container_width=True)

        # Scatter: support vs confidence
        fig = px.scatter(rules_disp.head(50),
                         x="support", y="confidence",
                         size="lift", color="lift",
                         hover_data=["antecedents","consequents"],
                         color_continuous_scale="Viridis",
                         title="Association Rules — Support vs Confidence (size=Lift)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"ARM failed: {e}. Try lowering min support.")


# ════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Anomaly Detection")
    contamination = st.slider("Isolation Forest Contamination", 0.01, 0.20,
                              0.05, 0.01)

    target = "Usage Stats (avg users/day)"
    df["Z_Score"] = np.abs(stats.zscore(df[target].fillna(0)))
    Q1, Q3 = df[target].quantile(0.25), df[target].quantile(0.75)
    IQR = Q3 - Q1
    df["Anomaly_IQR"] = (
        (df[target] < Q1 - 1.5*IQR) | (df[target] > Q3 + 1.5*IQR)
    ).astype(int)
    df["Anomaly_ZScore"] = (df["Z_Score"] > 3).astype(int)

    iso_f = [c for c in ["Usage Stats (avg users/day)", "Cost (USD/kWh)",
                          "Charging Capacity (kW)", "Reviews (Rating)",
                          "Maint_Enc"] if c in df.columns]
    X_iso = df[iso_f].fillna(df[iso_f].median())
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["Anomaly_IF"] = (iso.fit_predict(X_iso) == -1).astype(int)
    df["Anomaly_Score"] = (df["Anomaly_ZScore"] + df["Anomaly_IQR"]
                           + df["Anomaly_IF"])
    df["Anomaly"] = (df["Anomaly_Score"] >= 2).astype(int)

    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Anomalies (Consensus)", int(df["Anomaly"].sum()))
    col2.metric("📐 Z-Score Outliers", int(df["Anomaly_ZScore"].sum()))
    col3.metric("📦 IQR Outliers", int(df["Anomaly_IQR"].sum()))

    fig = px.scatter(df,
                     x=df.index, y=target,
                     color=df["Anomaly"].map({0:"Normal",1:"Anomaly"}),
                     color_discrete_map={"Normal":"#42a5f5","Anomaly":"#e53935"},
                     opacity=0.75,
                     title="Anomaly Detection — Station Usage Stats",
                     labels={"x":"Station Index",target:target})
    st.plotly_chart(fig, use_container_width=True)

    # Cost vs Rating anomaly
    df["CR_Anomaly"] = (
        (df["Cost (USD/kWh)"] > df["Cost (USD/kWh)"].quantile(0.85)) &
        (df["Reviews (Rating)"] < df["Reviews (Rating)"].quantile(0.15))
    ).astype(int)

    fig2 = px.scatter(df, x="Cost (USD/kWh)", y="Reviews (Rating)",
                      color=df["CR_Anomaly"].map({0:"Normal",1:"High Cost / Low Rating"}),
                      color_discrete_map={"Normal":"#42a5f5",
                                          "High Cost / Low Rating":"#e53935"},
                      title="High-Cost / Low-Rating Stations",
                      opacity=0.7)
    st.plotly_chart(fig2, use_container_width=True)

    # Anomaly table
    st.subheader("Anomalous Stations Detail")
    anom_cols = [c for c in ["Station ID", target, "Cost (USD/kWh)",
                              "Reviews (Rating)", "Charger Type",
                              "Station Operator", "Anomaly_Score"]
                 if c in df.columns]
    anom_df = df[df["Anomaly"]==1][anom_cols].sort_values(
        "Anomaly_Score", ascending=False)
    st.dataframe(anom_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 5 — GEO MAP
# ════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Geographic Distribution of Stations")
    if "Latitude" in df.columns and "Longitude" in df.columns:
        map_df = df.dropna(subset=["Latitude","Longitude"]).copy()
        fig_map = px.scatter_mapbox(
            map_df,
            lat="Latitude", lon="Longitude",
            color="Cluster" if "Cluster" in map_df.columns else "Charger Type",
            size="Usage Stats (avg users/day)",
            hover_name="Station ID" if "Station ID" in map_df.columns else None,
            hover_data={"Latitude":False,"Longitude":False,
                        "Usage Stats (avg users/day)":True,
                        "Charger Type":True,
                        "Reviews (Rating)":True},
            zoom=2, height=550,
            mapbox_style="open-street-map",
            title="EV Charging Stations — Clustered by Usage & Type"
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Heatmap layer
        st.subheader("Usage Demand Heatmap")
        heat_data = (map_df[["Latitude","Longitude",
                              "Usage Stats (avg users/day)"]]
                     .rename(columns={"Latitude":"lat","Longitude":"lon",
                                      "Usage Stats (avg users/day)":"weight"}))
        fig_heat = px.density_mapbox(
            heat_data, lat="lat", lon="lon", z="weight",
            radius=20, zoom=2, height=500,
            mapbox_style="stamen-terrain",
            color_continuous_scale="YlOrRd",
            title="Demand Heatmap — Avg Users/Day"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No Latitude/Longitude columns found in dataset.")


# ════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS
# ════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("📋 Key Insights & Recommendations")

    if "Charger Type" in df.columns:
        top_charger = (df.groupby("Charger Type")
                         ["Usage Stats (avg users/day)"]
                         .mean().idxmax())
        top_val = (df.groupby("Charger Type")
                     ["Usage Stats (avg users/day)"]
                     .mean().max())
        st.success(f"⚡ **Most Popular Charger:** {top_charger} "
                   f"({top_val:.1f} avg users/day)")

    if "Station Operator" in df.columns:
        best_op = (df.groupby("Station Operator")["Reviews (Rating)"]
                     .mean().idxmax())
        st.info(f"🏆 **Best-Rated Operator:** {best_op} "
                f"(avg rating: "
                f"{df.groupby('Station Operator')['Reviews (Rating)'].mean().max():.2f})")

    if "Distance to City (km)" in df.columns:
        med = df["Distance to City (km)"].median()
        city_avg = df[df["Distance to City (km)"]<=med]["Usage Stats (avg users/day)"].mean()
        rural_avg = df[df["Distance to City (km)"]>med]["Usage Stats (avg users/day)"].mean()
        st.warning(f"🏙️ City stations avg **{city_avg:.1f}** users/day vs "
                   f"rural **{rural_avg:.1f}** users/day")

    if "Anomaly" in df.columns:
        n_a = df["Anomaly"].sum()
        st.error(f"🚨 **{n_a}** anomalous stations detected — "
                 f"recommend operational audit")

    if "Renewable Energy Source" in df.columns:
        ren = df.groupby("Renewable Energy Source")[
            "Usage Stats (avg users/day)"].mean()
        st.success(f"🌿 Renewable stations: "
                   f"{'Yes' in ren.index and f'{ren.get(\"Yes\",0):.1f}' or 'N/A'} "
                   f"avg users/day vs non-renewable: "
                   f"{'No' in ren.index and f'{ren.get(\"No\",0):.1f}' or 'N/A'}")

    st.markdown("---")
    st.subheader("Strategic Recommendations")
    recommendations = [
        ("Expand DC Fast Charger infrastructure near city centres",
         "High-demand urban hubs are underserved"),
        ("Audit high-cost, low-rated stations identified as anomalies",
         "These stations are losing customers and revenue"),
        ("Prioritise renewable-powered stations in marketing",
         "Higher user preference & sustainability alignment"),
        ("Schedule preventive maintenance for frequently serviced stations",
         "Reduce unexpected downtime and improve reliability"),
        ("Deploy additional stations in high-demand clusters",
         "Cluster analysis reveals unmet demand hotspots"),
    ]
    for i, (rec, rationale) in enumerate(recommendations, 1):
        with st.expander(f"💡 Recommendation {i}: {rec}"):
            st.write(f"**Rationale:** {rationale}")

    st.markdown("---")
    st.caption("SmartEnergy Data Lab | Data Mining Summative Assessment "
               "| Scenario 2: SmartCharging Analytics")
