"""
Urban Traffic Zone Analyser — Bangalore
K-Means + DBSCAN Clustering | Streamlit Frontend

Dataset: Bangalore City Traffic Dataset (Kaggle)
https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="Urban Traffic Zone Analyser — Bangalore",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #1f77b4, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────── Helper Functions ──────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and return the traffic dataset."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame):
    """
    Encode categoricals, build the feature matrix, scale it,
    and compute 2-D / 3-D PCA projections.
    Returns (df_model, X, X_scaled, X_pca, X_pca3, pca2, pca3, scaler, le, FEATURES).
    """
    df_model = df.copy()

    # Binary encode roadwork
    df_model["Roadwork_enc"] = (
        df_model["Roadwork and Construction Activity"] == "Yes"
    ).astype(int)

    # Label-encode weather
    le = LabelEncoder()
    df_model["Weather_enc"] = le.fit_transform(df_model["Weather Conditions"])

    FEATURES = [
        "Traffic Volume", "Average Speed", "Congestion Level",
        "Road Capacity Utilization", "Travel Time Index", "Incident Reports",
        "Environmental Impact", "Public Transport Usage",
        "Traffic Signal Compliance", "Parking Usage",
        "Pedestrian and Cyclist Count", "Roadwork_enc", "Weather_enc",
    ]

    X = df_model[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_scaled)

    pca3 = PCA(n_components=3, random_state=42)
    X_pca3 = pca3.fit_transform(X_scaled)

    return df_model, X, X_scaled, X_pca, X_pca3, pca2, pca3, scaler, le, FEATURES


def label_zone(congestion: float) -> str:
    """Map mean congestion level to a human-readable zone label."""
    if congestion >= 90:
        return "Critical Zone"
    elif congestion >= 70:
        return "High Traffic Zone"
    elif congestion >= 40:
        return "Moderate Zone"
    else:
        return "Free Flow Zone"


# ─────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-jam.png", width=80)
    st.markdown("## 🚦 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "📊 EDA",
            "🔬 K-Means Clustering",
            "🔎 DBSCAN Clustering",
            "⚖️ Model Comparison",
            "📋 Metrics Summary",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Parameters")
    optimal_k = st.slider("K-Means — Number of clusters (k)", 2, 10, 2)
    eps_val = st.slider("DBSCAN — eps", 0.5, 3.0, 1.8, 0.1)
    min_samples_val = st.slider("DBSCAN — min_samples", 3, 30, 5)

    st.markdown("---")
    st.markdown(
        "<small>Dataset: <a href='https://www.kaggle.com/datasets/"
        "preethamgouda/banglore-city-traffic-dataset' target='_blank'>"
        "Kaggle — Bangalore Traffic</a></small>",
        unsafe_allow_html=True,
    )

# ─────────────────────── Load & Process Data ────────────────────────
DATA_PATH = "data/Banglore_traffic_Dataset.csv"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"**Dataset not found at `{DATA_PATH}`.**\n\n"
        "Please download the CSV from "
        "[Kaggle](https://www.kaggle.com/datasets/preethamgouda/"
        "banglore-city-traffic-dataset) and place it inside the `data/` folder."
    )
    st.stop()

(
    df_model, X, X_scaled, X_pca, X_pca3,
    pca2, pca3, scaler, le, FEATURES
) = engineer_features(df)

# ──────────────────── Run clustering models ─────────────────────────
# K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_model["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

profile_cols = [
    "Traffic Volume", "Average Speed", "Congestion Level",
    "Road Capacity Utilization", "Travel Time Index", "Incident Reports",
]
cluster_profile = df_model.groupby("KMeans_Cluster")[profile_cols].mean().round(2)
cluster_profile["Zone Label"] = cluster_profile["Congestion Level"].apply(label_zone)
zone_map = cluster_profile["Zone Label"].to_dict()
df_model["Zone Label"] = df_model["KMeans_Cluster"].map(zone_map)

# DBSCAN
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
db_labels = dbscan.fit_predict(X_scaled)
df_model["DBSCAN_Label"] = db_labels
df_model["Is_Anomaly"] = (db_labels == -1).astype(int)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db = (db_labels == -1).sum()


# ════════════════════════════════════════════════════════════════════
#                           PAGES
# ════════════════════════════════════════════════════════════════════

# ─────────────────────── 🏠 HOME ───────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-header">🚦 Urban Traffic Zone Analyser</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">K-Means + DBSCAN Clustering on Bangalore City Traffic Data</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>{df.shape[0]:,}</h3><p>Total Records</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h3>{df["Area Name"].nunique()}</h3><p>Bengaluru Areas</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Features</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><h3>{df["Weather Conditions"].nunique()}</h3><p>Weather Types</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🎯 Project Objective")
        st.markdown(
            "Cluster **Bengaluru urban zones** by traffic behaviour using "
            "**K-Means** (zone segmentation) and **DBSCAN** (anomaly / hotspot detection) "
            "on the [Bangalore City Traffic Dataset](https://www.kaggle.com/datasets/"
            "preethamgouda/banglore-city-traffic-dataset)."
        )

        st.markdown("### 📐 Methodology")
        st.markdown(
            "1. **EDA** — distributions, correlations, area-wise congestion\n"
            "2. **Feature Engineering** — encode categoricals, standard-scale numerics\n"
            "3. **PCA** — reduce to 2-D / 3-D for visualisation\n"
            "4. **K-Means** — elbow + silhouette to pick optimal k, then profile clusters\n"
            "5. **DBSCAN** — k-distance graph + grid search for eps / min_samples, "
            "then flag anomalies"
        )

    with col_right:
        st.markdown("### 📁 Dataset Features")
        st.dataframe(
            pd.DataFrame({
                "Column": df.columns.tolist(),
                "Dtype": df.dtypes.astype(str).tolist(),
                "Non-Null": df.notnull().sum().tolist(),
            }),
            use_container_width=True,
            height=400,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)


# ──────────────────────── 📊 EDA ───────────────────────────────────
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Descriptive Statistics ──
    st.markdown("### 📈 Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    # ── Feature Distributions ──
    st.markdown("### 📉 Feature Distributions")
    num_cols = [
        "Traffic Volume", "Average Speed", "Congestion Level",
        "Road Capacity Utilization", "Travel Time Index", "Incident Reports",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
    for ax, col in zip(axes.flatten(), num_cols):
        sns.histplot(df[col], ax=ax, kde=True, color="steelblue")
        ax.set_title(col)
        ax.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig)

    # ── Congestion by Area ──
    st.markdown("### 🏙️ Average Congestion Level by Area")
    area_congestion = (
        df.groupby("Area Name")["Congestion Level"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_bar = px.bar(
        area_congestion, x="Area Name", y="Congestion Level",
        color="Congestion Level",
        color_continuous_scale="Reds",
        title="Average Congestion Level by Area",
    )
    fig_bar.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown("### 🔗 Correlation Heatmap")
    corr_cols = [
        "Traffic Volume", "Average Speed", "Congestion Level",
        "Road Capacity Utilization", "Travel Time Index",
        "Incident Reports", "Environmental Impact",
        "Public Transport Usage", "Pedestrian and Cyclist Count",
    ]
    corr = df[corr_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, vmin=-1, vmax=1, ax=ax2,
    )
    ax2.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)

    # ── Weather vs Congestion ──
    st.markdown("### 🌦️ Congestion Level by Weather Condition")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Weather Conditions", y="Congestion Level", palette="Set2", ax=ax3)
    ax3.set_title("Congestion Level by Weather Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)


# ────────────────── 🔬 K-Means Clustering ──────────────────────────
elif page == "🔬 K-Means Clustering":
    st.markdown("## 🔬 K-Means Clustering")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Optimal-K selection charts ──
    st.markdown("### 📐 Optimal K Selection")
    K_RANGE = range(2, 11)
    inertias, silhouettes, db_scores = [], [], []
    for k in K_RANGE:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_labels = km_temp.fit_predict(X_scaled)
        inertias.append(km_temp.inertia_)
        silhouettes.append(silhouette_score(X_scaled, temp_labels))
        db_scores.append(davies_bouldin_score(X_scaled, temp_labels))

    fig_sel = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Elbow Method (Inertia)", "Silhouette Score (↑)", "Davies-Bouldin (↓)"],
    )
    fig_sel.add_trace(go.Scatter(x=list(K_RANGE), y=inertias, mode="lines+markers", name="Inertia"), row=1, col=1)
    fig_sel.add_trace(go.Scatter(x=list(K_RANGE), y=silhouettes, mode="lines+markers", name="Silhouette", line=dict(color="green")), row=1, col=2)
    fig_sel.add_trace(go.Scatter(x=list(K_RANGE), y=db_scores, mode="lines+markers", name="DB Score", line=dict(color="red")), row=1, col=3)
    # Highlight selected k
    fig_sel.add_vline(x=optimal_k, line_dash="dash", line_color="orange", row=1, col=1)
    fig_sel.add_vline(x=optimal_k, line_dash="dash", line_color="orange", row=1, col=2)
    fig_sel.add_vline(x=optimal_k, line_dash="dash", line_color="orange", row=1, col=3)
    fig_sel.update_layout(height=400, showlegend=False, title_text=f"Selected k = {optimal_k} (orange line)")
    st.plotly_chart(fig_sel, use_container_width=True)

    # ── Cluster Profiles ──
    st.markdown("### 🏷️ Cluster Profiles & Zone Labels")
    display_profile = cluster_profile.copy()
    display_profile.index = [f"Cluster {i}: {zone_map[i]}" for i in display_profile.index]
    st.dataframe(display_profile, use_container_width=True)

    # ── Cluster distribution ──
    st.markdown("### 📊 Cluster Distribution")
    dist = df_model["KMeans_Cluster"].value_counts().sort_index().reset_index()
    dist.columns = ["Cluster", "Count"]
    dist["Zone"] = dist["Cluster"].map(zone_map)
    fig_dist = px.bar(dist, x="Zone", y="Count", color="Zone",
                      color_discrete_sequence=px.colors.qualitative.Set1,
                      title="Number of Records per Zone")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── PCA 2D Scatter ──
    st.markdown("### 🗺️ PCA 2-D Cluster Scatter")
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Cluster": df_model["KMeans_Cluster"].astype(str),
        "Zone": df_model["Zone Label"],
        "Area": df_model["Area Name"],
    })
    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="Zone",
        hover_data=["Area"],
        title=f"K-Means Clusters (k={optimal_k}) — PCA 2D",
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    # Add centroids
    centroids_pca = pca2.transform(kmeans.cluster_centers_)
    fig_pca.add_trace(go.Scatter(
        x=centroids_pca[:, 0], y=centroids_pca[:, 1],
        mode="markers", marker=dict(symbol="x", size=16, color="black", line=dict(width=2)),
        name="Centroids",
    ))
    fig_pca.update_layout(
        xaxis_title=f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)",
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Cluster Profile Heatmap ──
    st.markdown("### 🔥 Cluster Profile Heatmap")
    heatmap_data = cluster_profile[profile_cols]
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-9)
    fig_hm, ax_hm = plt.subplots(figsize=(12, max(3, optimal_k * 1.2)))
    heatmap_norm_display = heatmap_norm.copy()
    heatmap_norm_display.index = [f"C{i}: {zone_map[i]}" for i in heatmap_norm_display.index]
    sns.heatmap(
        heatmap_norm_display, annot=heatmap_data.values, fmt=".1f",
        cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Normalized Value"}, ax=ax_hm,
    )
    ax_hm.set_title("Cluster Profile Heatmap (K-Means)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_hm)

    # ── Area × Zone Distribution ──
    st.markdown("### 🏙️ Zone Distribution per Area")
    area_cluster = (
        pd.crosstab(df_model["Area Name"], df_model["Zone Label"], normalize="index") * 100
    )
    fig_ac, ax_ac = plt.subplots(figsize=(12, 6))
    area_cluster.plot(kind="bar", stacked=True, colormap="Set2", ax=ax_ac)
    ax_ac.set_title("Zone Label Distribution per Bengaluru Area", fontsize=14, fontweight="bold")
    ax_ac.set_ylabel("Percentage (%)")
    ax_ac.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig_ac)


# ──────────────────── 🔎 DBSCAN Clustering ─────────────────────────
elif page == "🔎 DBSCAN Clustering":
    st.markdown("## 🔎 DBSCAN Clustering")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── k-Distance Graph ──
    st.markdown("### 📏 k-Distance Graph (eps selection helper)")
    nbrs = NearestNeighbors(n_neighbors=min_samples_val).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])[::-1]
    fig_kd, ax_kd = plt.subplots(figsize=(10, 5))
    ax_kd.plot(k_distances, color="steelblue")
    ax_kd.axhline(y=eps_val, color="red", linestyle="--", label=f"eps = {eps_val}")
    ax_kd.set_title("k-Distance Graph for DBSCAN eps Selection", fontsize=14, fontweight="bold")
    ax_kd.set_xlabel("Points (sorted)")
    ax_kd.set_ylabel(f"{min_samples_val}-NN Distance")
    ax_kd.legend()
    ax_kd.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_kd)

    # ── DBSCAN summary metrics ──
    st.markdown("### 📋 DBSCAN Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters Found", n_clusters_db)
    c2.metric("Noise Points", f"{n_noise_db} ({n_noise_db / len(db_labels) * 100:.1f}%)")
    non_noise = db_labels != -1
    if n_clusters_db >= 2 and non_noise.sum() > 1:
        db_sil = silhouette_score(X_scaled[non_noise], db_labels[non_noise])
        c3.metric("Silhouette (excl. noise)", f"{db_sil:.4f}")
        db_dbi = davies_bouldin_score(X_scaled[non_noise], db_labels[non_noise])
        c4.metric("Davies-Bouldin (excl. noise)", f"{db_dbi:.4f}")
    else:
        c3.metric("Silhouette", "N/A (< 2 clusters)")
        c4.metric("Davies-Bouldin", "N/A")

    # ── Label distribution ──
    st.markdown("### 📊 Label Distribution")
    label_dist = pd.Series(db_labels).value_counts().sort_index().reset_index()
    label_dist.columns = ["Label", "Count"]
    label_dist["Type"] = label_dist["Label"].apply(lambda x: "Noise / Anomaly" if x == -1 else f"Cluster {x}")
    fig_ld = px.bar(label_dist, x="Type", y="Count", color="Type",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title="DBSCAN Label Distribution")
    st.plotly_chart(fig_ld, use_container_width=True)

    # ── PCA 2D Scatter ──
    st.markdown("### 🗺️ PCA 2-D DBSCAN Scatter")
    db_pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Label": ["Noise / Anomaly" if l == -1 else f"Cluster {l}" for l in db_labels],
        "Area": df_model["Area Name"],
    })
    fig_db = px.scatter(
        db_pca_df, x="PC1", y="PC2", color="Label",
        hover_data=["Area"],
        title=f"DBSCAN Clusters — PCA 2D (eps={eps_val}, min_samples={min_samples_val})",
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_db.update_layout(
        xaxis_title=f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)",
    )
    st.plotly_chart(fig_db, use_container_width=True)

    # ── Anomaly Profile ──
    st.markdown("### 🚨 Anomaly vs Normal — Feature Comparison")
    anomaly_profile = df_model.groupby("Is_Anomaly")[profile_cols].mean().round(2)
    anomaly_profile.index = ["Normal", "Anomaly"]
    st.dataframe(anomaly_profile, use_container_width=True)

    fig_ap, axes_ap = plt.subplots(2, 3, figsize=(16, 9))
    fig_ap.suptitle("Normal vs Anomaly — Feature Comparison (DBSCAN)", fontsize=14, fontweight="bold")
    for ax, col in zip(axes_ap.flatten(), profile_cols):
        sns.boxplot(data=df_model, x="Is_Anomaly", y=col, palette=["steelblue", "tomato"], ax=ax)
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_title(col)
        ax.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig_ap)

    # ── Anomalies by Area & Weather ──
    st.markdown("### 🌍 Anomalies by Area & Weather")
    col_a, col_b = st.columns(2)
    with col_a:
        anomaly_area = df_model[df_model["Is_Anomaly"] == 1]["Area Name"].value_counts().reset_index()
        anomaly_area.columns = ["Area", "Count"]
        fig_aa = px.bar(anomaly_area, x="Area", y="Count", color="Count",
                        color_continuous_scale="Reds", title="Anomaly Count by Area")
        fig_aa.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_aa, use_container_width=True)
    with col_b:
        anomaly_weather = df_model[df_model["Is_Anomaly"] == 1]["Weather Conditions"].value_counts().reset_index()
        anomaly_weather.columns = ["Weather", "Count"]
        fig_aw = px.bar(anomaly_weather, x="Weather", y="Count", color="Count",
                        color_continuous_scale="Blues", title="Anomaly Count by Weather")
        st.plotly_chart(fig_aw, use_container_width=True)


# ────────────────── ⚖️ Model Comparison ─────────────────────────────
elif page == "⚖️ Model Comparison":
    st.markdown("## ⚖️ K-Means vs DBSCAN — Side-by-Side Comparison")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # K-Means scatter
    with col1:
        km_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "Cluster": df_model["KMeans_Cluster"].astype(str),
            "Zone": df_model["Zone Label"],
        })
        fig_km = px.scatter(
            km_df, x="PC1", y="PC2", color="Zone", opacity=0.5,
            title=f"K-Means (k={optimal_k})",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_km.update_layout(height=500)
        st.plotly_chart(fig_km, use_container_width=True)

    # DBSCAN scatter
    with col2:
        db_df2 = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "Label": ["Noise" if l == -1 else f"Cluster {l}" for l in db_labels],
        })
        fig_db2 = px.scatter(
            db_df2, x="PC1", y="PC2", color="Label", opacity=0.5,
            title=f"DBSCAN (eps={eps_val}, min_s={min_samples_val})",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_db2.update_layout(height=500)
        st.plotly_chart(fig_db2, use_container_width=True)

    # ── 3D Comparison ──
    st.markdown("### 🌐 3-D PCA Comparison")
    tab1, tab2 = st.tabs(["K-Means 3D", "DBSCAN 3D"])

    with tab1:
        km3d = pd.DataFrame({
            "PC1": X_pca3[:, 0], "PC2": X_pca3[:, 1], "PC3": X_pca3[:, 2],
            "Zone": df_model["Zone Label"], "Area": df_model["Area Name"],
        })
        fig_3d_km = px.scatter_3d(
            km3d, x="PC1", y="PC2", z="PC3", color="Zone",
            hover_data=["Area"], opacity=0.5,
            title=f"K-Means 3D (k={optimal_k})",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_3d_km.update_layout(height=600)
        st.plotly_chart(fig_3d_km, use_container_width=True)

    with tab2:
        db3d = pd.DataFrame({
            "PC1": X_pca3[:, 0], "PC2": X_pca3[:, 1], "PC3": X_pca3[:, 2],
            "Label": ["Noise" if l == -1 else f"Cluster {l}" for l in db_labels],
            "Area": df_model["Area Name"],
        })
        fig_3d_db = px.scatter_3d(
            db3d, x="PC1", y="PC2", z="PC3", color="Label",
            hover_data=["Area"], opacity=0.5,
            title=f"DBSCAN 3D (eps={eps_val}, min_s={min_samples_val})",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_3d_db.update_layout(height=600)
        st.plotly_chart(fig_3d_db, use_container_width=True)

    # ── Textual comparison ──
    st.markdown("### 🆚 Algorithm Comparison")
    comp_data = {
        "Aspect": [
            "Type", "Requires K?", "Handles Noise?",
            "Cluster Shape", "Best Use Case",
        ],
        "K-Means": [
            "Centroid-based", "Yes", "No",
            "Spherical / convex", "Zone segmentation — structured clusters",
        ],
        "DBSCAN": [
            "Density-based", "No (uses eps, min_samples)", "Yes (labels as -1)",
            "Arbitrary", "Anomaly / hotspot detection — noise = alert",
        ],
    }
    st.table(pd.DataFrame(comp_data))


# ───────────────── 📋 Metrics Summary ──────────────────────────────
elif page == "📋 Metrics Summary":
    st.markdown("## 📋 Final Model Metrics Summary")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # K-Means metrics
    km_sil = silhouette_score(X_scaled, df_model["KMeans_Cluster"])
    km_db_score = davies_bouldin_score(X_scaled, df_model["KMeans_Cluster"])
    km_ch = calinski_harabasz_score(X_scaled, df_model["KMeans_Cluster"])

    st.markdown("### 🟢 K-Means")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Clusters", optimal_k)
    mc2.metric("Silhouette ↑", f"{km_sil:.4f}")
    mc3.metric("Davies-Bouldin ↓", f"{km_db_score:.4f}")
    mc4.metric("Calinski-Harabasz ↑", f"{km_ch:.2f}")

    # DBSCAN metrics
    st.markdown("### 🔵 DBSCAN")
    non_noise = db_labels != -1
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Clusters", n_clusters_db)
    dc2.metric("Noise Points", f"{n_noise_db} ({n_noise_db / len(db_labels) * 100:.1f}%)")
    if n_clusters_db >= 2 and non_noise.sum() > 1:
        db_sil_final = silhouette_score(X_scaled[non_noise], db_labels[non_noise])
        db_db_final = davies_bouldin_score(X_scaled[non_noise], db_labels[non_noise])
        dc3.metric("Silhouette ↑ (excl. noise)", f"{db_sil_final:.4f}")
        dc4.metric("Davies-Bouldin ↓ (excl. noise)", f"{db_db_final:.4f}")
    else:
        dc3.metric("Silhouette", "N/A")
        dc4.metric("Davies-Bouldin", "N/A")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Interpretation ──
    st.markdown("### 📝 Interpretation")
    st.info(
        "**K-Means** is used for **structured zone segmentation** — it partitions "
        "all data points into well-defined traffic zones (Critical, High, Moderate, Free Flow).\n\n"
        "**DBSCAN** is used for **anomaly / hotspot detection** — data points flagged as "
        "noise (label = -1) represent unusual traffic patterns that warrant investigation."
    )

    # ── Zone labels table ──
    st.markdown("### 🏷️ Zone Label Definitions")
    st.table(pd.DataFrame({
        "Zone Label": ["Critical Zone", "High Traffic Zone", "Moderate Zone", "Free Flow Zone"],
        "Congestion Level": ["≥ 90", "70 – 89", "40 – 69", "< 40"],
        "Interpretation": [
            "Severe congestion, near-full road capacity",
            "Heavy traffic, significant delays",
            "Moderate delays, manageable traffic",
            "Minimal traffic, smooth flow",
        ],
    }))

    # ── Export clustered data ──
    st.markdown("### 💾 Download Clustered Dataset")
    export_cols = [
        "Area Name", "Road/Intersection Name",
        "KMeans_Cluster", "Zone Label", "DBSCAN_Label", "Is_Anomaly",
    ]
    csv_export = df_model[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download bangalore_traffic_clustered.csv",
        csv_export,
        "bangalore_traffic_clustered.csv",
        "text/csv",
    )

# ─────────────────────── Footer ─────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
    "Urban Traffic Zone Analyser — SG 4 | K-Means + DBSCAN | "
    "Dataset: <a href='https://www.kaggle.com/datasets/preethamgouda/"
    "banglore-city-traffic-dataset'>Kaggle</a>"
    "</div>",
    unsafe_allow_html=True,
)
