import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="SHM Data Exploration", layout="wide")
st.markdown("<h1 style='text-align:center;'>📊 Structural Health Monitoring Dashboard</h1>", unsafe_allow_html=True)

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/Processed/featured_dataset_V2.csv")
    # Detect datetime column
    datetime_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_col = col
                break
            except:
                pass
    return df, datetime_col

df, datetime_col = load_data()

# Separate columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📄 Dataset Overview",
     "📌 Condition Distribution",
     "📈 Time Series Trends",
     "📊 Feature Correlation",
     "🔍 Categorical Feature Analysis"]
)

# Sidebar Filters
st.sidebar.markdown("### 📅 Date Range Filter")
if datetime_col:
    min_date, max_date = df[datetime_col].min(), df[datetime_col].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        df = df[(df[datetime_col] >= pd.to_datetime(date_range[0])) &
                (df[datetime_col] <= pd.to_datetime(date_range[1]))]

st.sidebar.markdown("### ⚙️ Advanced Settings")
show_rolling = st.sidebar.checkbox("Enable Rolling Average", value=True)
rolling_window = st.sidebar.slider("Rolling Window Size", 1, 48, 24)

# -------------------------
# Page Logic
# -------------------------
if page == "📄 Dataset Overview":
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {df.columns.tolist()}")
    st.subheader("Basic Statistics")
    st.dataframe(df[numeric_cols].describe())

elif page == "📌 Condition Distribution":
    target_col = st.selectbox("Select target column", [c for c in categorical_cols if c != datetime_col])
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x=target_col, data=df, palette="Set2", order=df[target_col].value_counts().index)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        pie_data = df[target_col].value_counts().reset_index()
        pie_data.columns = [target_col, "Count"]
        fig = px.pie(pie_data, names=target_col, values="Count", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Time Series Trends":
    if datetime_col is None:
        st.warning("No datetime column detected.")
    else:
        sensor = st.selectbox("Select Sensor", numeric_cols)
        z_thresh = st.slider("Z-score threshold", 1.0, 5.0, 3.0)

        df_sorted = df.sort_values(datetime_col).copy()
        if show_rolling:
            df_sorted[f"{sensor}_rolling"] = df_sorted[sensor].rolling(rolling_window).mean()

        # Anomaly detection
        rolling_mean = df_sorted[sensor].rolling(rolling_window).mean()
        rolling_std = df_sorted[sensor].rolling(rolling_window).std()
        df_sorted["z_score"] = (df_sorted[sensor] - rolling_mean) / rolling_std
        anomalies = df_sorted[np.abs(df_sorted["z_score"]) > z_thresh]

        # Plot
        fig = px.line(df_sorted, x=datetime_col, y=sensor, title=f"{sensor} over Time")
        if show_rolling:
            fig.add_scatter(x=df_sorted[datetime_col], y=df_sorted[f"{sensor}_rolling"], mode="lines", name="Rolling Mean")
        if not anomalies.empty:
            fig.add_scatter(x=anomalies[datetime_col], y=anomalies[sensor], mode="markers", name="Anomaly",
                            marker=dict(color="red", size=8, symbol="x"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Feature Correlation":
    st.subheader("Correlation Heatmap")
    selected_cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:5])
    if len(selected_cols) >= 2:
        corr = df[selected_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif page == "🔍 Categorical Feature Analysis":
    selected_cat = st.selectbox("Select categorical feature", [c for c in categorical_cols if c != datetime_col])
    target_for_relation = st.selectbox("Select target variable", [c for c in categorical_cols if c != datetime_col])

    fig = px.histogram(df, x=selected_cat, color=target_for_relation, barmode="group")
    st.plotly_chart(fig, use_container_width=True)
