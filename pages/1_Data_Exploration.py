import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(page_title="SHM Data Exploration", layout="wide")
st.title("📊 Structural Health Monitoring - Data Exploration")

# -------------------------
# Load Data
# -------------------------
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

# Separate column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Dataset Overview",
    "📌 Condition Distribution",
    "📈 Time Series Trends",
    "📊 Feature Correlation",
    "🔍 Categorical Feature Analysis"
])

# -------------------------
# TAB 1 - Dataset Overview
# -------------------------
with tab1:
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("**Shape of Dataset:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())

    st.subheader("Basic Statistics (Numeric Features)")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

# -------------------------
# TAB 2 - Condition Distribution
# -------------------------
with tab2:
    target_col = st.selectbox("Select target column", [c for c in categorical_cols if c != datetime_col])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Count Plot")
        fig, ax = plt.subplots()
        sns.countplot(x=target_col, data=df, palette="Set2", order=df[target_col].value_counts().index)
        ax.set_ylabel("Count")
        ax.set_xlabel(target_col)
        st.pyplot(fig)

    with col2:
        st.subheader("Pie Chart")
        pie_data = df[target_col].value_counts().reset_index()
        pie_data.columns = [target_col, "Count"]
        fig = px.pie(pie_data, names=target_col, values="Count", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TAB 3 - Time Series Trends
# -------------------------
with tab3:
    if datetime_col is None:
        st.warning("No datetime column detected. Cannot show time series trends.")
    else:
        st.subheader("Time Series Sensor Trends with Anomaly Detection")
        sensor = st.selectbox("Select sensor (numeric column)", numeric_cols)
        window = st.slider("Rolling average window (hours)", 1, 48, 24)
        z_thresh = st.slider("Z-score threshold for anomaly detection", 1.0, 5.0, 3.0)

        df_sorted = df.sort_values(datetime_col).copy()
        df_sorted[f"{sensor}_rolling"] = df_sorted[sensor].rolling(window).mean()

        # Anomaly detection
        rolling_mean = df_sorted[sensor].rolling(window).mean()
        rolling_std = df_sorted[sensor].rolling(window).std()
        df_sorted["z_score"] = (df_sorted[sensor] - rolling_mean) / rolling_std
        anomalies = df_sorted[np.abs(df_sorted["z_score"]) > z_thresh]

        # Plot sensor trend
        fig = px.line(df_sorted, x=datetime_col, y=sensor, title=f"{sensor} over Time with Anomalies")
        fig.add_scatter(x=df_sorted[datetime_col], y=df_sorted[f"{sensor}_rolling"],
                        mode="lines", name=f"Rolling Mean ({window}h)")
        if not anomalies.empty:
            fig.add_scatter(x=anomalies[datetime_col], y=anomalies[sensor],
                            mode="markers", name="Anomaly",
                            marker=dict(color="red", size=8, symbol="x"))
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Condition Progression
        # -------------------------
        if "structural_condition" in df.columns:
            st.subheader("Structural Condition Progression Over Time")

            condition_order = ["No damage", "Minor", "Moderate", "Severe"]
            df_sorted["structural_condition"] = df_sorted["structural_condition"].astype(str).str.strip().str.title()

            df_sorted["condition_numeric"] = df_sorted["structural_condition"].apply(
                lambda x: condition_order.index(x) if x in condition_order else np.nan
            )

            df_condition = df_sorted.dropna(subset=["condition_numeric"])

            if df_condition.empty:
                st.warning("No valid structural condition values found in the data.")
            else:
                cond_fig = px.line(
                    df_condition,
                    x=datetime_col,
                    y="condition_numeric",
                    markers=True,
                    title="Condition State Progression",
                    color="structural_condition",
                    color_discrete_map={
                        "No damage": "green",
                        "Minor": "yellow",
                        "Moderate": "orange",
                        "Severe": "red"
                    }
                )
                cond_fig.update_yaxes(
                    tickvals=list(range(len(condition_order))),
                    ticktext=condition_order
                )
                st.plotly_chart(cond_fig, use_container_width=True)

# -------------------------
# TAB 4 - Feature Correlation
# -------------------------
with tab4:
    st.subheader("Correlation Heatmap (Numeric Features Only)")
    selected_num_cols = st.multiselect("Select numeric columns for correlation", numeric_cols, default=numeric_cols[:5])
    if len(selected_num_cols) >= 2:
        corr = df[selected_num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select at least 2 numeric columns.")

# -------------------------
# TAB 5 - Categorical Feature Analysis
# -------------------------
with tab5:
    selected_cat = st.selectbox("Select categorical feature", [c for c in categorical_cols if c != datetime_col])

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x=selected_cat, data=df, palette="viridis", order=df[selected_cat].value_counts().index)
        plt.xticks(rotation=45)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        pie_data = df[selected_cat].value_counts().reset_index()
        pie_data.columns = [selected_cat, "Count"]
        fig = px.pie(pie_data, names=selected_cat, values="Count", hole=0.3,
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Target Relationship")
    target_for_relation = st.selectbox("Select target variable", [c for c in categorical_cols if c != datetime_col])
    if selected_cat != target_for_relation:
        fig = px.histogram(df, x=selected_cat, color=target_for_relation, barmode="group")
        st.plotly_chart(fig, use_container_width=True)
