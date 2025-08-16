import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# ==== Path setup ====
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from Models.models import HybridModel  # your model

st.header("🔮 Interactive Prediction with Time Series Insights")

# ==== Load feature info ====
df = pd.read_csv("Data/Processed/featured_dataset_V2.csv")
df = df.drop(["structural_condition"] , axis=1)
feature_ranges = {}
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        feature_ranges[col] = (float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if not feature_ranges:
    st.error("No numerical features found in dataset!")
    st.stop()

# ==== Load model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel(
    input_channels=len(feature_ranges),
    cnn_channels=32,
    lstm_hidden=32,
    lstm_layers=2,
    num_classes=4
).to(device)

weights_path = "Models/Trained_models/V2_Best_model.pth"
if not os.path.exists(weights_path):
    st.error(f"Model weights not found at {weights_path}")
    st.stop()

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# ==== Initialize prediction history ====
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=list(feature_ranges.keys()) + ["Predicted_Class"])

# ==== Sliders for user input ====
st.subheader("Set Feature Values")
input_values = {}
for feature, (min_val, max_val, mean_val) in feature_ranges.items():
    val = st.slider(
        f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=(max_val - min_val) / 100
    )
    input_values[feature] = val

# ==== Predict button ====
if st.button("Predict Condition"):
    seq_len = 1
    input_tensor = torch.tensor(
        [[list(input_values.values())] * seq_len], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    st.success(f"Predicted Class: **{pred_class}**")

    # Append to history
    new_entry = input_values.copy()
    new_entry["Predicted_Class"] = pred_class
    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, pd.DataFrame([new_entry])],
        ignore_index=True
    )

# ==== Time-series style plot ====
import matplotlib.pyplot as plt
import plotly.graph_objects as go

if not st.session_state.prediction_history.empty:
    st.subheader("📈 Time-Series with Predictions")

    hist_df = st.session_state.prediction_history.reset_index().rename(columns={"index": "Step"})

    fig = go.Figure()

    # Add feature lines
    for feature in feature_ranges.keys():
        fig.add_trace(go.Scatter(
            x=hist_df["Step"],
            y=hist_df[feature],
            mode='lines+markers',
            name=feature
        ))

    # Add predicted class as separate trace
    fig.add_trace(go.Scatter(
        x=hist_df["Step"],
        y=hist_df["Predicted_Class"],
        mode='lines+markers',
        name='Predicted Class',
        line=dict(color='red', dash='dash'),
        yaxis="y2"  # Secondary y-axis
    ))

    # Layout with secondary y-axis
    fig.update_layout(
        title="Feature Trends with Model Predictions",
        xaxis_title="Step",
        yaxis=dict(title="Feature Values"),
        yaxis2=dict(
            title="Predicted Class",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(orientation="h", y=-0.2),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
