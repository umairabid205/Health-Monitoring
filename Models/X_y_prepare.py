# check_data_shapes.py
import numpy as np
import pandas as pd
import os

files = {
    "v1": "Data/Processed/cleaned_bridge_dataset_V1.csv",
    "v2": "Data/Processed/featured_dataset_V2.csv",
    "v3_X": "Data/Processed/X_resampled.csv",
    "v3_y": "Data/Processed/y_resampled.csv",
}

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def to_numpy_if_df(x):
    return x.values if isinstance(x, pd.DataFrame) else np.array(x)

def prepare_X_as_3d(X):
    """
    Ensure X has shape (N, seq_len, channels).
    Rules:
      - If X.ndim == 1 -> treat as (N, 1, 1)
      - If X.ndim == 2 -> treat as (N, 1, features)  (seq_len=1, channels=features)
      - If X.ndim == 3 -> assume already (N, seq_len, channels)
      - If X.ndim > 3 -> raise
    Returns numpy array (float32).
    """
    X = to_numpy_if_df(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1, 1)
    elif X.ndim == 2:
        # shape (N, features) -> (N, seq_len=1, channels=features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
    elif X.ndim == 3:
        # assume correct order (N, seq_len, channels)
        pass
    else:
        raise ValueError(f"Unsupported X.ndim = {X.ndim}")
    return X.astype(np.float32)

def prepare_y(y):
    """
    Ensure y is 1D integer array of class indices [0..K-1].
    If y is strings or non-consecutive integers, factorize and show mapping.
    """
    y = to_numpy_if_df(y).squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    if not np.issubdtype(y.dtype, np.integer):
        uniq, inv = np.unique(y, return_inverse=True)
        mapping = {i: v for i, v in enumerate(uniq)}
        return inv.astype(np.int64), mapping
    else:
        # if integer but not 0..K-1, map to 0..K-1
        uniques = np.unique(y)
        if (uniques == np.arange(len(uniques))).all():
            return y.astype(np.int64), None
        else:
            # create mapping
            val_to_idx = {v: i for i, v in enumerate(uniques)}
            mapped = np.vectorize(val_to_idx.get)(y).astype(np.int64)
            mapping = {i: v for i, v in enumerate(uniques)}
            return mapped, mapping

# --- load ---
print("Loading files (will raise if a path is incorrect):")
df_v1 = load_csv(files["v1"])
df_v2 = load_csv(files["v2"])
x3_df = load_csv(files["v3_X"])
y3_df = load_csv(files["v3_y"])

# Quick info
print("\n--- V1 info ---")
print("Pandas df shape:", df_v1.shape)
print("Columns:", df_v1.columns.tolist()[:20])
print(df_v1.dtypes.value_counts().to_dict())
# print("Sample rows:\n", df_v1.head(3).to_dict(orient='records'))

print("\n--- V2 info ---")
print("Pandas df shape:", df_v2.shape)
print("Columns:", df_v2.columns.tolist()[:20])
print(df_v2.dtypes.value_counts().to_dict())
# print("Sample rows:\n", df_v2.head(3).to_dict(orient='records'))

print("\n--- V3 (X) info ---")
print("Pandas df shape:", x3_df.shape)
print("dtypes:", x3_df.dtypes.value_counts().to_dict())
# print("Sample rows:\n", x3_df.head(3).to_dict(orient='records'))

print("\n--- V3 (y) info ---")
print("Pandas df shape:", y3_df.shape)
print("dtypes:", y3_df.dtypes.value_counts().to_dict())
# print("Sample rows:\n", y3_df.head(10).to_dict(orient='records'))

# --- convert to numpy arrays used in your script ---
# For V1 and V2 you earlier dropped ["damage_class", "structural_condition"]
X1_raw = df_v1.drop(["damage_class", "structural_condition" ,"date" ,"time"], axis=1, errors='ignore')
y1_raw = df_v1.get("structural_condition")  # may be None if not present

X2_raw = df_v2.drop(["damage_class", "structural_condition" ,"date" ,"time"], axis=1, errors='ignore')
y2_raw = df_v2.get("structural_condition")

X3_raw = x3_df
y3_raw = y3_df

# Prepare shapes
X1 = prepare_X_as_3d(X1_raw)
X2 = prepare_X_as_3d(X2_raw)
X3 = prepare_X_as_3d(X3_raw)

y1, map1 = prepare_y(y1_raw) if y1_raw is not None else (None, None)
y2, map2 = prepare_y(y2_raw) if y2_raw is not None else (None, None)
y3, map3 = prepare_y(y3_raw)

np.save("Data/Processed/X1_features.npy" , X1)
np.save("Data/Processed/X2_features.npy" , X2)
np.save("Data/Processed/X3_features.npy" , X3)

np.save("Data/Processed/y1_labels.npy" , y1)
np.save("Data/Processed/y2_labels.npy" , y2)
np.save("Data/Processed/y3_labels.npy" , y3)


print("\n--- After prepare ---")
print("X1 shape (N, seq_len, channels):", X1.shape, "dtype:", X1.dtype)
print("X2 shape (N, seq_len, channels):", X2.shape, "dtype:", X2.dtype)
print("X3 shape (N, seq_len, channels):", X3.shape, "dtype:", X3.dtype)

print("y1:", None if y1 is None else (y1.shape, y1.dtype, "unique_classes:", np.unique(y1).tolist()))
print("y2:", None if y2 is None else (y2.shape, y2.dtype, "unique_classes:", np.unique(y2).tolist()))
print("y3:", (y3.shape, y3.dtype, "unique_classes:", np.unique(y3).tolist()))

if map1:
    print("y1 mapping (index -> original):", map1)
if map2:
    print("y2 mapping (index -> original):", map2)
if map3:
    print("y3 mapping (index -> original):", map3)

# quick sanity checks to catch common mistakes
for name, X in [("X1", X1), ("X2", X2), ("X3", X3)]:
    if X.ndim != 3:
        raise AssertionError(f"{name} is not 3D after prepare: ndim={X.ndim}")
    if X.shape[0] == 0:
        raise AssertionError(f"{name} has zero samples")

print("\nSanity checks passed. If everything looks good, paste this output here and I'll give the next block: DataLoader/tensor conversion + a small model-input check (forward pass dry run).")
