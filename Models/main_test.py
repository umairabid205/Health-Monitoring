import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Go two levels up from this file to reach project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from Models.models import HybridModel 



# Load test data

data = np.load("Data\Processed\Testing_Prepared_Data\V2_featured_test.npz")
X= data["X_test"]
y= data["y_test"]



# Convert to torch tensors
X_test_tensor = torch.tensor(X, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.long)



# Initialize model

# ==== Model Run ====

model = HybridModel(
    input_channels=19,
    cnn_channels=64,
    lstm_hidden=128,
    lstm_layers=3,
    num_classes=4
)# pass the best parameters from optuna used in train


# Load the saved state_dict
model.load_state_dict(torch.load("Models/Trained_models/V2_Best_model.pth"))
model.eval()



#  Run inference
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)


# Convert predictions to numpy
y_pred = predicted.cpu().numpy()
y_true = y_test_tensor.cpu().numpy()



# Print evaluation metrics
# Evaluation metrics
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_true, y_pred))




# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = sorted(np.unique(y_true))




# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()




# Plot Class-wise Precision, Recall, F1
report = classification_report(y_true, y_pred, output_dict=True)
metrics_df = (
    np.array([[report[str(c)]["precision"], report[str(c)]["recall"], report[str(c)]["f1-score"]]
              for c in class_labels])
)

metrics_names = ["Precision", "Recall", "F1-score"]

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics_names):
    plt.plot(class_labels, metrics_df[:, i], marker='o', label=metric)

plt.ylim(0, 1.05)
plt.xlabel("Class Label")
plt.ylabel("Score")
plt.title("Class-wise Metrics")
plt.legend()
plt.grid(True)
plt.show()