# train_behavior_classifier.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# -------------------------
# Define the Behavior Classifier
# -------------------------
class AdvancedBehaviorClassifier(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_classes=7):
        """
        input_size: 33 landmarks x 4 = 132 features.
        num_classes: 7 behavior classes (e.g., Normal, Walking, Standing, Talking, Fighting, Chasing/Tailing, Armed)
        """
        super(AdvancedBehaviorClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# -------------------------
# Step 1: Load Data from CSV
# -------------------------
df = pd.read_csv("pose_training_data.csv")
X = df.drop("label", axis=1).values  # Shape: (num_samples, 132)
y = df["label"].values

# -------------------------
# Step 2: Encode Labels and Scale Features
# -------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # e.g., mapping strings to integers

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Step 3: Split Data into Training and Validation Sets
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# -------------------------
# Step 4: Initialize and Train the Model
# -------------------------
input_size = X_train.shape[1]  # 132
num_classes = len(np.unique(y_encoded))  # Should be 7
model = AdvancedBehaviorClassifier(input_size=input_size, hidden_size=128, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Evaluate on the validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, predicted = torch.max(val_outputs, 1)
        accuracy = (predicted == y_val_tensor).float().mean()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Accuracy: {accuracy.item():.4f}")

# -------------------------
# Step 5: Save the Model, Scaler, and Label Encoder
# -------------------------
torch.save(model.state_dict(), "behavior_classifier.pth")
with open("feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Training complete! Model saved as behavior_classifier.pth, scaler as feature_scaler.pkl, and label encoder as label_encoder.pkl.")
