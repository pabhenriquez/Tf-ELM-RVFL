"""
Example: Deep RVFL (Deep Random Vector Functional Link) for Classification

This example demonstrates how to use the Deep RVFL model, which stacks multiple
hidden layers while maintaining direct connections from input to output.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import sys
sys.path.append('..')

from Layers.DeepRVFLLayer import DeepRVFLLayer
from Models.DeepRVFLModel import DeepRVFLModel


# Hyperparameters
num_neurons = 100
num_layers = 3
regularization = 1.0

# Loading sample dataset
path = "../Data/ionosphere.txt"
df = pd.read_csv(path, delimiter='\t').fillna(0)
X = df.values[:, 1:].astype(np.float32)
y = df.values[:, 0]

# Label encoding and feature normalization
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = preprocessing.normalize(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 60)
print("Deep RVFL Classification Example")
print("=" * 60)
print(f"Dataset: ionosphere")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Neurons per layer: {num_neurons}")
print(f"Number of layers: {num_layers}")
print(f"Regularization (C): {regularization}")
print("=" * 60)

# Initialize Deep RVFL layer
deep_rvfl_layer = DeepRVFLLayer(
    number_neurons=num_neurons,
    n_layers=num_layers,
    activation='relu',
    C=regularization,
    include_bias=True
)

# Create Deep RVFL model
model = DeepRVFLModel(deep_rvfl_layer, classification=True)

# Train the model
print("\nTraining Deep RVFL model...")
model.fit(X_train, y_train)

# Print model summary
model.summary()

# Evaluate
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Compare different layer configurations
print("\n" + "=" * 60)
print("Comparing different layer configurations:")
print("=" * 60)

layer_configs = [
    (50, 2),
    (100, 2),
    (50, 3),
    (100, 3),
    (50, 5),
    (100, 5),
]

for neurons, layers in layer_configs:
    layer = DeepRVFLLayer(number_neurons=neurons, n_layers=layers, activation='relu', C=regularization)
    temp_model = DeepRVFLModel(layer, classification=True)
    temp_model.fit(X_train, y_train)
    pred = temp_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  Neurons: {neurons:3d}, Layers: {layers}: Accuracy = {acc:.4f}")

# Regression example
print("\n" + "=" * 60)
print("Deep RVFL Regression Example")
print("=" * 60)

from sklearn.datasets import load_diabetes

# Load diabetes dataset for regression
diabetes = load_diabetes()
X_reg = diabetes.data.astype(np.float32)
y_reg = diabetes.target.astype(np.float32)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Normalize
X_train_reg = preprocessing.normalize(X_train_reg)
X_test_reg = preprocessing.normalize(X_test_reg)

# Create regression model
layer_reg = DeepRVFLLayer(number_neurons=50, n_layers=2, activation='relu', C=1.0)
model_reg = DeepRVFLModel(layer_reg, classification=False)

model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)

mae = np.mean(np.abs(y_pred_reg.flatten() - y_test_reg))
print(f"Mean Absolute Error (MAE): {mae:.4f}")
