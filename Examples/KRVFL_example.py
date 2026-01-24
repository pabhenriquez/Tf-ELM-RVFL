"""
Example: Kernel RVFL (Kernel Random Vector Functional Link) for Classification

This example demonstrates how to use the Kernel RVFL model, which combines
kernel methods with direct input links.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import sys
sys.path.append('..')

from Layers.KRVFLLayer import KRVFLLayer
from Models.KRVFLModel import KRVFLModel
from Resources.Kernel import Kernel, CombinedProductKernel


# Hyperparameters
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
print("Kernel RVFL Classification Example")
print("=" * 60)
print(f"Dataset: ionosphere")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Regularization (C): {regularization}")
print("=" * 60)

# Initialize Kernel RVFL with RBF kernel
kernel = Kernel("rbf", param=1.0)
krvfl_layer = KRVFLLayer(
    kernel=kernel,
    activation='mish',
    C=regularization,
    include_direct_link=True
)

# Create KRVFL model
model = KRVFLModel(krvfl_layer, classification=True)

# Train the model
print("\nTraining Kernel RVFL model...")
model.fit(X_train, y_train)

# Print model summary
model.summary()

# Evaluate
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Compare different kernels
print("\n" + "=" * 60)
print("Comparing different kernels:")
print("=" * 60)

kernels = [
    ("RBF (gamma=0.5)", Kernel("rbf", param=0.5)),
    ("RBF (gamma=1.0)", Kernel("rbf", param=1.0)),
    ("RBF (gamma=2.0)", Kernel("rbf", param=2.0)),
    ("Polynomial (d=2)", Kernel("polynomial", param=2)),
    ("Polynomial (d=3)", Kernel("polynomial", param=3)),
    ("Linear", Kernel("linear")),
    ("Sigmoid", Kernel("sigmoid", param=0.1)),
]

for name, kernel in kernels:
    layer = KRVFLLayer(kernel=kernel, C=regularization, include_direct_link=True)
    temp_model = KRVFLModel(layer, classification=True)
    temp_model.fit(X_train, y_train)
    pred = temp_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  {name:25s}: {acc:.4f}")

# Compare with and without direct link
print("\n" + "=" * 60)
print("Effect of Direct Link (KRVFL vs KELM):")
print("=" * 60)

kernel = Kernel("rbf", param=1.0)

# With direct link (KRVFL)
layer_with_link = KRVFLLayer(kernel=kernel, C=regularization, include_direct_link=True)
model_with_link = KRVFLModel(layer_with_link, classification=True)
model_with_link.fit(X_train, y_train)
acc_with_link = accuracy_score(y_test, model_with_link.predict(X_test))

# Without direct link (effectively KELM)
layer_without_link = KRVFLLayer(kernel=kernel, C=regularization, include_direct_link=False)
model_without_link = KRVFLModel(layer_without_link, classification=True)
model_without_link.fit(X_train, y_train)
acc_without_link = accuracy_score(y_test, model_without_link.predict(X_test))

print(f"  With direct link (KRVFL):    {acc_with_link:.4f}")
print(f"  Without direct link (KELM):  {acc_without_link:.4f}")

# Combined Kernel example
print("\n" + "=" * 60)
print("Combined Kernel Example:")
print("=" * 60)

combined_kernel = CombinedProductKernel([
    Kernel("rbf", param=1.0),
    Kernel("polynomial", param=2)
])

layer_combined = KRVFLLayer(kernel=combined_kernel, C=regularization, include_direct_link=True)
model_combined = KRVFLModel(layer_combined, classification=True)
model_combined.fit(X_train, y_train)
acc_combined = accuracy_score(y_test, model_combined.predict(X_test))

print(f"  Combined (RBF * Polynomial): {acc_combined:.4f}")
