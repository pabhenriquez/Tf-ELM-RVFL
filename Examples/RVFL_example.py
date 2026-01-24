"""
Example: Basic RVFL (Random Vector Functional Link) for Classification

This example demonstrates how to use the RVFL model for a classification task.
RVFL differs from ELM by including direct connections from input to output.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import sys
sys.path.append('..')

from Layers.RVFLLayer import RVFLLayer
from Models.RVFLModel import RVFLModel


# Hyperparameters
num_neurons = 500
n_splits = 5
n_repeats = 10
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

print("=" * 60)
print("RVFL Classification Example")
print("=" * 60)
print(f"Dataset: ionosphere")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")
print(f"Neurons: {num_neurons}")
print(f"Regularization (C): {regularization}")
print("=" * 60)

# Initialize RVFL layer
rvfl_layer = RVFLLayer(
    number_neurons=num_neurons,
    activation='mish',
    C=regularization,
    include_bias=True
)

# Create RVFL model
model = RVFLModel(rvfl_layer, classification=True)

# Cross-validation
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')

print(f"\nCross-validation results ({n_splits}-fold, {n_repeats} repeats):")
print(f"Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Train on full dataset and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTrain/Test Split Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save("../Saved_Models/RVFL_Model.h5")
print("\nModel saved to: Saved_Models/RVFL_Model.h5")

# Compare with different activations
print("\n" + "=" * 60)
print("Comparing different activation functions:")
print("=" * 60)

activations = ['relu', 'tanh', 'sigmoid', 'mish', 'gelu']
for act in activations:
    layer = RVFLLayer(number_neurons=num_neurons, activation=act, C=regularization)
    temp_model = RVFLModel(layer, classification=True)
    temp_model.fit(X_train, y_train)
    pred = temp_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  {act:12s}: {acc:.4f}")
