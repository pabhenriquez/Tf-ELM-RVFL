"""
Example: Ensemble Deep RVFL for Classification

This example demonstrates how to use the Ensemble Deep RVFL model, which trains
separate output weights for each layer and combines predictions through voting
or averaging.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import sys
sys.path.append('..')

from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer
from Models.EnsembleDeepRVFLModel import EnsembleDeepRVFLModel


# Hyperparameters
num_neurons = 100
num_layers = 5  # Number of ensemble members
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
print("Ensemble Deep RVFL Classification Example")
print("=" * 60)
print(f"Dataset: ionosphere")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Neurons per layer: {num_neurons}")
print(f"Ensemble members (layers): {num_layers}")
print(f"Regularization (C): {regularization}")
print("=" * 60)

# Initialize Ensemble Deep RVFL layer
ensemble_layer = EnsembleDeepRVFLLayer(
    number_neurons=num_neurons,
    n_layers=num_layers,
    activation='relu',
    C=regularization,
    include_bias=True
)

# Create Ensemble Deep RVFL model with voting
model_vote = EnsembleDeepRVFLModel(ensemble_layer, classification=True, ensemble_method='vote')

# Train the model
print("\nTraining Ensemble Deep RVFL model...")
model_vote.fit(X_train, y_train)

# Print model summary
model_vote.summary()

# Evaluate with both voting and addition methods
results = model_vote.evaluate(X_test, y_test)

print(f"\nResults:")
print(f"  Voting Accuracy:   {results['vote_accuracy']:.4f}")
print(f"  Addition Accuracy: {results['addition_accuracy']:.4f}")

# Get detailed predictions
vote_pred, (add_pred, add_proba) = model_vote.predict_all(X_test)
print(f"\nVoting predictions (first 10): {vote_pred[:10]}")
print(f"Addition predictions (first 10): {add_pred[:10]}")

# Compare different ensemble sizes
print("\n" + "=" * 60)
print("Comparing different ensemble sizes:")
print("=" * 60)

ensemble_sizes = [2, 3, 5, 7, 10]

for size in ensemble_sizes:
    layer = EnsembleDeepRVFLLayer(
        number_neurons=num_neurons,
        n_layers=size,
        activation='relu',
        C=regularization
    )
    temp_model = EnsembleDeepRVFLModel(layer, classification=True)
    temp_model.fit(X_train, y_train)
    results = temp_model.evaluate(X_test, y_test)
    print(f"  Ensemble size {size:2d}: Vote={results['vote_accuracy']:.4f}, Add={results['addition_accuracy']:.4f}")

# Regression example
print("\n" + "=" * 60)
print("Ensemble Deep RVFL Regression Example")
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
layer_reg = EnsembleDeepRVFLLayer(number_neurons=50, n_layers=5, activation='relu', C=1.0)
model_reg = EnsembleDeepRVFLModel(layer_reg, classification=False)

model_reg.fit(X_train_reg, y_train_reg)
mae = model_reg.evaluate(X_test_reg, y_test_reg)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
