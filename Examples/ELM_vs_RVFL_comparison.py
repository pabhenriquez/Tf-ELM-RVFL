"""
Example: Comparison between ELM and RVFL

This example compares ELM (Extreme Learning Machine) and RVFL (Random Vector
Functional Link) models to demonstrate the effect of direct input links.

Key difference:
- ELM: Output is computed from hidden layer only: y = H * beta
- RVFL: Output includes direct input link: y = [H, X] * beta
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris

import sys
sys.path.append('..')

# Import ELM
from Layers.ELMLayer import ELMLayer
from Models.ELMModel import ELMModel

# Import RVFL
from Layers.RVFLLayer import RVFLLayer
from Models.RVFLModel import RVFLModel

# Import Deep variants
from Layers.DeepRVFLLayer import DeepRVFLLayer
from Models.DeepRVFLModel import DeepRVFLModel


def compare_on_dataset(X, y, dataset_name, task='classification'):
    """Compare ELM and RVFL on a given dataset."""

    X = X.astype(np.float32)
    X = preprocessing.normalize(X)

    if task == 'classification':
        if len(y.shape) == 1 and not np.issubdtype(y.dtype, np.floating):
            pass  # Already encoded
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Task: {task}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"{'='*60}")

    results = {}

    # Different neuron counts to test
    neuron_counts = [50, 100, 200, 500]

    print(f"\n{'Neurons':<10} {'ELM':>12} {'RVFL':>12} {'Improvement':>12}")
    print("-" * 50)

    for neurons in neuron_counts:
        # ELM
        elm_layer = ELMLayer(number_neurons=neurons, activation='mish', C=1.0)
        elm_model = ELMModel(elm_layer, classification=(task=='classification'))
        elm_model.fit(X_train, y_train)
        elm_pred = elm_model.predict(X_test)

        # RVFL
        rvfl_layer = RVFLLayer(number_neurons=neurons, activation='mish', C=1.0)
        rvfl_model = RVFLModel(rvfl_layer, classification=(task=='classification'))
        rvfl_model.fit(X_train, y_train)
        rvfl_pred = rvfl_model.predict(X_test)

        if task == 'classification':
            elm_score = accuracy_score(y_test, elm_pred)
            rvfl_score = accuracy_score(y_test, rvfl_pred)
            improvement = (rvfl_score - elm_score) * 100
            print(f"{neurons:<10} {elm_score:>12.4f} {rvfl_score:>12.4f} {improvement:>+11.2f}%")
        else:
            elm_score = np.sqrt(mean_squared_error(y_test, elm_pred))
            rvfl_score = np.sqrt(mean_squared_error(y_test, rvfl_pred.flatten()))
            improvement = (elm_score - rvfl_score) / elm_score * 100
            print(f"{neurons:<10} {elm_score:>12.4f} {rvfl_score:>12.4f} {improvement:>+11.2f}%")

        results[neurons] = {'elm': elm_score, 'rvfl': rvfl_score}

    return results


def cross_val_comparison(X, y, neurons=100, n_splits=5, n_repeats=10):
    """Compare ELM and RVFL using cross-validation."""

    X = X.astype(np.float32)
    X = preprocessing.normalize(X)

    if not np.issubdtype(y.dtype, np.floating):
        le = LabelEncoder()
        y = le.fit_transform(y)

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # ELM
    elm_layer = ELMLayer(number_neurons=neurons, activation='mish', C=1.0)
    elm_model = ELMModel(elm_layer, classification=True)
    elm_scores = cross_val_score(elm_model, X, y, cv=cv, scoring='accuracy')

    # RVFL
    rvfl_layer = RVFLLayer(number_neurons=neurons, activation='mish', C=1.0)
    rvfl_model = RVFLModel(rvfl_layer, classification=True)
    rvfl_scores = cross_val_score(rvfl_model, X, y, cv=cv, scoring='accuracy')

    return elm_scores, rvfl_scores


# Main comparison
print("=" * 60)
print("ELM vs RVFL Comparison")
print("=" * 60)

# Classification datasets
print("\n" + "#" * 60)
print("CLASSIFICATION TASKS")
print("#" * 60)

# Breast Cancer dataset
bc = load_breast_cancer()
compare_on_dataset(bc.data, bc.target, "Breast Cancer", task='classification')

# Iris dataset
iris = load_iris()
compare_on_dataset(iris.data, iris.target, "Iris", task='classification')

# Custom dataset
path = "../Data/ionosphere.txt"
try:
    df = pd.read_csv(path, delimiter='\t').fillna(0)
    X_iono = df.values[:, 1:].astype(np.float32)
    y_iono = df.values[:, 0]
    le = LabelEncoder()
    y_iono = le.fit_transform(y_iono)
    compare_on_dataset(X_iono, y_iono, "Ionosphere", task='classification')
except:
    print("Ionosphere dataset not found, skipping...")

# Regression
print("\n" + "#" * 60)
print("REGRESSION TASKS")
print("#" * 60)

diabetes = load_diabetes()
compare_on_dataset(diabetes.data, diabetes.target, "Diabetes", task='regression')

# Cross-validation comparison
print("\n" + "#" * 60)
print("CROSS-VALIDATION COMPARISON (Breast Cancer)")
print("#" * 60)

elm_cv, rvfl_cv = cross_val_comparison(bc.data, bc.target, neurons=100)

print(f"\nELM:  Mean = {np.mean(elm_cv):.4f} (+/- {np.std(elm_cv):.4f})")
print(f"RVFL: Mean = {np.mean(rvfl_cv):.4f} (+/- {np.std(rvfl_cv):.4f})")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(rvfl_cv, elm_cv)
print(f"\nPaired t-test: t={t_stat:.4f}, p={p_value:.4f}")
if p_value < 0.05:
    if np.mean(rvfl_cv) > np.mean(elm_cv):
        print("RVFL is significantly better than ELM (p < 0.05)")
    else:
        print("ELM is significantly better than RVFL (p < 0.05)")
else:
    print("No significant difference between ELM and RVFL")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
RVFL typically performs better than ELM because:
1. Direct input links preserve original feature information
2. The model can learn both linear and nonlinear relationships
3. More parameters without significant computational overhead

When to use RVFL over ELM:
- When input features contain useful linear information
- For tabular data with meaningful feature scales
- When slightly better accuracy justifies the extra parameters
""")
