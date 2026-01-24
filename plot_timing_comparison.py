"""
Timing Comparison: ELM/RVFL vs Traditional ML Classifiers
Generates bar plot comparing training times
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed, skipping...")

from Models.ELMModel import ELMModel
from Models.RVFLModel import RVFLModel
from Layers.ELMLayer import ELMLayer
from Layers.RVFLLayer import RVFLLayer

def load_ionosphere():
    """Load Ionosphere dataset from OpenML"""
    try:
        data = fetch_openml('ionosphere', version=1, as_frame=False)
        X = data.data.astype(float)
        y = (data.target == 'g').astype(int)
        return X, y
    except:
        X, y = load_breast_cancer(return_X_y=True)
        return X, y

def get_datasets():
    """Load all classification datasets"""
    datasets = {}

    X, y = load_iris(return_X_y=True)
    datasets['Iris'] = (X, y)

    X, y = load_wine(return_X_y=True)
    datasets['Wine'] = (X, y)

    X, y = load_breast_cancer(return_X_y=True)
    datasets['Breast Cancer'] = (X, y)

    X, y = load_ionosphere()
    datasets['Ionosphere'] = (X, y)

    return datasets

def get_models():
    """Create all models to compare"""
    models = {}

    # ELM
    elm_layer = ELMLayer(number_neurons=100, activation='relu', C=10)
    models['ELM'] = ELMModel(elm_layer, task='classification')

    # RVFL
    rvfl_layer = RVFLLayer(number_neurons=100, activation='relu', C=10)
    models['RVFL'] = RVFLModel(rvfl_layer, task='classification')

    # MLP
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

    # Random Forest
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)

    # Gradient Boosting
    models['Gradient Boost'] = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)

    return models

def measure_training_time(model, X_train, y_train, n_runs=10):
    """Measure average training time over multiple runs"""
    times = []
    for _ in range(n_runs):
        # Recreate model for fresh state
        if hasattr(model, 'layer'):
            # ELM/RVFL models
            if isinstance(model, ELMModel):
                layer = ELMLayer(number_neurons=100, activation='relu', C=10)
                fresh_model = ELMModel(layer, task='classification')
            else:
                layer = RVFLLayer(number_neurons=100, activation='relu', C=10)
                fresh_model = RVFLModel(layer, task='classification')
        else:
            # sklearn models - clone
            from sklearn.base import clone
            fresh_model = clone(model)

        start = time.perf_counter()
        fresh_model.fit(X_train, y_train)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)

def main():
    print("=" * 70)
    print("TIMING COMPARISON: ELM/RVFL vs Traditional Classifiers")
    print("=" * 70)

    datasets = get_datasets()
    base_models = get_models()

    # Store results
    results = {name: {} for name in base_models.keys()}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{dataset_name} (n={len(X)}, features={X.shape[1]})")
        print("-" * 50)

        # Preprocess
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        for model_name, model in base_models.items():
            mean_time, std_time = measure_training_time(model, X_train, y_train, n_runs=10)
            results[model_name][dataset_name] = (mean_time, std_time)  # Keep in seconds
            print(f"  {model_name:15s}: {mean_time:8.4f} s (+/- {std_time:.4f})")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    dataset_names = list(datasets.keys())
    model_names = list(base_models.keys())
    n_datasets = len(dataset_names)
    n_models = len(model_names)

    # Colors - ELM/RVFL in blue tones, others in different colors
    colors = ['#1f77b4', '#2ca02c',  # ELM, RVFL (blue, green)
              '#ff7f0e', '#d62728', '#9467bd', '#8c564b']  # MLP, RF, GB, XGB

    x = np.arange(n_datasets)
    width = 0.11

    for i, model_name in enumerate(model_names):
        times = [results[model_name][ds][0] for ds in dataset_names]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=model_name, color=colors[i], alpha=0.85)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Training Time (s)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')  # Log scale for better visualization
    ax.set_ylabel('Training Time (s, log scale)', fontsize=12)

    plt.tight_layout()
    plt.savefig('paper/figures/timing_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/timing_comparison.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("Figures saved to paper/figures/timing_comparison.pdf and .png")

    # Print LaTeX table
    print("\n% LaTeX Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Training time comparison (seconds) across classification datasets.}")
    print("\\label{tab:timing}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * n_datasets + "}")
    print("\\toprule")
    print("\\textbf{Model} & " + " & ".join([f"\\textbf{{{ds}}}" for ds in dataset_names]) + " \\\\")
    print("\\midrule")

    for model_name in model_names:
        row = model_name
        for ds in dataset_names:
            mean_t, std_t = results[model_name][ds]
            row += f" & {mean_t:.3f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    plt.show()

if __name__ == "__main__":
    main()
