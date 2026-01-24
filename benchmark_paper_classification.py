"""
Classification Benchmark
Evaluates ELM and RVFL models on standard datasets
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import models
from Models.ELMModel import ELMModel
from Models.RVFLModel import RVFLModel
from Models.KELMModel import KELMModel
from Models.KRVFLModel import KRVFLModel
from Models.SSELMModel import SSELMModel
from Models.SSRVFLModel import SSRVFLModel
from Models.ML_ELMModel import ML_ELMModel
from Models.ML_RVFLModel import ML_RVFLModel
from Models.RCELMModel import RCELMModel
from Models.RCRVFLModel import RCRVFLModel

from Layers.ELMLayer import ELMLayer
from Layers.RVFLLayer import RVFLLayer
from Layers.KELMLayer import KELMLayer
from Layers.KRVFLLayer import KRVFLLayer
from Resources.Kernel import Kernel

# Function to load Ionosphere from UCI
def load_ionosphere():
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml('ionosphere', version=1, as_frame=False)
        X = data.data.astype(float)
        y = (data.target == 'g').astype(int)
        return X, y, "Ionosphere", 351, 34, 2
    except:
        # Fallback: use breast cancer
        X, y = load_breast_cancer(return_X_y=True)
        return X, y, "Breast Cancer", 569, 30, 2

# Classification datasets
def get_classification_datasets():
    datasets = []

    # 1. Iris
    X, y = load_iris(return_X_y=True)
    datasets.append(("Iris", X, y, 150, 4, 3))

    # 2. Wine
    X, y = load_wine(return_X_y=True)
    datasets.append(("Wine", X, y, 178, 13, 3))

    # 3. Breast Cancer
    X, y = load_breast_cancer(return_X_y=True)
    datasets.append(("Breast Cancer", X, y, 569, 30, 2))

    # 4. Ionosphere
    X, y, name, n, f, c = load_ionosphere()
    datasets.append((name, X, y, n, f, c))

    return datasets

# Models to evaluate
def get_classification_models():
    from Models.OSELMModel import OSELMModel
    from Models.OSRVFLModel import OSRVFLModel
    from Models.DeepELMModel import DeepELMModel
    from Models.DeepRVFLModel import DeepRVFLModel
    from Models.EnsembleDeepRVFLModel import EnsembleDeepRVFLModel
    from Layers.OSELMLayer import OSELMLayer
    from Layers.OSRVFLLayer import OSRVFLLayer
    from Layers.DeepRVFLLayer import DeepRVFLLayer
    from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer

    models = {}

    # Basic - 100 neurons, C=0.001
    elm_layer = ELMLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['ELM'] = ELMModel(elm_layer, task='classification')

    rvfl_layer = RVFLLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['RVFL'] = RVFLModel(rvfl_layer, task='classification')

    # Kernel
    kernel1 = Kernel(kernel_name='rbf', param=1.0)
    kelm_layer = KELMLayer(kernel1, C=0.001)
    models['K-ELM'] = KELMModel(kelm_layer, task='classification')

    kernel2 = Kernel(kernel_name='rbf', param=1.0)
    krvfl_layer = KRVFLLayer(kernel2, C=0.001)
    models['K-RVFL'] = KRVFLModel(krvfl_layer, task='classification')

    # Online
    oselm_layer = OSELMLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['OS-ELM'] = OSELMModel(oselm_layer, prefetch_size=50, batch_size=16, classification=True)

    osrvfl_layer = OSRVFLLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['OS-RVFL'] = OSRVFLModel(osrvfl_layer, prefetch_size=50, batch_size=16, classification=True)

    # Multi-layer
    mlelm = ML_ELMModel()
    mlelm.add(ELMLayer(number_neurons=50, activation='sigmoid'))
    mlelm.add(ELMLayer(number_neurons=100, activation='sigmoid', C=0.001))
    models['ML-ELM'] = mlelm

    mlrvfl = ML_RVFLModel()
    mlrvfl.add(RVFLLayer(number_neurons=50, activation='sigmoid'))
    mlrvfl.add(RVFLLayer(number_neurons=100, activation='sigmoid', C=0.001))
    models['ML-RVFL'] = mlrvfl

    # Deep (DeepELM uses add(), DeepRVFL uses layer)
    deep_elm = DeepELMModel()
    deep_elm.add(ELMLayer(number_neurons=100, activation='sigmoid'))
    deep_elm.add(ELMLayer(number_neurons=100, activation='sigmoid', C=0.001))
    models['Deep-ELM'] = deep_elm

    deep_rvfl_layer = DeepRVFLLayer(number_neurons=100, n_layers=2, activation='sigmoid', C=0.001)  # 2 layers
    models['Deep-RVFL'] = DeepRVFLModel(deep_rvfl_layer, task='classification')

    # Ensemble
    edrvfl_layer = EnsembleDeepRVFLLayer(number_neurons=100, n_layers=3, activation='sigmoid', C=0.001)
    models['edRVFL'] = EnsembleDeepRVFLModel(edrvfl_layer, classification=True)

    return models

def evaluate_model(model, X, y, cv):
    """Evaluate model with multiple metrics"""
    try:
        # Accuracy
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score=np.nan)
        acc_mean = np.nanmean(acc_scores)
        acc_std = np.nanstd(acc_scores)

        # F1-score (weighted for multiclass)
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', error_score=np.nan)
        f1_mean = np.nanmean(f1_scores)
        f1_std = np.nanstd(f1_scores)

        # Precision (weighted)
        prec_scores = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted', error_score=np.nan)
        prec_mean = np.nanmean(prec_scores)
        prec_std = np.nanstd(prec_scores)

        return {
            'accuracy': (acc_mean, acc_std),
            'f1_score': (f1_mean, f1_std),
            'precision': (prec_mean, prec_std)
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    print("=" * 80)
    print("CLASSIFICATION BENCHMARK - TfELM-RVFL")
    print("=" * 80)

    datasets = get_classification_datasets()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    all_results = {}

    for dataset_name, X, y, n_samples, n_features, n_classes in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
        print("=" * 80)

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = get_classification_models()
        dataset_results = {}

        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...", end=" ")
            result = evaluate_model(model, X_scaled, y, cv)
            if result:
                dataset_results[model_name] = result
                print(f"Acc={result['accuracy'][0]:.4f}, F1={result['f1_score'][0]:.4f}")
            else:
                print("FAILED")

        all_results[dataset_name] = dataset_results

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE - CLASSIFICATION")
    print("=" * 80)

    print("\n% Accuracy Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Classification accuracy across benchmark datasets.}")
    print("\\label{tab:classification_accuracy}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    print("\\toprule")
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{d[0]}}}" for d in datasets]) + " \\\\"
    print(header)
    print("\\midrule")

    model_names = list(get_classification_models().keys())
    for model_name in model_names:
        row = f"{model_name}"
        for dataset_name, _, _, _, _, _ in datasets:
            if dataset_name in all_results and model_name in all_results[dataset_name]:
                acc, std = all_results[dataset_name][model_name]['accuracy']
                row += f" & {acc:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # F1-Score Table
    print("\n% F1-Score Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{F1-score (weighted) across benchmark datasets.}")
    print("\\label{tab:classification_f1}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    print("\\toprule")
    print(header)
    print("\\midrule")

    for model_name in model_names:
        row = f"{model_name}"
        for dataset_name, _, _, _, _, _ in datasets:
            if dataset_name in all_results and model_name in all_results[dataset_name]:
                f1, std = all_results[dataset_name][model_name]['f1_score']
                row += f" & {f1:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
