"""
Unsupervised Learning Benchmark (Clustering)
Evaluates US-ELM and US-RVFL models on standard datasets
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans

# Import unsupervised models
from Models.USELMModel import USELMModel
from Models.USRVFLModel import USRVFLModel
from Models.USKELMModel import USKELMModel
from Models.USKRVFLModel import USKRVFLModel

from Layers.USELMLayer import USELMLayer
from Layers.USRVFLLayer import USRVFLLayer
from Layers.USKELMLayer import USKELMLayer
from Layers.USKRVFLLayer import USKRVFLLayer

# Datasets for clustering
def get_clustering_datasets():
    datasets = []

    # 1. Iris (3 natural clusters)
    X, y = load_iris(return_X_y=True)
    datasets.append(("Iris", X, y, 150, 4, 3))

    # 2. Wine (3 clusters)
    X, y = load_wine(return_X_y=True)
    datasets.append(("Wine", X, y, 178, 13, 3))

    # 3. Digits (subset, 10 clusters)
    X, y = load_digits(return_X_y=True)
    # Use subset for speed
    np.random.seed(42)
    idx = np.random.choice(len(X), size=500, replace=False)
    X_sub, y_sub = X[idx], y[idx]
    datasets.append(("Digits", X_sub, y_sub, 500, 64, 10))

    return datasets

def get_unsupervised_models(n_clusters, n_features):
    """Get unsupervised models for clustering"""
    # Consistent parameters: 100 neurons, sigmoid, lam=0.001
    models = {}

    # US-ELM
    try:
        layer = USELMLayer(number_neurons=100, embedding_size=min(50, n_features), lam=0.001, activation='sigmoid')
        models['US-ELM'] = ('uselm', USELMModel(layer, task='clustering', n_clusters=n_clusters))
    except Exception as e:
        print(f"  US-ELM init error: {e}")

    # US-RVFL
    try:
        layer = USRVFLLayer(number_neurons=100, embedding_size=min(50, n_features), lam=0.001, activation='sigmoid')
        models['US-RVFL'] = ('usrvfl', USRVFLModel(layer, task='clustering', n_clusters=n_clusters))
    except Exception as e:
        print(f"  US-RVFL init error: {e}")

    # USK-ELM (Kernel) - lam=0.001
    try:
        layer = USKELMLayer(embedding_size=min(50, n_features), lam=0.001, kernel='rbf')
        models['USK-ELM'] = ('uskelm', USKELMModel(layer, task='clustering', n_clusters=n_clusters))
    except Exception as e:
        print(f"  USK-ELM init error: {e}")

    # USK-RVFL (Kernel) - lam=0.001
    try:
        layer = USKRVFLLayer(embedding_size=min(50, n_features), lam=0.001, kernel='rbf')
        models['USK-RVFL'] = ('uskrvfl', USKRVFLModel(layer, task='clustering', n_clusters=n_clusters))
    except Exception as e:
        print(f"  USK-RVFL init error: {e}")

    return models

def evaluate_clustering(X, labels_pred, labels_true):
    """Evaluate clustering with multiple metrics"""
    results = {}

    try:
        # Internal metrics (do not require true labels)
        if len(np.unique(labels_pred)) > 1:
            results['silhouette'] = silhouette_score(X, labels_pred)
            results['davies_bouldin'] = davies_bouldin_score(X, labels_pred)
            results['calinski_harabasz'] = calinski_harabasz_score(X, labels_pred)
        else:
            results['silhouette'] = -1
            results['davies_bouldin'] = -1
            results['calinski_harabasz'] = -1

        # External metrics (require true labels)
        results['ari'] = adjusted_rand_score(labels_true, labels_pred)
        results['nmi'] = normalized_mutual_info_score(labels_true, labels_pred)

    except Exception as e:
        print(f"    Evaluation error: {e}")
        results = {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'ari': np.nan,
            'nmi': np.nan
        }

    return results

def main():
    print("=" * 80)
    print("UNSUPERVISED BENCHMARK (CLUSTERING) - TfELM-RVFL")
    print("=" * 80)

    datasets = get_clustering_datasets()
    all_results = {}
    n_runs = 5  # Number of runs to calculate std

    for dataset_name, X, y_true, n_samples, n_features, n_clusters in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples: {n_samples}, Features: {n_features}, True clusters: {n_clusters}")
        print("=" * 80)

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset_results = {}
        model_names_list = ['US-ELM', 'US-RVFL', 'USK-ELM', 'USK-RVFL']

        for model_name in model_names_list:
            print(f"  Evaluating {model_name}...", end=" ")
            run_results = {'silhouette': [], 'ari': [], 'nmi': []}

            for run in range(n_runs):
                try:
                    # Create fresh model for each run - 500 neurons, sigmoid, lam=0.1
                    if model_name == 'US-ELM':
                        layer = USELMLayer(number_neurons=500, embedding_size=min(50, n_features), lam=0.1, activation='sigmoid')
                        model = USELMModel(layer, task='clustering', n_clusters=n_clusters)
                    elif model_name == 'US-RVFL':
                        layer = USRVFLLayer(number_neurons=500, embedding_size=min(50, n_features), lam=0.1, activation='sigmoid')
                        model = USRVFLModel(layer, task='clustering', n_clusters=n_clusters)
                    elif model_name == 'USK-ELM':
                        layer = USKELMLayer(embedding_size=min(50, n_features), lam=0.1, kernel='rbf')
                        model = USKELMModel(layer, task='clustering', n_clusters=n_clusters)
                    elif model_name == 'USK-RVFL':
                        layer = USKRVFLLayer(embedding_size=min(50, n_features), lam=0.1, kernel='rbf')
                        model = USKRVFLModel(layer, task='clustering', n_clusters=n_clusters)

                    model.fit(X_scaled)
                    labels_pred = model.predict(X_scaled)

                    if hasattr(labels_pred, 'numpy'):
                        labels_pred = labels_pred.numpy()
                    labels_pred = np.array(labels_pred).flatten()

                    results = evaluate_clustering(X_scaled, labels_pred, y_true)
                    run_results['silhouette'].append(results['silhouette'])
                    run_results['ari'].append(results['ari'])
                    run_results['nmi'].append(results['nmi'])
                except Exception as e:
                    pass

            if len(run_results['silhouette']) > 0:
                dataset_results[model_name] = {
                    'silhouette': (np.mean(run_results['silhouette']), np.std(run_results['silhouette'])),
                    'ari': (np.mean(run_results['ari']), np.std(run_results['ari'])),
                    'nmi': (np.mean(run_results['nmi']), np.std(run_results['nmi']))
                }
                print(f"Sil={dataset_results[model_name]['silhouette'][0]:.2f}±{dataset_results[model_name]['silhouette'][1]:.2f}, "
                      f"ARI={dataset_results[model_name]['ari'][0]:.2f}±{dataset_results[model_name]['ari'][1]:.2f}, "
                      f"NMI={dataset_results[model_name]['nmi'][0]:.2f}±{dataset_results[model_name]['nmi'][1]:.2f}")
            else:
                print("FAILED")
                dataset_results[model_name] = {
                    'silhouette': (np.nan, np.nan),
                    'ari': (np.nan, np.nan),
                    'nmi': (np.nan, np.nan)
                }

        all_results[dataset_name] = dataset_results

    # Print LaTeX tables
    print("\n" + "=" * 80)
    print("LATEX TABLES - CLUSTERING")
    print("=" * 80)

    # Combined clustering table
    print("\n% Combined Clustering Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Clustering performance (mean $\\pm$ std). Sil=Silhouette, ARI=Adjusted Rand Index, NMI=Normalized Mutual Information.}")
    print("\\label{tab:clustering}")
    print("\\small")
    print("\\begin{tabular}{l|ccc|ccc|ccc}")
    print("\\toprule")
    print(" & \\multicolumn{3}{c|}{\\textbf{Iris}} & \\multicolumn{3}{c|}{\\textbf{Wine}} & \\multicolumn{3}{c}{\\textbf{Digits}} \\\\")
    print("\\textbf{Model} & Sil & ARI & NMI & Sil & ARI & NMI & Sil & ARI & NMI \\\\")
    print("\\midrule")

    model_names = ['US-ELM', 'US-RVFL', 'USK-ELM', 'USK-RVFL']
    for model_name in model_names:
        row = f"{model_name}"
        for dataset_name, _, _, _, _, _ in datasets:
            if dataset_name in all_results and model_name in all_results[dataset_name]:
                sil_mean, sil_std = all_results[dataset_name][model_name]['silhouette']
                ari_mean, ari_std = all_results[dataset_name][model_name]['ari']
                nmi_mean, nmi_std = all_results[dataset_name][model_name]['nmi']
                if np.isnan(sil_mean):
                    row += " & -- & -- & --"
                else:
                    row += f" & {sil_mean:.2f}$\\pm${sil_std:.2f} & {ari_mean:.2f}$\\pm${ari_std:.2f} & {nmi_mean:.2f}$\\pm${nmi_std:.2f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
