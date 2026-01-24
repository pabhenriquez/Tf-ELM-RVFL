"""
Benchmark Consolidado for Paper SoftwareX
- 4 Classification Datasets (Acc, F1, AUC)
- 3 Regression Datasets (R², RMSE, MAE)
- 3 Datasets Clustering (Silhouette, ARI, NMI)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, fetch_california_housing, load_digits
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Import models
from Models.ELMModel import ELMModel
from Models.RVFLModel import RVFLModel
from Models.ML_ELMModel import ML_ELMModel
from Models.ML_RVFLModel import ML_RVFLModel
from Models.RCELMModel import RCELMModel
from Models.RCRVFLModel import RCRVFLModel
from Models.USELMModel import USELMModel
from Models.USRVFLModel import USRVFLModel

from Layers.ELMLayer import ELMLayer
from Layers.RVFLLayer import RVFLLayer
from Layers.USELMLayer import USELMLayer
from Layers.USRVFLLayer import USRVFLLayer

# ============================================================================
# CLASSIFICATION
# ============================================================================

def load_ionosphere():
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml('ionosphere', version=1, as_frame=False)
        X = data.data.astype(float)
        y = (data.target == 'g').astype(int)
        return X, y
    except:
        X, y = load_breast_cancer(return_X_y=True)
        return X, y

def get_classification_datasets():
    datasets = []

    X, y = load_iris(return_X_y=True)
    datasets.append(("Iris", X, y, 150, 4, 3))

    X, y = load_wine(return_X_y=True)
    datasets.append(("Wine", X, y, 178, 13, 3))

    X, y = load_breast_cancer(return_X_y=True)
    datasets.append(("Breast Cancer", X, y, 569, 30, 2))

    X, y = load_ionosphere()
    datasets.append(("Ionosphere", X, y, len(y), X.shape[1], 2))

    return datasets

def get_classification_models():
    models = {}

    models['ELM'] = ELMModel(ELMLayer(number_neurons=1000, activation='relu', C=10), task='classification')
    models['RVFL'] = RVFLModel(RVFLLayer(number_neurons=1000, activation='relu', C=10), task='classification')

    mlelm = ML_ELMModel()
    mlelm.add(ELMLayer(number_neurons=500, activation='relu'))
    mlelm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    models['ML-ELM'] = mlelm

    mlrvfl = ML_RVFLModel()
    mlrvfl.add(RVFLLayer(number_neurons=500, activation='relu'))
    mlrvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    models['ML-RVFL'] = mlrvfl

    rcelm = RCELMModel(task='classification')
    rcelm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    rcelm.add(ELMLayer(number_neurons=500, activation='relu', C=10))
    models['RC-ELM'] = rcelm

    rcrvfl = RCRVFLModel(task='classification')
    rcrvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    rcrvfl.add(RVFLLayer(number_neurons=500, activation='relu', C=10))
    models['RC-RVFL'] = rcrvfl

    return models

def run_classification_benchmark():
    print("=" * 100)
    print("BENCHMARK CLASSIFICATION")
    print("=" * 100)

    datasets = get_classification_datasets()
    all_results = {}

    for dataset_name, X, y, n_samples, n_features, n_classes in datasets:
        print(f"\nDataset: {dataset_name} ({n_samples} samples, {n_features} features, {n_classes} classes)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        models = get_classification_models()

        dataset_results = {}
        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ")
            try:
                acc = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', error_score=np.nan))
                f1 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted', error_score=np.nan))

                # AUC
                if n_classes == 2:
                    auc = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', error_score=np.nan))
                else:
                    auc = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc_ovr_weighted', error_score=np.nan))

                dataset_results[model_name] = {'acc': acc, 'f1': f1, 'auc': auc}
                print(f"Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                dataset_results[model_name] = {'acc': np.nan, 'f1': np.nan, 'auc': np.nan}

        all_results[dataset_name] = dataset_results

    return all_results, datasets

# ============================================================================
# REGRESSION
# ============================================================================

def load_ames_housing():
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml(name="house_prices", as_frame=True, parser='auto')
        df = data.frame
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
            y = df['SalePrice'].values
        else:
            y = df[numeric_cols[-1]].values
            numeric_cols = numeric_cols[:-1]
        X = df[numeric_cols[:10]].fillna(0).values
        return X, y, "Ames Housing"
    except:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
        return X, y, "Synthetic"

def get_regression_datasets():
    datasets = []

    X, y = load_diabetes(return_X_y=True)
    datasets.append(("Diabetes", X, y, 442, 10))

    X, y = fetch_california_housing(return_X_y=True)
    np.random.seed(42)
    idx = np.random.choice(len(X), size=min(1500, len(X)), replace=False)
    datasets.append(("California", X[idx], y[idx], len(idx), 8))

    X, y, name = load_ames_housing()
    datasets.append((name, X, y, len(y), X.shape[1]))

    return datasets

def get_regression_models():
    models = {}

    models['ELM'] = ELMModel(ELMLayer(number_neurons=1000, activation='relu', C=10), task='regression')
    models['RVFL'] = RVFLModel(RVFLLayer(number_neurons=1000, activation='relu', C=10), task='regression')

    mlelm = ML_ELMModel(classification=False)
    mlelm.add(ELMLayer(number_neurons=500, activation='relu'))
    mlelm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    models['ML-ELM'] = mlelm

    mlrvfl = ML_RVFLModel(classification=False)
    mlrvfl.add(RVFLLayer(number_neurons=500, activation='relu'))
    mlrvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    models['ML-RVFL'] = mlrvfl

    rcelm = RCELMModel(task='regression')
    rcelm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    rcelm.add(ELMLayer(number_neurons=500, activation='relu', C=10))
    models['RC-ELM'] = rcelm

    rcrvfl = RCRVFLModel(task='regression')
    rcrvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    rcrvfl.add(RVFLLayer(number_neurons=500, activation='relu', C=10))
    models['RC-RVFL'] = rcrvfl

    return models

def run_regression_benchmark():
    print("\n" + "=" * 100)
    print("BENCHMARK REGRESSION")
    print("=" * 100)

    datasets = get_regression_datasets()
    all_results = {}

    for dataset_name, X, y, n_samples, n_features in datasets:
        print(f"\nDataset: {dataset_name} ({n_samples} samples, {n_features} features)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        models = get_regression_models()

        dataset_results = {}
        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ")
            try:
                r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='r2', error_score=np.nan))
                mse = -np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error', error_score=np.nan))
                mae = -np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error', error_score=np.nan))
                rmse = np.sqrt(mse)

                dataset_results[model_name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
                print(f"R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
            except Exception as e:
                print(f"Error: {e}")
                dataset_results[model_name] = {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}

        all_results[dataset_name] = dataset_results

    return all_results, datasets

# ============================================================================
# CLUSTERING
# ============================================================================

def get_clustering_datasets():
    datasets = []

    X, y = load_iris(return_X_y=True)
    datasets.append(("Iris", X, y, 150, 4, 3))

    X, y = load_wine(return_X_y=True)
    datasets.append(("Wine", X, y, 178, 13, 3))

    X, y = load_digits(return_X_y=True)
    np.random.seed(42)
    idx = np.random.choice(len(X), size=500, replace=False)
    datasets.append(("Digits", X[idx], y[idx], 500, 64, 10))

    return datasets

def run_clustering_benchmark():
    print("\n" + "=" * 100)
    print("BENCHMARK CLUSTERING")
    print("=" * 100)

    datasets = get_clustering_datasets()
    all_results = {}

    for dataset_name, X, y_true, n_samples, n_features, n_clusters in datasets:
        print(f"\nDataset: {dataset_name} ({n_samples} samples, {n_features} features, {n_clusters} clusters)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset_results = {}

        # KMeans baseline
        print(f"  KMeans...", end=" ")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        dataset_results['KMeans'] = {'silhouette': sil, 'ari': ari, 'nmi': nmi}
        print(f"Sil={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")

        # US-ELM
        print(f"  US-ELM...", end=" ")
        try:
            layer = USELMLayer(number_neurons=500, embedding_size=min(50, n_features), lam=0.001, activation='relu')
            model = USELMModel(layer, task='clustering', n_clusters=n_clusters)
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            labels = np.array(labels).flatten()
            sil = silhouette_score(X_scaled, labels)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
            dataset_results['US-ELM'] = {'silhouette': sil, 'ari': ari, 'nmi': nmi}
            print(f"Sil={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            dataset_results['US-ELM'] = {'silhouette': np.nan, 'ari': np.nan, 'nmi': np.nan}

        # US-RVFL
        print(f"  US-RVFL...", end=" ")
        try:
            layer = USRVFLLayer(number_neurons=500, embedding_size=min(50, n_features), lam=0.001, activation='relu')
            model = USRVFLModel(layer, task='clustering', n_clusters=n_clusters)
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            labels = np.array(labels).flatten()
            sil = silhouette_score(X_scaled, labels)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
            dataset_results['US-RVFL'] = {'silhouette': sil, 'ari': ari, 'nmi': nmi}
            print(f"Sil={sil:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            dataset_results['US-RVFL'] = {'silhouette': np.nan, 'ari': np.nan, 'nmi': np.nan}

        all_results[dataset_name] = dataset_results

    return all_results, datasets

# ============================================================================
# GENERATE LATEX TABLES
# ============================================================================

def generate_latex_tables(clf_results, clf_datasets, reg_results, reg_datasets, clust_results, clust_datasets):
    print("\n" + "=" * 100)
    print("CONSOLIDATED LATEX TABLES")
    print("=" * 100)

    # CLASSIFICATION
    print("\n% CLASSIFICATION TABLE")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Classification performance across benchmark datasets. Best results per dataset in bold.}")
    print("\\label{tab:classification}")
    print("\\small")
    print("\\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    print("\\toprule")
    print(" & \\multicolumn{3}{c|}{\\textbf{Iris}} & \\multicolumn{3}{c|}{\\textbf{Wine}} & \\multicolumn{3}{c|}{\\textbf{Breast Cancer}} & \\multicolumn{3}{c}{\\textbf{Ionosphere}} \\\\")
    print("\\textbf{Model} & Acc & F1 & AUC & Acc & F1 & AUC & Acc & F1 & AUC & Acc & F1 & AUC \\\\")
    print("\\midrule")

    model_names = ['ELM', 'RVFL', 'ML-ELM', 'ML-RVFL', 'RC-ELM', 'RC-RVFL']
    dataset_names = [d[0] for d in clf_datasets]

    for model_name in model_names:
        row = f"{model_name}"
        for ds_name in dataset_names:
            if ds_name in clf_results and model_name in clf_results[ds_name]:
                r = clf_results[ds_name][model_name]
                acc = r['acc'] if not np.isnan(r['acc']) else 0
                f1 = r['f1'] if not np.isnan(r['f1']) else 0
                auc = r['auc'] if not np.isnan(r['auc']) else 0
                row += f" & {acc:.2f} & {f1:.2f} & {auc:.2f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # REGRESSION
    print("\n% REGRESSION TABLE")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Regression performance across benchmark datasets. Best results per dataset in bold.}")
    print("\\label{tab:regression}")
    print("\\small")
    print("\\begin{tabular}{l|ccc|ccc|ccc}")
    print("\\toprule")

    reg_ds_names = [d[0] for d in reg_datasets]
    header = " & " + " & ".join([f"\\multicolumn{{3}}{{c|}}{{{name}}}" if i < len(reg_ds_names)-1 else f"\\multicolumn{{3}}{{c}}{{{name}}}" for i, name in enumerate(reg_ds_names)])
    print(header + " \\\\")
    print("\\textbf{Model} & $R^2$ & RMSE & MAE & $R^2$ & RMSE & MAE & $R^2$ & RMSE & MAE \\\\")
    print("\\midrule")

    for model_name in model_names:
        row = f"{model_name}"
        for ds_name in reg_ds_names:
            if ds_name in reg_results and model_name in reg_results[ds_name]:
                r = reg_results[ds_name][model_name]
                r2 = r['r2'] if not np.isnan(r['r2']) else 0
                rmse = r['rmse'] if not np.isnan(r['rmse']) else 0
                mae = r['mae'] if not np.isnan(r['mae']) else 0
                # Normalize RMSE y MAE for mostrar
                if rmse > 1000:
                    row += f" & {r2:.2f} & {rmse/1000:.1f}k & {mae/1000:.1f}k"
                else:
                    row += f" & {r2:.2f} & {rmse:.1f} & {mae:.1f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # CLUSTERING
    print("\n% CLUSTERING TABLE")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Clustering performance across benchmark datasets. Best results per dataset in bold.}")
    print("\\label{tab:clustering}")
    print("\\small")
    print("\\begin{tabular}{l|ccc|ccc|ccc}")
    print("\\toprule")

    clust_ds_names = [d[0] for d in clust_datasets]
    header = " & " + " & ".join([f"\\multicolumn{{3}}{{c|}}{{{name}}}" if i < len(clust_ds_names)-1 else f"\\multicolumn{{3}}{{c}}{{{name}}}" for i, name in enumerate(clust_ds_names)])
    print(header + " \\\\")
    print("\\textbf{Model} & Sil & ARI & NMI & Sil & ARI & NMI & Sil & ARI & NMI \\\\")
    print("\\midrule")

    clust_model_names = ['KMeans', 'US-ELM', 'US-RVFL']
    for model_name in clust_model_names:
        row = f"{model_name}"
        for ds_name in clust_ds_names:
            if ds_name in clust_results and model_name in clust_results[ds_name]:
                r = clust_results[ds_name][model_name]
                sil = r['silhouette'] if not np.isnan(r['silhouette']) else 0
                ari = r['ari'] if not np.isnan(r['ari']) else 0
                nmi = r['nmi'] if not np.isnan(r['nmi']) else 0
                row += f" & {sil:.2f} & {ari:.2f} & {nmi:.2f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    clf_results, clf_datasets = run_classification_benchmark()
    reg_results, reg_datasets = run_regression_benchmark()
    clust_results, clust_datasets = run_clustering_benchmark()
    generate_latex_tables(clf_results, clf_datasets, reg_results, reg_datasets, clust_results, clust_datasets)
