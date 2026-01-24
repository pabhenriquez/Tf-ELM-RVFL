"""
Benchmark por Categorys for Paper SoftwareX
Reporta average ± std por category, separando ELM vs RVFL
Categorys according to literatura:
1. Basic: ELM, RVFL
2. Kernel: K-ELM, K-RVFL
3. Semi-supervised: SS-ELM, SS-RVFL
4. Unsupervised: US-ELM, US-RVFL
5. Online: OS-ELM, OS-RVFL
6. Multi-layer: ML-ELM, ML-RVFL
7. Deep Representation: Dr-ELM, Dr-RVFL, EHDr-ELM, EHDr-RVFL
8. Residual: RC-ELM, RC-RVFL
9. Ensemble: EnsembleDeepRVFL
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Import models
from Models.ELMModel import ELMModel
from Models.RVFLModel import RVFLModel
from Models.KELMModel import KELMModel
from Models.KRVFLModel import KRVFLModel
from Models.SSELMModel import SSELMModel
from Models.SSRVFLModel import SSRVFLModel
from Models.OSELMModel import OSELMModel
from Models.OSRVFLModel import OSRVFLModel
from Models.ML_ELMModel import ML_ELMModel
from Models.ML_RVFLModel import ML_RVFLModel
from Models.DrELMModel import DrELMModel
from Models.DrRVFLModel import DrRVFLModel
from Models.EHDrELMModel import EHDrELMModel
from Models.EHDrRVFLModel import EHDrRVFLModel
from Models.DeepELMModel import DeepELMModel
from Models.DeepRVFLModel import DeepRVFLModel
from Models.RCELMModel import RCELMModel
from Models.RCRVFLModel import RCRVFLModel
from Models.USELMModel import USELMModel
from Models.USRVFLModel import USRVFLModel
from Models.EnsembleDeepRVFLModel import EnsembleDeepRVFLModel

from Layers.ELMLayer import ELMLayer
from Layers.RVFLLayer import RVFLLayer
from Layers.KELMLayer import KELMLayer
from Layers.KRVFLLayer import KRVFLLayer
from Layers.OSELMLayer import OSELMLayer
from Layers.OSRVFLLayer import OSRVFLLayer
from Layers.USELMLayer import USELMLayer
from Layers.USRVFLLayer import USRVFLLayer
from Layers.GELM_AE_Layer import GELM_AE_Layer
from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer
from Layers.DeepRVFLLayer import DeepRVFLLayer
from Resources.Kernel import Kernel

N_RUNS = 5  # Number of repeticiones for calcular average y std

# ============================================================================
# DATASETS
# ============================================================================

def get_classification_datasets():
    datasets = []

    X, y = load_iris(return_X_y=True)
    datasets.append(("Iris", X, y))

    X, y = load_wine(return_X_y=True)
    datasets.append(("Wine", X, y))

    X, y = load_breast_cancer(return_X_y=True)
    datasets.append(("Breast Cancer", X, y))

    # Ionosphere
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml('ionosphere', version=1, as_frame=False)
        X = data.data.astype(float)
        y = (data.target == 'g').astype(int)
        datasets.append(("Ionosphere", X, y))
    except:
        pass

    return datasets

def get_regression_datasets():
    datasets = []

    X, y = load_diabetes(return_X_y=True)
    datasets.append(("Diabetes", X, y))

    X, y = fetch_california_housing(return_X_y=True)
    np.random.seed(42)
    idx = np.random.choice(len(X), size=1500, replace=False)
    datasets.append(("California", X[idx], y[idx]))

    # Ames Housing
    from sklearn.datasets import fetch_openml
    try:
        data = fetch_openml(name="house_prices", as_frame=True, parser='auto')
        df = data.frame
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
            y = df['SalePrice'].values
        X = df[numeric_cols[:10]].fillna(0).values
        datasets.append(("Ames Housing", X, y))
    except:
        pass

    return datasets

def get_clustering_datasets():
    datasets = []

    X, y = load_iris(return_X_y=True)
    datasets.append(("Iris", X, y, 3))

    X, y = load_wine(return_X_y=True)
    datasets.append(("Wine", X, y, 3))

    X, y = load_digits(return_X_y=True)
    np.random.seed(42)
    idx = np.random.choice(len(X), size=500, replace=False)
    datasets.append(("Digits", X[idx], y[idx], 10))

    return datasets

# ============================================================================
# MODELS BY CATEGORY (according to literature)
# ============================================================================

def get_basic_models(task='classification'):
    """Category 1: Basic - ELM, RVFL"""
    models = {'ELM': [], 'RVFL': []}
    models['ELM'].append(('ELM', ELMModel(ELMLayer(number_neurons=1000, activation='relu', C=10), task=task)))
    models['RVFL'].append(('RVFL', RVFLModel(RVFLLayer(number_neurons=1000, activation='relu', C=10), task=task)))
    return models

def get_kernel_models(task='classification'):
    """Category 2: Kernel - K-ELM, K-RVFL"""
    models = {'ELM': [], 'RVFL': []}
    rbf_kernel = Kernel(kernel_name='rbf', param=1.0)
    models['ELM'].append(('K-ELM', KELMModel(KELMLayer(kernel=rbf_kernel, C=10), task=task)))
    models['RVFL'].append(('K-RVFL', KRVFLModel(KRVFLLayer(kernel=rbf_kernel, C=10), task=task)))
    return models

def get_semisupervised_models():
    """Category 3: Semi-supervised - SS-ELM, SS-RVFL (classification only)"""
    models = {'ELM': [], 'RVFL': []}
    models['ELM'].append(('SS-ELM', SSELMModel(ELMLayer(number_neurons=1000, activation='relu', C=10))))
    models['RVFL'].append(('SS-RVFL', SSRVFLModel(RVFLLayer(number_neurons=1000, activation='relu', C=10))))
    return models

def get_online_models(task='classification'):
    """Category 4: Online/Sequential - OS-ELM, OS-RVFL"""
    models = {'ELM': [], 'RVFL': []}
    is_classification = (task == 'classification')
    models['ELM'].append(('OS-ELM', OSELMModel(OSELMLayer(number_neurons=1000, activation='relu', C=10),
                                               prefetch_size=50, batch_size=32, classification=is_classification)))
    models['RVFL'].append(('OS-RVFL', OSRVFLModel(OSRVFLLayer(number_neurons=1000, activation='relu', C=10),
                                                  prefetch_size=50, batch_size=32, classification=is_classification)))
    return models

def get_multilayer_models(task='classification'):
    """Category 5: Multi-layer - ML-ELM, ML-RVFL"""
    models = {'ELM': [], 'RVFL': []}

    ml_elm = ML_ELMModel(classification=(task=='classification'))
    ml_elm.add(ELMLayer(number_neurons=500, activation='relu'))
    ml_elm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    models['ELM'].append(('ML-ELM', ml_elm))

    ml_rvfl = ML_RVFLModel(classification=(task=='classification'))
    ml_rvfl.add(RVFLLayer(number_neurons=500, activation='relu'))
    ml_rvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    models['RVFL'].append(('ML-RVFL', ml_rvfl))

    return models

def get_deep_models(task='classification'):
    """Category 6: Deep - DeepELM, DeepRVFL"""
    models = {'ELM': [], 'RVFL': []}

    # DeepELM
    deep_elm = DeepELMModel(classification=(task=='classification'))
    deep_elm.add(ELMLayer(number_neurons=500, activation='relu'))
    deep_elm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    models['ELM'].append(('Deep-ELM', deep_elm))

    # DeepRVFL
    layer = DeepRVFLLayer(number_neurons=500, n_layers=3, activation='relu', C=10)
    models['RVFL'].append(('Deep-RVFL', DeepRVFLModel(layer, task=task)))

    return models

def get_deep_representation_models():
    """Category 6: Deep Representation - Dr-ELM, Dr-RVFL, EHDr-ELM, EHDr-RVFL (classification only)"""
    models = {'ELM': [], 'RVFL': []}

    # Dr-ELM
    dr_elm = DrELMModel()
    dr_elm.add(GELM_AE_Layer(number_neurons=500))
    dr_elm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    models['ELM'].append(('Dr-ELM', dr_elm))

    # Dr-RVFL
    dr_rvfl = DrRVFLModel()
    dr_rvfl.add(GELM_AE_Layer(number_neurons=500))
    dr_rvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    models['RVFL'].append(('Dr-RVFL', dr_rvfl))

    # EHDr-ELM
    ehdr_elm = EHDrELMModel()
    ehdr_elm.add(GELM_AE_Layer(number_neurons=500))
    ehdr_elm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    models['ELM'].append(('EHDr-ELM', ehdr_elm))

    # EHDr-RVFL
    ehdr_rvfl = EHDrRVFLModel()
    ehdr_rvfl.add(GELM_AE_Layer(number_neurons=500))
    ehdr_rvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    models['RVFL'].append(('EHDr-RVFL', ehdr_rvfl))

    return models

def get_residual_models(task='classification'):
    """Category 7: Residual - RC-ELM, RC-RVFL"""
    models = {'ELM': [], 'RVFL': []}

    # RC-ELM
    rc_elm = RCELMModel(task=task)
    rc_elm.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    rc_elm.add(ELMLayer(number_neurons=500, activation='relu', C=10))
    models['ELM'].append(('RC-ELM', rc_elm))

    # RC-RVFL
    rc_rvfl = RCRVFLModel(task=task)
    rc_rvfl.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    rc_rvfl.add(RVFLLayer(number_neurons=500, activation='relu', C=10))
    models['RVFL'].append(('RC-RVFL', rc_rvfl))

    return models

def get_ensemble_models():
    """Category 8: Ensemble - EnsembleDeepRVFL (classification only)"""
    models = {'ELM': [], 'RVFL': []}

    layer = EnsembleDeepRVFLLayer(number_neurons=500, n_layers=5, activation='relu', C=10)
    models['RVFL'].append(('Ensemble-RVFL', EnsembleDeepRVFLModel(layer, classification=True)))

    return models

def get_unsupervised_models(n_clusters, n_features):
    """Category 9: Unsupervised - US-ELM, US-RVFL"""
    models = {'ELM': [], 'RVFL': []}

    layer_elm = USELMLayer(number_neurons=500, embedding_size=min(50, n_features), lam=0.001, activation='relu')
    models['ELM'].append(('US-ELM', USELMModel(layer_elm, task='clustering', n_clusters=n_clusters)))

    layer_rvfl = USRVFLLayer(number_neurons=500, embedding_size=min(50, n_features), lam=0.001, activation='relu')
    models['RVFL'].append(('US-RVFL', USRVFLModel(layer_rvfl, task='clustering', n_clusters=n_clusters)))

    return models

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_classification(model, X_train, X_test, y_train, y_test):
    """Evaluate classification model"""
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {'acc': acc, 'f1': f1}
    except Exception as e:
        return None

def evaluate_regression(model, X_train, X_test, y_train, y_test):
    """Evaluate regression model"""
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    except Exception as e:
        return None

def evaluate_clustering(model, X, y_true):
    """Evaluar modelo de clustering"""
    try:
        model.fit(X)
        labels = model.predict(X)
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        labels = np.array(labels).flatten()

        sil = silhouette_score(X, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        return {'sil': sil, 'ari': ari, 'nmi': nmi}
    except Exception as e:
        return None

def get_fresh_models(cat_name, task):
    """Get fresh models for a category"""
    if cat_name == 'Basic':
        return get_basic_models(task)
    elif cat_name == 'Kernel':
        return get_kernel_models(task)
    elif cat_name == 'Semi-supervised':
        return get_semisupervised_models()
    elif cat_name == 'Online':
        return get_online_models(task)
    elif cat_name == 'Multi-layer':
        return get_multilayer_models(task)
    elif cat_name == 'Deep':
        return get_deep_models(task)
    elif cat_name == 'Deep Repr.':
        return get_deep_representation_models()
    elif cat_name == 'Residual':
        return get_residual_models(task)
    elif cat_name == 'Ensemble':
        return get_ensemble_models()
    return {'ELM': [], 'RVFL': []}

def run_classification_benchmark():
    """Classification benchmark"""
    datasets = get_classification_datasets()
    results = {}

    # Categories for classification
    categories = ['Basic', 'Kernel', 'Semi-supervised', 'Online', 'Multi-layer', 'Deep', 'Deep Repr.', 'Residual', 'Ensemble']

    for ds_name, X, y in datasets:
        print(f"\n  Dataset: {ds_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        category_results = {}

        for cat_name in categories:
            elm_scores = []
            rvfl_scores = []

            for run in range(N_RUNS):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=run*42
                )

                fresh_models = get_fresh_models(cat_name, 'classification')

                # Evaluate ELM variants
                for name, model in fresh_models['ELM']:
                    result = evaluate_classification(model, X_train, X_test, y_train, y_test)
                    if result:
                        elm_scores.append(result)

                # Evaluate RVFL variants
                for name, model in fresh_models['RVFL']:
                    result = evaluate_classification(model, X_train, X_test, y_train, y_test)
                    if result:
                        rvfl_scores.append(result)

            # Calculate averages
            if elm_scores:
                category_results[f'{cat_name}_ELM'] = {
                    'acc': (np.mean([s['acc'] for s in elm_scores]), np.std([s['acc'] for s in elm_scores])),
                    'f1': (np.mean([s['f1'] for s in elm_scores]), np.std([s['f1'] for s in elm_scores]))
                }
                print(f"    {cat_name} ELM: Acc={category_results[f'{cat_name}_ELM']['acc'][0]:.3f}")

            if rvfl_scores:
                category_results[f'{cat_name}_RVFL'] = {
                    'acc': (np.mean([s['acc'] for s in rvfl_scores]), np.std([s['acc'] for s in rvfl_scores])),
                    'f1': (np.mean([s['f1'] for s in rvfl_scores]), np.std([s['f1'] for s in rvfl_scores]))
                }
                print(f"    {cat_name} RVFL: Acc={category_results[f'{cat_name}_RVFL']['acc'][0]:.3f}")

        results[ds_name] = category_results

    return results, datasets

def run_regression_benchmark():
    """Regression benchmark"""
    datasets = get_regression_datasets()
    results = {}

    # Categories for regression (without Semi-supervised, Deep Repr., Ensemble which are classification only)
    categories = ['Basic', 'Kernel', 'Online', 'Multi-layer', 'Deep', 'Residual']

    for ds_name, X, y in datasets:
        print(f"\n  Dataset: {ds_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        category_results = {}

        for cat_name in categories:
            elm_scores = []
            rvfl_scores = []

            for run in range(N_RUNS):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=run*42
                )

                fresh_models = get_fresh_models(cat_name, 'regression')

                # Evaluate ELM variants
                for name, model in fresh_models['ELM']:
                    result = evaluate_regression(model, X_train, X_test, y_train, y_test)
                    if result:
                        elm_scores.append(result)

                # Evaluate RVFL variants
                for name, model in fresh_models['RVFL']:
                    result = evaluate_regression(model, X_train, X_test, y_train, y_test)
                    if result:
                        rvfl_scores.append(result)

            # Calculate averages
            if elm_scores:
                category_results[f'{cat_name}_ELM'] = {
                    'r2': (np.mean([s['r2'] for s in elm_scores]), np.std([s['r2'] for s in elm_scores])),
                    'rmse': (np.mean([s['rmse'] for s in elm_scores]), np.std([s['rmse'] for s in elm_scores])),
                    'mae': (np.mean([s['mae'] for s in elm_scores]), np.std([s['mae'] for s in elm_scores]))
                }
                print(f"    {cat_name} ELM: R²={category_results[f'{cat_name}_ELM']['r2'][0]:.3f}")

            if rvfl_scores:
                category_results[f'{cat_name}_RVFL'] = {
                    'r2': (np.mean([s['r2'] for s in rvfl_scores]), np.std([s['r2'] for s in rvfl_scores])),
                    'rmse': (np.mean([s['rmse'] for s in rvfl_scores]), np.std([s['rmse'] for s in rvfl_scores])),
                    'mae': (np.mean([s['mae'] for s in rvfl_scores]), np.std([s['mae'] for s in rvfl_scores]))
                }
                print(f"    {cat_name} RVFL: R²={category_results[f'{cat_name}_RVFL']['r2'][0]:.3f}")

        results[ds_name] = category_results

    return results, datasets

def run_clustering_benchmark():
    """Clustering benchmark (US-ELM, US-RVFL models only)"""
    datasets = get_clustering_datasets()
    results = {}

    for ds_name, X, y_true, n_clusters in datasets:
        print(f"\n  Dataset: {ds_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        category_results = {}

        # US-ELM
        elm_scores = []
        for run in range(N_RUNS):
            try:
                layer = USELMLayer(number_neurons=500, embedding_size=min(50, X.shape[1]), lam=0.001, activation='relu')
                model = USELMModel(layer, task='clustering', n_clusters=n_clusters)
                result = evaluate_clustering(model, X_scaled, y_true)
                if result:
                    elm_scores.append(result)
            except:
                pass
        if elm_scores:
            category_results['Unsupervised_ELM'] = {
                'sil': (np.mean([s['sil'] for s in elm_scores]), np.std([s['sil'] for s in elm_scores])),
                'ari': (np.mean([s['ari'] for s in elm_scores]), np.std([s['ari'] for s in elm_scores])),
                'nmi': (np.mean([s['nmi'] for s in elm_scores]), np.std([s['nmi'] for s in elm_scores]))
            }
            print(f"    US-ELM: ARI={category_results['Unsupervised_ELM']['ari'][0]:.3f}")

        # US-RVFL
        rvfl_scores = []
        for run in range(N_RUNS):
            try:
                layer = USRVFLLayer(number_neurons=500, embedding_size=min(50, X.shape[1]), lam=0.001, activation='relu')
                model = USRVFLModel(layer, task='clustering', n_clusters=n_clusters)
                result = evaluate_clustering(model, X_scaled, y_true)
                if result:
                    rvfl_scores.append(result)
            except:
                pass
        if rvfl_scores:
            category_results['Unsupervised_RVFL'] = {
                'sil': (np.mean([s['sil'] for s in rvfl_scores]), np.std([s['sil'] for s in rvfl_scores])),
                'ari': (np.mean([s['ari'] for s in rvfl_scores]), np.std([s['ari'] for s in rvfl_scores])),
                'nmi': (np.mean([s['nmi'] for s in rvfl_scores]), np.std([s['nmi'] for s in rvfl_scores]))
            }
            print(f"    US-RVFL: ARI={category_results['Unsupervised_RVFL']['ari'][0]:.3f}")

        results[ds_name] = category_results

    return results, datasets

# ============================================================================
# GENERATE LATEX TABLES
# ============================================================================

def print_classification_table(results, datasets):
    print("\n% CLASSIFICATION TABLE BY CATEGORY")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Classification performance by model category (mean $\\pm$ std).}")
    print("\\label{tab:classification}")
    print("\\scriptsize")

    ds_names = [d[0] for d in datasets]
    categories = ['Basic', 'Kernel', 'Semi-supervised', 'Online', 'Multi-layer', 'Deep', 'Deep Repr.', 'Residual', 'Ensemble']

    print("\\begin{tabular}{ll" + "|cc" * len(ds_names) + "}")
    print("\\toprule")

    # Header
    header = "\\textbf{Category} & \\textbf{Type}"
    for ds in ds_names:
        header += f" & \\multicolumn{{2}}{{c|}}{{{ds}}}"
    header = header.rstrip("|") + " \\\\"
    print(header)

    subheader = " & "
    for _ in ds_names:
        subheader += " & Acc & F1"
    subheader += " \\\\"
    print(subheader)
    print("\\midrule")

    for cat in categories:
        # ELM row
        row = f"{cat} & ELM"
        for ds in ds_names:
            key = f'{cat}_ELM'
            if ds in results and key in results[ds]:
                acc_m, acc_s = results[ds][key]['acc']
                f1_m, f1_s = results[ds][key]['f1']
                row += f" & {acc_m:.2f}$\\pm${acc_s:.2f} & {f1_m:.2f}$\\pm${f1_s:.2f}"
            else:
                row += " & -- & --"
        row += " \\\\"
        print(row)

        # RVFL row
        row = f" & RVFL"
        for ds in ds_names:
            key = f'{cat}_RVFL'
            if ds in results and key in results[ds]:
                acc_m, acc_s = results[ds][key]['acc']
                f1_m, f1_s = results[ds][key]['f1']
                row += f" & {acc_m:.2f}$\\pm${acc_s:.2f} & {f1_m:.2f}$\\pm${f1_s:.2f}"
            else:
                row += " & -- & --"
        row += " \\\\"
        print(row)
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def print_regression_table(results, datasets):
    print("\n% REGRESSION TABLE BY CATEGORY")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Regression performance by model category (mean $\\pm$ std).}")
    print("\\label{tab:regression}")
    print("\\scriptsize")

    ds_names = [d[0] for d in datasets]
    categories = ['Basic', 'Kernel', 'Online', 'Multi-layer', 'Deep', 'Residual']

    print("\\begin{tabular}{ll" + "|ccc" * len(ds_names) + "}")
    print("\\toprule")

    # Header
    header = "\\textbf{Category} & \\textbf{Type}"
    for ds in ds_names:
        header += f" & \\multicolumn{{3}}{{c|}}{{{ds}}}"
    header = header.rstrip("|") + " \\\\"
    print(header)

    subheader = " & "
    for _ in ds_names:
        subheader += " & $R^2$ & RMSE & MAE"
    subheader += " \\\\"
    print(subheader)
    print("\\midrule")

    for cat in categories:
        # ELM row
        row = f"{cat} & ELM"
        for ds in ds_names:
            key = f'{cat}_ELM'
            if ds in results and key in results[ds]:
                r2_m, r2_s = results[ds][key]['r2']
                rmse_m, rmse_s = results[ds][key]['rmse']
                mae_m, mae_s = results[ds][key]['mae']
                if rmse_m > 1000:
                    row += f" & {r2_m:.2f} & {rmse_m/1000:.1f}k & {mae_m/1000:.1f}k"
                else:
                    row += f" & {r2_m:.2f} & {rmse_m:.1f} & {mae_m:.1f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)

        # RVFL row
        row = f" & RVFL"
        for ds in ds_names:
            key = f'{cat}_RVFL'
            if ds in results and key in results[ds]:
                r2_m, r2_s = results[ds][key]['r2']
                rmse_m, rmse_s = results[ds][key]['rmse']
                mae_m, mae_s = results[ds][key]['mae']
                if rmse_m > 1000:
                    row += f" & {r2_m:.2f} & {rmse_m/1000:.1f}k & {mae_m/1000:.1f}k"
                else:
                    row += f" & {r2_m:.2f} & {rmse_m:.1f} & {mae_m:.1f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def print_clustering_table(results, datasets):
    print("\n% CLUSTERING TABLE")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Clustering performance (mean $\\pm$ std). Sil=Silhouette, ARI=Adjusted Rand Index, NMI=Normalized Mutual Information.}")
    print("\\label{tab:clustering}")
    print("\\small")

    ds_names = [d[0] for d in datasets]

    print("\\begin{tabular}{l" + "|ccc" * len(ds_names) + "}")
    print("\\toprule")

    # Header
    header = "\\textbf{Model}"
    for ds in ds_names:
        header += f" & \\multicolumn{{3}}{{c|}}{{{ds}}}"
    header = header.rstrip("|") + " \\\\"
    print(header)

    subheader = ""
    for _ in ds_names:
        subheader += " & Sil & ARI & NMI"
    subheader += " \\\\"
    print(subheader)
    print("\\midrule")

    models = ['Unsupervised_ELM', 'Unsupervised_RVFL']
    model_labels = ['US-ELM', 'US-RVFL']

    for model, label in zip(models, model_labels):
        row = f"{label}"
        for ds in ds_names:
            if ds in results and model in results[ds]:
                sil_m, sil_s = results[ds][model]['sil']
                ari_m, ari_s = results[ds][model]['ari']
                nmi_m, nmi_s = results[ds][model]['nmi']
                row += f" & {sil_m:.2f}$\\pm${sil_s:.2f} & {ari_m:.2f}$\\pm${ari_s:.2f} & {nmi_m:.2f}$\\pm${nmi_s:.2f}"
            else:
                row += " & -- & -- & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("BENCHMARK BY CATEGORIES")
    print("=" * 100)

    # Classification
    print("\n" + "=" * 50)
    print("CLASSIFICATION")
    print("=" * 50)
    clf_results, clf_datasets = run_classification_benchmark()

    # Regression
    print("\n" + "=" * 50)
    print("REGRESSION")
    print("=" * 50)
    reg_results, reg_datasets = run_regression_benchmark()

    # Clustering
    print("\n" + "=" * 50)
    print("CLUSTERING")
    print("=" * 50)
    clust_results, clust_datasets = run_clustering_benchmark()

    # Generate LaTeX tables
    print("\n" + "=" * 100)
    print("LATEX TABLES")
    print("=" * 100)

    print_classification_table(clf_results, clf_datasets)
    print_regression_table(reg_results, reg_datasets)
    print_clustering_table(clust_results, clust_datasets)
