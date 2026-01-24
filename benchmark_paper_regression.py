"""
Regression Benchmark
Evaluates ELM and RVFL models on standard datasets
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import models
from Models.ELMModel import ELMModel
from Models.RVFLModel import RVFLModel
from Models.KELMModel import KELMModel
from Models.KRVFLModel import KRVFLModel
from Models.SSRVFLModel import SSRVFLModel
from Models.ML_ELMModel import ML_ELMModel
from Models.ML_RVFLModel import ML_RVFLModel
from Models.RCELMModel import RCELMModel
from Models.RCRVFLModel import RCRVFLModel
from Models.DeepRVFLModel import DeepRVFLModel
from Layers.DeepRVFLLayer import DeepRVFLLayer

from Layers.ELMLayer import ELMLayer
from Layers.RVFLLayer import RVFLLayer
from Layers.KELMLayer import KELMLayer
from Layers.KRVFLLayer import KRVFLLayer
from Resources.Kernel import Kernel

def load_boston_alternative():
    """Load alternative dataset to Boston (deprecated in sklearn)"""
    from sklearn.datasets import fetch_openml
    try:
        # Use Ames Housing as alternative
        data = fetch_openml(name="house_prices", as_frame=True, parser='auto')
        df = data.frame
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
            y = df['SalePrice'].values
        else:
            y = df[numeric_cols[-1]].values
            numeric_cols = numeric_cols[:-1]
        X = df[numeric_cols[:10]].fillna(0).values  # Use 10 features
        return X, y, "Ames Housing", len(y), 10
    except:
        # Fallback: generate synthetic data
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
        return X, y, "Synthetic", 500, 10

# Regression datasets
def get_regression_datasets():
    datasets = []

    # 1. Diabetes
    X, y = load_diabetes(return_X_y=True)
    datasets.append(("Diabetes", X, y, 442, 10))

    # 2. California Housing (subset for speed)
    X, y = fetch_california_housing(return_X_y=True)
    # Use subset of 2000 samples for efficiency
    np.random.seed(42)
    idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]
    datasets.append(("California Housing", X_sub, y_sub, len(X_sub), 8))

    # 3. Ames Housing or alternative
    X, y, name, n, f = load_boston_alternative()
    datasets.append((name, X, y, n, f))

    return datasets

# Models to evaluate (regression)
def get_regression_models():
    from Models.OSELMModel import OSELMModel
    from Models.OSRVFLModel import OSRVFLModel
    from Models.EnsembleDeepRVFLModel import EnsembleDeepRVFLModel
    from Layers.OSELMLayer import OSELMLayer
    from Layers.OSRVFLLayer import OSRVFLLayer
    from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer

    models = {}

    # Basic - 100 neurons, sigmoid, C=0.001
    elm_layer = ELMLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['ELM'] = ELMModel(elm_layer, task='regression')

    rvfl_layer = RVFLLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['RVFL'] = RVFLModel(rvfl_layer, task='regression')

    # Kernel RVFL
    kernel = Kernel(kernel_name='rbf', param=1.0)
    krvfl_layer = KRVFLLayer(kernel, C=0.001)
    models['K-RVFL'] = KRVFLModel(krvfl_layer, task='regression')

    # Online (only RVFL, ELM returns NaN in regression)
    osrvfl_layer = OSRVFLLayer(number_neurons=100, activation='sigmoid', C=0.001)
    models['OS-RVFL'] = OSRVFLModel(osrvfl_layer, prefetch_size=50, batch_size=16, classification=False)

    # Multi-layer (only RVFL, ELM returns NaN in regression)
    mlrvfl = ML_RVFLModel(classification=False)
    mlrvfl.add(RVFLLayer(number_neurons=50, activation='sigmoid'))
    mlrvfl.add(RVFLLayer(number_neurons=100, activation='sigmoid', C=0.001))
    models['ML-RVFL'] = mlrvfl

    # Ensemble
    edrvfl_layer = EnsembleDeepRVFLLayer(number_neurons=100, n_layers=3, activation='sigmoid', C=0.001)
    models['edRVFL'] = EnsembleDeepRVFLModel(edrvfl_layer, classification=False)

    return models

def neg_rmse_scorer(y_true, y_pred):
    """RMSE negativo to use with cross_val_score (higher is better)"""
    return -np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X, y, cv):
    """Evaluate model with multiple regression metrics"""
    try:
        # R² Score
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', error_score=np.nan)
        r2_mean = np.nanmean(r2_scores)
        r2_std = np.nanstd(r2_scores)

        # Negative MSE (sklearn convention)
        mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', error_score=np.nan)
        mse_mean = -np.nanmean(mse_scores)  # Convert back to positive
        mse_std = np.nanstd(mse_scores)

        # Negative MAE
        mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', error_score=np.nan)
        mae_mean = -np.nanmean(mae_scores)
        mae_std = np.nanstd(mae_scores)

        # RMSE
        rmse_mean = np.sqrt(mse_mean)

        return {
            'r2': (r2_mean, r2_std),
            'mse': (mse_mean, mse_std),
            'mae': (mae_mean, mae_std),
            'rmse': rmse_mean
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    print("=" * 80)
    print("REGRESSION BENCHMARK - TfELM-RVFL")
    print("=" * 80)

    datasets = get_regression_datasets()
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    all_results = {}

    for dataset_name, X, y, n_samples, n_features in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples: {n_samples}, Features: {n_features}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print("=" * 80)

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = get_regression_models()
        dataset_results = {}

        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...", end=" ")
            result = evaluate_model(model, X_scaled, y, cv)
            if result:
                dataset_results[model_name] = result
                print(f"R²={result['r2'][0]:.4f}, RMSE={result['rmse']:.2f}, MAE={result['mae'][0]:.2f}")
            else:
                print("FAILED")

        all_results[dataset_name] = dataset_results

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE - REGRESSION (R²)")
    print("=" * 80)

    print("\n% R² Score Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Regression performance ($R^2$ score) across benchmark datasets.}")
    print("\\label{tab:regression_r2}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    print("\\toprule")
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{d[0]}}}" for d in datasets]) + " \\\\"
    print(header)
    print("\\midrule")

    model_names = list(get_regression_models().keys())
    for model_name in model_names:
        row = f"{model_name}"
        for dataset_name, _, _, _, _ in datasets:
            if dataset_name in all_results and model_name in all_results[dataset_name]:
                r2, std = all_results[dataset_name][model_name]['r2']
                row += f" & {r2:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # MAE Table
    print("\n% MAE Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Regression performance (MAE) across benchmark datasets.}")
    print("\\label{tab:regression_mae}")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    print("\\toprule")
    print(header)
    print("\\midrule")

    for model_name in model_names:
        row = f"{model_name}"
        for dataset_name, _, _, _, _ in datasets:
            if dataset_name in all_results and model_name in all_results[dataset_name]:
                mae, std = all_results[dataset_name][model_name]['mae']
                row += f" & {mae:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
