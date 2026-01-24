"""
Benchmark of all ELM and RVFL models for REGRESSION
According to the scientific literature

NOTE: The following models are NOT included because they are classification only:
- WELM/WRVFL: Classification only (imbalanced data) - Zong et al., 2013
- LRF-ELM/LRF-RVFL: Image classification only - Huang et al., 2015
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

print("=" * 70)
print("BENCHMARK: REGRESSION MODELS")
print("=" * 70)

# Load data - Diabetes (faster for benchmark)
data = load_diabetes()
X = StandardScaler().fit_transform(data.data.astype(np.float32))
y = data.target.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Semi-supervised data
n_labeled = len(y_train) // 2
X_labeled = X_train[:n_labeled]
y_labeled = y_train[:n_labeled]
X_unlabeled = X_train[n_labeled:]
y_unlabeled = y_train[n_labeled:]

print(f"\nDataset: Diabetes ({len(X)} muestras, {X.shape[1]} features)")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Target: Diabetes progression")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

results = {}
r2_results = {}

print("\n" + "-" * 70)
print("BASIC MODELS (Classification + Regression)")
print("-" * 70)

# 1. ELM
try:
    from Layers.ELMLayer import ELMLayer
    from Models.ELMModel import ELMModel
    layer = ELMLayer(number_neurons=500, activation='mish', C=1.0)
    model = ELMModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['ELM'] = mse
    r2_results['ELM'] = r2
    print(f"  ELM:                    MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  ELM: ERROR - {e}")

# 2. RVFL
try:
    from Layers.RVFLLayer import RVFLLayer
    from Models.RVFLModel import RVFLModel
    layer = RVFLLayer(number_neurons=500, activation='mish', C=1.0)
    model = RVFLModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['RVFL'] = mse
    r2_results['RVFL'] = r2
    print(f"  RVFL:                   MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  RVFL: ERROR - {e}")

# 3. KELM
try:
    from Layers.KELMLayer import KELMLayer
    from Models.KELMModel import KELMModel
    from Resources.Kernel import Kernel
    kernel = Kernel("rbf", param=1.0)
    layer = KELMLayer(kernel, activation='mish', C=1.0)
    model = KELMModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['KELM'] = mse
    r2_results['KELM'] = r2
    print(f"  KELM:                   MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  KELM: ERROR - {e}")

# 4. KRVFL
try:
    from Layers.KRVFLLayer import KRVFLLayer
    from Models.KRVFLModel import KRVFLModel
    kernel = Kernel("rbf", param=1.0)
    layer = KRVFLLayer(kernel, activation='mish', C=1.0)
    model = KRVFLModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['KRVFL'] = mse
    r2_results['KRVFL'] = r2
    print(f"  KRVFL:                  MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  KRVFL: ERROR - {e}")

# 5. SubELM
try:
    from Layers.SubELMLayer import SubELMLayer
    layer = SubELMLayer(number_neurons=500, number_subnets=50, neurons_subnets=20, activation='mish')
    model = ELMModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['SubELM'] = mse
    r2_results['SubELM'] = r2
    print(f"  SubELM:                 MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  SubELM: ERROR - {e}")

# 6. SubRVFL
try:
    from Layers.SubRVFLLayer import SubRVFLLayer
    layer = SubRVFLLayer(number_neurons=500, activation='mish', feature_ratio=0.8)
    model = RVFLModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['SubRVFL'] = mse
    r2_results['SubRVFL'] = r2
    print(f"  SubRVFL:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  SubRVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("ONLINE SEQUENTIAL MODELS (Classification + Regression)")
print("-" * 70)

# 7. OS-ELM
try:
    from Layers.OSELMLayer import OSELMLayer
    from Models.OSELMModel import OSELMModel
    layer = OSELMLayer(500, 'tanh', C=0.001)
    model = OSELMModel(layer, prefetch_size=100, batch_size=32, classification=False)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['OS-ELM'] = mse
    r2_results['OS-ELM'] = r2
    print(f"  OS-ELM:                 MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  OS-ELM: ERROR - {e}")

# 8. OS-RVFL
try:
    from Layers.OSRVFLLayer import OSRVFLLayer
    from Models.OSRVFLModel import OSRVFLModel
    layer = OSRVFLLayer(500, 'tanh', C=0.001)
    model = OSRVFLModel(layer, prefetch_size=100, batch_size=32, classification=False)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['OS-RVFL'] = mse
    r2_results['OS-RVFL'] = r2
    print(f"  OS-RVFL:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  OS-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("SEMI-SUPERVISED MODELS (Classification + Regression)")
print("-" * 70)

# 9. SS-ELM
try:
    from Layers.SSELMLayer import SSELMLayer
    from Models.SSELMModel import SSELMModel
    layer = SSELMLayer(number_neurons=500, lam=0.001, C=1.0)
    model = SSELMModel(layer, classification=False)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['SS-ELM'] = mse
    r2_results['SS-ELM'] = r2
    print(f"  SS-ELM:                 MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  SS-ELM: ERROR - {e}")

# 10. SS-RVFL
try:
    from Layers.SSRVFLLayer import SSRVFLLayer
    from Models.SSRVFLModel import SSRVFLModel
    layer = SSRVFLLayer(number_neurons=500, lam=0.001, C=1.0)
    model = SSRVFLModel(layer, classification=False)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['SS-RVFL'] = mse
    r2_results['SS-RVFL'] = r2
    print(f"  SS-RVFL:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  SS-RVFL: ERROR - {e}")

# 11. SSK-ELM
try:
    from Layers.SSKELMLayer import SSKELMLayer
    from Models.SSKELMModel import SSKELMModel
    kernel = Kernel("rbf", param=1.0)
    layer = SSKELMLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKELMModel(layer, classification=False)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['SSK-ELM'] = mse
    r2_results['SSK-ELM'] = r2
    print(f"  SSK-ELM:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  SSK-ELM: ERROR - {e}")

# 12. SSK-RVFL
try:
    from Layers.SSKRVFLLayer import SSKRVFLLayer
    from Models.SSKRVFLModel import SSKRVFLModel
    kernel = Kernel("rbf", param=1.0)
    layer = SSKRVFLLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKRVFLModel(layer, classification=False)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['SSK-RVFL'] = mse
    r2_results['SSK-RVFL'] = r2
    print(f"  SSK-RVFL:               MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  SSK-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("DEEP/MULTI-LAYER MODELS (Classification + Regression)")
print("-" * 70)

# 13. DeepELM
try:
    from Models.DeepELMModel import DeepELMModel
    model = DeepELMModel(classification=False)
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['DeepELM'] = mse
    r2_results['DeepELM'] = r2
    print(f"  DeepELM:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  DeepELM: ERROR - {e}")

# 14. DeepRVFL
try:
    from Layers.DeepRVFLLayer import DeepRVFLLayer
    from Models.DeepRVFLModel import DeepRVFLModel
    layer = DeepRVFLLayer(number_neurons=100, n_layers=3, activation='relu', C=1.0)
    model = DeepRVFLModel(layer, task='regression')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['DeepRVFL'] = mse
    r2_results['DeepRVFL'] = r2
    print(f"  DeepRVFL:               MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  DeepRVFL: ERROR - {e}")

# 15. ML-ELM
try:
    from Models.ML_ELMModel import ML_ELMModel
    from Layers.GELM_AE_Layer import GELM_AE_Layer
    model = ML_ELMModel(verbose=0)
    model.add(GELM_AE_Layer(number_neurons=50))
    model.add(GELM_AE_Layer(number_neurons=100))
    model.add(ELMLayer(number_neurons=500))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['ML-ELM'] = mse
    r2_results['ML-ELM'] = r2
    print(f"  ML-ELM:                 MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  ML-ELM: ERROR - {e}")

# 16. ML-RVFL
try:
    from Models.ML_RVFLModel import ML_RVFLModel
    from Layers.GRVFL_AE_Layer import GRVFL_AE_Layer
    model = ML_RVFLModel(verbose=0)
    model.add(GRVFL_AE_Layer(number_neurons=50))
    model.add(GRVFL_AE_Layer(number_neurons=100))
    model.add(RVFLLayer(number_neurons=500))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['ML-RVFL'] = mse
    r2_results['ML-RVFL'] = r2
    print(f"  ML-RVFL:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  ML-RVFL: ERROR - {e}")

# 17. Dr-ELM
try:
    from Models.DrELMModel import DrELMModel
    model = DrELMModel(activation='mish', verbose=0)
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['Dr-ELM'] = mse
    r2_results['Dr-ELM'] = r2
    print(f"  Dr-ELM:                 MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  Dr-ELM: ERROR - {e}")

# 18. Dr-RVFL
try:
    from Models.DrRVFLModel import DrRVFLModel
    model = DrRVFLModel(activation='mish', verbose=0)
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['Dr-RVFL'] = mse
    r2_results['Dr-RVFL'] = r2
    print(f"  Dr-RVFL:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  Dr-RVFL: ERROR - {e}")

# 19. EHDr-ELM
try:
    from Models.EHDrELMModel import EHDrELMModel
    model = EHDrELMModel(verbose=0)
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['EHDr-ELM'] = mse
    r2_results['EHDr-ELM'] = r2
    print(f"  EHDr-ELM:               MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  EHDr-ELM: ERROR - {e}")

# 20. EHDr-RVFL
try:
    from Models.EHDrRVFLModel import EHDrRVFLModel
    model = EHDrRVFLModel(verbose=0)
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['EHDr-RVFL'] = mse
    r2_results['EHDr-RVFL'] = r2
    print(f"  EHDr-RVFL:              MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  EHDr-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("RC MODELS (Primarily REGRESSION - Chen et al., 2018)")
print("-" * 70)

# 23. RC-ELM (designed primarily for REGRESSION)
try:
    from Models.RCELMModel import RCELMModel
    model = RCELMModel(verbose=0, task='regression')
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['RC-ELM'] = mse
    r2_results['RC-ELM'] = r2
    print(f"  RC-ELM:                 MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  RC-ELM: ERROR - {e}")

# 24. RC-RVFL (designed primarily for REGRESSION)
try:
    from Models.RCRVFLModel import RCRVFLModel
    model = RCRVFLModel(verbose=0, task='regression')
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results['RC-RVFL'] = mse
    r2_results['RC-RVFL'] = r2
    print(f"  RC-RVFL:                MSE={mse:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  RC-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("EXCLUDED MODELS (Classification only according to literature)")
print("-" * 70)
print("  WELM:                   EXCLUDED (classification only - Zong et al., 2013)")
print("  WRVFL:                  EXCLUDED (classification only - Zong et al., 2013)")
print("  LRF-ELM:                EXCLUDED (images only - Huang et al., 2015)")
print("  LRF-RVFL:               EXCLUDED (images only - Huang et al., 2015)")
print("  EnsembleDeepRVFL:       EXCLUDED (classification only - voting)")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 70)
print("REGRESSION RANKING (Sorted by MSE - lower is better)")
print("=" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1])
print(f"\n{'Rank':<6} {'Model':<20} {'MSE':<12} {'R²':<10}")
print("-" * 50)
for i, (model, mse) in enumerate(sorted_results, 1):
    r2 = r2_results.get(model, 0)
    print(f"{i:<6} {model:<20} {mse:<12.2f} {r2:<10.4f}")

print(f"\n  Total modelos: {len(results)}")
print(f"  Mejor: {sorted_results[0][0]} (MSE={sorted_results[0][1]:.2f})")
print(f"  MSE Promedio: {np.mean(list(results.values())):.2f}")
print(f"  R² Promedio: {np.mean(list(r2_results.values())):.4f}")
print("=" * 70)
