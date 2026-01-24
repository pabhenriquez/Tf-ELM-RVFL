"""
Benchmark of all ELM and RVFL models for CLASSIFICATION
According to the scientific literature
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

print("=" * 70)
print("BENCHMARK: CLASSIFICATION MODELS")
print("=" * 70)

# Load data - Ionosphere
ionosphere = fetch_openml(name='ionosphere', version=1, as_frame=False)
X = StandardScaler().fit_transform(ionosphere.data.astype(np.float32))
y = (ionosphere.target == 'g').astype(np.int32)  # 'g' = good, 'b' = bad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Semi-supervised data
n_labeled = len(y_train) // 2
X_labeled = X_train[:n_labeled]
y_labeled = y_train[:n_labeled]
X_unlabeled = X_train[n_labeled:]
y_unlabeled = y_train[n_labeled:]

# Data for LRF (image)
img_size = 6
pad_size = img_size * img_size - X_train.shape[1]
X_train_img = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
X_test_img = np.pad(X_test, ((0, 0), (0, pad_size)), mode='constant')
X_train_img = X_train_img.reshape(-1, img_size, img_size, 1).astype(np.float32)
X_test_img = X_test_img.reshape(-1, img_size, img_size, 1).astype(np.float32)

print(f"\nDataset: Ionosphere ({len(X)} muestras, {X.shape[1]} features)")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

results = {}

print("\n" + "-" * 70)
print("BASIC MODELS (Classification + Regression)")
print("-" * 70)

# 1. ELM
try:
    from Layers.ELMLayer import ELMLayer
    from Models.ELMModel import ELMModel
    layer = ELMLayer(number_neurons=1000, activation='relu', C=1.0)
    model = ELMModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['ELM'] = acc
    print(f"  ELM:                    {acc:.4f}")
except Exception as e:
    print(f"  ELM: ERROR - {e}")

# 2. RVFL
try:
    from Layers.RVFLLayer import RVFLLayer
    from Models.RVFLModel import RVFLModel
    layer = RVFLLayer(number_neurons=1000, activation='relu', C=1.0)
    model = RVFLModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['RVFL'] = acc
    print(f"  RVFL:                   {acc:.4f}")
except Exception as e:
    print(f"  RVFL: ERROR - {e}")

# 3. KELM
try:
    from Layers.KELMLayer import KELMLayer
    from Models.KELMModel import KELMModel
    from Resources.Kernel import Kernel
    kernel = Kernel("rbf", param=1.0)
    layer = KELMLayer(kernel, activation='relu', C=1.0)
    model = KELMModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['KELM'] = acc
    print(f"  KELM:                   {acc:.4f}")
except Exception as e:
    print(f"  KELM: ERROR - {e}")

# 4. KRVFL
try:
    from Layers.KRVFLLayer import KRVFLLayer
    from Models.KRVFLModel import KRVFLModel
    kernel = Kernel("rbf", param=1.0)
    layer = KRVFLLayer(kernel, activation='relu', C=1.0)
    model = KRVFLModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['KRVFL'] = acc
    print(f"  KRVFL:                  {acc:.4f}")
except Exception as e:
    print(f"  KRVFL: ERROR - {e}")

# 5. SubELM
try:
    from Layers.SubELMLayer import SubELMLayer
    layer = SubELMLayer(number_neurons=1000, number_subnets=100, neurons_subnets=20, activation='relu')
    model = ELMModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['SubELM'] = acc
    print(f"  SubELM:                 {acc:.4f}")
except Exception as e:
    print(f"  SubELM: ERROR - {e}")

# 6. SubRVFL
try:
    from Layers.SubRVFLLayer import SubRVFLLayer
    layer = SubRVFLLayer(number_neurons=1000, activation='relu', feature_ratio=0.8)
    model = RVFLModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['SubRVFL'] = acc
    print(f"  SubRVFL:                {acc:.4f}")
except Exception as e:
    print(f"  SubRVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("WEIGHTED MODELS (Classification only - Zong et al., 2013)")
print("-" * 70)

# 7. WELM
try:
    from Layers.WELMLayer import WELMLayer
    layer = WELMLayer(number_neurons=1000, activation='relu', weight_method='wei-1')
    model = ELMModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['WELM'] = acc
    print(f"  WELM:                   {acc:.4f}")
except Exception as e:
    print(f"  WELM: ERROR - {e}")

# 8. WRVFL
try:
    from Layers.WRVFLLayer import WRVFLLayer
    layer = WRVFLLayer(number_neurons=1000, activation='relu', weight_method='wei-1')
    model = RVFLModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['WRVFL'] = acc
    print(f"  WRVFL:                  {acc:.4f}")
except Exception as e:
    print(f"  WRVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("ONLINE SEQUENTIAL MODELS (Classification + Regression)")
print("-" * 70)

# 9. OS-ELM
try:
    from Layers.OSELMLayer import OSELMLayer
    from Models.OSELMModel import OSELMModel
    layer = OSELMLayer(1000, 'tanh', C=0.001)
    model = OSELMModel(layer, prefetch_size=100, batch_size=32, classification=True)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['OS-ELM'] = acc
    print(f"  OS-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  OS-ELM: ERROR - {e}")

# 10. OS-RVFL
try:
    from Layers.OSRVFLLayer import OSRVFLLayer
    from Models.OSRVFLModel import OSRVFLModel
    layer = OSRVFLLayer(1000, 'tanh', C=0.001)
    model = OSRVFLModel(layer, prefetch_size=100, batch_size=32, classification=True)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['OS-RVFL'] = acc
    print(f"  OS-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  OS-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("SEMI-SUPERVISED MODELS (Classification + Regression)")
print("-" * 70)

# 11. SS-ELM
try:
    from Layers.SSELMLayer import SSELMLayer
    from Models.SSELMModel import SSELMModel
    layer = SSELMLayer(number_neurons=1000, lam=0.001, C=1.0)
    model = SSELMModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['SS-ELM'] = acc
    print(f"  SS-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  SS-ELM: ERROR - {e}")

# 12. SS-RVFL
try:
    from Layers.SSRVFLLayer import SSRVFLLayer
    from Models.SSRVFLModel import SSRVFLModel
    layer = SSRVFLLayer(number_neurons=1000, lam=0.001, C=1.0)
    model = SSRVFLModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['SS-RVFL'] = acc
    print(f"  SS-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  SS-RVFL: ERROR - {e}")

# 13. SSK-ELM
try:
    from Layers.SSKELMLayer import SSKELMLayer
    from Models.SSKELMModel import SSKELMModel
    kernel = Kernel("rbf", param=1.0)
    layer = SSKELMLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKELMModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['SSK-ELM'] = acc
    print(f"  SSK-ELM:                {acc:.4f}")
except Exception as e:
    print(f"  SSK-ELM: ERROR - {e}")

# 14. SSK-RVFL
try:
    from Layers.SSKRVFLLayer import SSKRVFLLayer
    from Models.SSKRVFLModel import SSKRVFLModel
    kernel = Kernel("rbf", param=1.0)
    layer = SSKRVFLLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKRVFLModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['SSK-RVFL'] = acc
    print(f"  SSK-RVFL:               {acc:.4f}")
except Exception as e:
    print(f"  SSK-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("DEEP/MULTI-LAYER MODELS (Classification + Regression)")
print("-" * 70)

# 15. DeepELM
try:
    from Models.DeepELMModel import DeepELMModel
    model = DeepELMModel(classification=True)
    model.add(ELMLayer(number_neurons=200, activation='relu'))
    model.add(ELMLayer(number_neurons=200, activation='relu'))
    model.add(ELMLayer(number_neurons=200, activation='relu'))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['DeepELM'] = acc
    print(f"  DeepELM:                {acc:.4f}")
except Exception as e:
    print(f"  DeepELM: ERROR - {e}")

# 16. DeepRVFL
try:
    from Layers.DeepRVFLLayer import DeepRVFLLayer
    from Models.DeepRVFLModel import DeepRVFLModel
    layer = DeepRVFLLayer(number_neurons=200, n_layers=3, activation='relu', C=1.0)
    model = DeepRVFLModel(layer, task='classification')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['DeepRVFL'] = acc
    print(f"  DeepRVFL:               {acc:.4f}")
except Exception as e:
    print(f"  DeepRVFL: ERROR - {e}")

# 17. EnsembleDeepRVFL
try:
    from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer
    from Models.EnsembleDeepRVFLModel import EnsembleDeepRVFLModel
    layer = EnsembleDeepRVFLLayer(number_neurons=200, n_layers=5, activation='relu', C=1.0)
    model = EnsembleDeepRVFLModel(layer, classification=True, ensemble_method='vote')
    model.fit(X_train, y_train)
    eval_results = model.evaluate(X_test, y_test)
    results['EnsembleRVFL_Vote'] = eval_results['vote_accuracy']
    results['EnsembleRVFL_Add'] = eval_results['addition_accuracy']
    print(f"  EnsembleRVFL (Vote):    {eval_results['vote_accuracy']:.4f}")
    print(f"  EnsembleRVFL (Add):     {eval_results['addition_accuracy']:.4f}")
except Exception as e:
    print(f"  EnsembleRVFL: ERROR - {e}")

# 18. ML-ELM
try:
    from Models.ML_ELMModel import ML_ELMModel
    from Layers.GELM_AE_Layer import GELM_AE_Layer
    model = ML_ELMModel(verbose=0)
    model.add(GELM_AE_Layer(number_neurons=100))
    model.add(GELM_AE_Layer(number_neurons=200))
    model.add(ELMLayer(number_neurons=1000))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['ML-ELM'] = acc
    print(f"  ML-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  ML-ELM: ERROR - {e}")

# 19. ML-RVFL
try:
    from Models.ML_RVFLModel import ML_RVFLModel
    from Layers.GRVFL_AE_Layer import GRVFL_AE_Layer
    model = ML_RVFLModel(verbose=0)
    model.add(GRVFL_AE_Layer(number_neurons=100))
    model.add(GRVFL_AE_Layer(number_neurons=200))
    model.add(RVFLLayer(number_neurons=1000))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['ML-RVFL'] = acc
    print(f"  ML-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  ML-RVFL: ERROR - {e}")

# 20. Dr-ELM
try:
    from Models.DrELMModel import DrELMModel
    model = DrELMModel(activation='relu', verbose=0)
    model.add(ELMLayer(number_neurons=200, activation='identity'))
    model.add(ELMLayer(number_neurons=200, activation='identity'))
    model.add(ELMLayer(number_neurons=200, activation='identity'))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['Dr-ELM'] = acc
    print(f"  Dr-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  Dr-ELM: ERROR - {e}")

# 21. Dr-RVFL
try:
    from Models.DrRVFLModel import DrRVFLModel
    model = DrRVFLModel(activation='relu', verbose=0)
    model.add(RVFLLayer(number_neurons=200, activation='identity'))
    model.add(RVFLLayer(number_neurons=200, activation='identity'))
    model.add(RVFLLayer(number_neurons=200, activation='identity'))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['Dr-RVFL'] = acc
    print(f"  Dr-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  Dr-RVFL: ERROR - {e}")

# 22. EHDr-ELM
try:
    from Models.EHDrELMModel import EHDrELMModel
    model = EHDrELMModel(verbose=0)
    model.add(ELMLayer(number_neurons=200, activation='relu', C=1.8))
    model.add(ELMLayer(number_neurons=200, activation='relu', C=1.8))
    model.add(ELMLayer(number_neurons=200, activation='relu', C=1.8))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['EHDr-ELM'] = acc
    print(f"  EHDr-ELM:               {acc:.4f}")
except Exception as e:
    print(f"  EHDr-ELM: ERROR - {e}")

# 23. EHDr-RVFL
try:
    from Models.EHDrRVFLModel import EHDrRVFLModel
    model = EHDrRVFLModel(verbose=0)
    model.add(RVFLLayer(number_neurons=200, activation='relu', C=1.8))
    model.add(RVFLLayer(number_neurons=200, activation='relu', C=1.8))
    model.add(RVFLLayer(number_neurons=200, activation='relu', C=1.8))
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results['EHDr-RVFL'] = acc
    print(f"  EHDr-RVFL:              {acc:.4f}")
except Exception as e:
    print(f"  EHDr-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("RC MODELS (Primarily Regression - Chen et al., 2018)")
print("-" * 70)

# 26. RC-ELM (works for classification but designed for regression)
try:
    from Models.RCELMModel import RCELMModel
    model = RCELMModel(verbose=0, task='classification')
    model.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    model.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    model.add(ELMLayer(number_neurons=1000, activation='relu', C=10))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test, pred)
    results['RC-ELM'] = acc
    print(f"  RC-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  RC-ELM: ERROR - {e}")

# 27. RC-RVFL
try:
    from Models.RCRVFLModel import RCRVFLModel
    model = RCRVFLModel(verbose=0, task='classification')
    model.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    model.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    model.add(RVFLLayer(number_neurons=1000, activation='relu', C=10))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test, pred)
    results['RC-RVFL'] = acc
    print(f"  RC-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  RC-RVFL: ERROR - {e}")

print("\n" + "-" * 70)
print("LRF MODELS (Image Classification only - Huang et al., 2015)")
print("-" * 70)

# 28. LRF-ELM
try:
    from Models.LRFELMModel import LRFELMModel
    layer = ELMLayer(number_neurons=1000, activation='relu', C=1.0)
    elm_model = ELMModel(layer, task='classification')
    model = LRFELMModel(elm_model=elm_model, num_feature_maps=16, filter_size=3,
                        num_input_channels=1, pool_size=2)
    model.fit(X_train_img, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_img))
    results['LRF-ELM'] = acc
    print(f"  LRF-ELM:                {acc:.4f}")
except Exception as e:
    print(f"  LRF-ELM: ERROR - {e}")

# 29. LRF-RVFL
try:
    from Models.LRFRVFLModel import LRFRVFLModel
    layer = RVFLLayer(number_neurons=1000, activation='relu', C=1.0)
    rvfl_model = RVFLModel(layer, task='classification')
    model = LRFRVFLModel(rvfl_model=rvfl_model, num_feature_maps=16, filter_size=3,
                         num_input_channels=1, pool_size=2)
    model.fit(X_train_img, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_img))
    results['LRF-RVFL'] = acc
    print(f"  LRF-RVFL:               {acc:.4f}")
except Exception as e:
    print(f"  LRF-RVFL: ERROR - {e}")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 70)
print("CLASSIFICATION RANKING (Sorted by Accuracy)")
print("=" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"\n{'Rank':<6} {'Model':<25} {'Accuracy':<10}")
print("-" * 45)
for i, (model, acc) in enumerate(sorted_results, 1):
    print(f"{i:<6} {model:<25} {acc:.4f}")

print(f"\n  Total models: {len(results)}")
print(f"  Mejor: {sorted_results[0][0]} ({sorted_results[0][1]:.4f})")
print(f"  Promedio: {np.mean(list(results.values())):.4f}")
print("=" * 70)
