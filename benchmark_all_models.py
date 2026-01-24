"""
Benchmark complete for all models ELM y RVFL
for CLASSIFICATION and REGRESSION problems

NOTE: According to scientific literature, some models have task restrictions:
- WELM/WRVFL: CLASSIFICATION only (designed for imbalanced data)
- LRF-ELM/LRF-RVFL: Solo IMAGE CLASSIFICATION
- RC-ELM/RC-RVFL: Primarily REGRESSION (residual compensation)
- US-ELM/US-RVFL/USK-ELM/USK-RVFL: Solo EMBEDDING/CLUSTERING (no supervisados)

Referencias:
- WELM: Zong et al. "Weighted ELM for imbalance learning" (Neurocomputing, 2013)
- RC-ELM: Chen et al. "Residual compensation ELM for regression" (Neurocomputing, 2018)
- LRF-ELM: Huang et al. "Local Receptive Fields Based ELM" (IEEE CI Magazine, 2015)
- US-ELM: Huang et al. "Unsupervised extreme learning machines" (Neurocomputing, 2014)
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_breast_cancer, load_diabetes

print("=" * 80)
print("COMPLETE BENCHMARK: ALL ELM AND RVFL MODELS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

# Dataset for classification
clf_data = load_breast_cancer()
X_clf = StandardScaler().fit_transform(clf_data.data.astype(np.float32))
y_clf = clf_data.target
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Dataset for regression
reg_data = load_diabetes()
X_reg = StandardScaler().fit_transform(reg_data.data.astype(np.float32))
y_reg = reg_data.target.astype(np.float32)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Data semi-supervised
n_labeled = len(y_train_clf) // 2
X_labeled = X_train_clf[:n_labeled]
y_labeled = y_train_clf[:n_labeled]
X_unlabeled = X_train_clf[n_labeled:]
y_unlabeled = y_train_clf[n_labeled:]

# Data for LRF (imagen)
img_size = 6
pad_size = img_size * img_size - X_train_clf.shape[1]
X_train_img = np.pad(X_train_clf, ((0, 0), (0, pad_size)), mode='constant')
X_test_img = np.pad(X_test_clf, ((0, 0), (0, pad_size)), mode='constant')
X_train_img = X_train_img.reshape(-1, img_size, img_size, 1).astype(np.float32)
X_test_img = X_test_img.reshape(-1, img_size, img_size, 1).astype(np.float32)

print(f"\nDataset Classification: Breast Cancer ({len(X_clf)} samples, {X_clf.shape[1]} features)")
print(f"Dataset Regression: Diabetes ({len(X_reg)} samples, {X_reg.shape[1]} features)")

# Resultados
clf_results = {}
reg_results = {}

# ============================================================================
# CLASSIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("CLASSIFICATION - All models")
print("=" * 80)

# 1. ELM Basic
try:
    from Layers.ELMLayer import ELMLayer
    from Models.ELMModel import ELMModel
    layer = ELMLayer(number_neurons=500, activation='mish', C=1.0)
    model = ELMModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['ELM'] = acc
    print(f"  ELM:                    {acc:.4f}")
except Exception as e:
    print(f"  ELM: ERROR - {e}")

# 2. RVFL Basic
try:
    from Layers.RVFLLayer import RVFLLayer
    from Models.RVFLModel import RVFLModel
    layer = RVFLLayer(number_neurons=500, activation='mish', C=1.0)
    model = RVFLModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['RVFL'] = acc
    print(f"  RVFL:                   {acc:.4f}")
except Exception as e:
    print(f"  RVFL: ERROR - {e}")

# 3. WELM (Weighted)
try:
    from Layers.WELMLayer import WELMLayer
    layer = WELMLayer(number_neurons=500, activation='mish', weight_method='wei-1')
    model = ELMModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['WELM'] = acc
    print(f"  WELM:                   {acc:.4f}")
except Exception as e:
    print(f"  WELM: ERROR - {e}")

# 4. WRVFL (Weighted)
try:
    from Layers.WRVFLLayer import WRVFLLayer
    layer = WRVFLLayer(number_neurons=500, activation='mish', weight_method='wei-1')
    model = RVFLModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['WRVFL'] = acc
    print(f"  WRVFL:                  {acc:.4f}")
except Exception as e:
    print(f"  WRVFL: ERROR - {e}")

# 5. KELM (Kernel)
try:
    from Layers.KELMLayer import KELMLayer
    from Models.KELMModel import KELMModel
    from Resources.Kernel import Kernel
    kernel = Kernel("rbf", param=1.0)
    layer = KELMLayer(kernel, activation='mish', C=1.0)
    model = KELMModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['KELM'] = acc
    print(f"  KELM:                   {acc:.4f}")
except Exception as e:
    print(f"  KELM: ERROR - {e}")

# 6. KRVFL (Kernel)
try:
    from Layers.KRVFLLayer import KRVFLLayer
    from Models.KRVFLModel import KRVFLModel
    kernel = Kernel("rbf", param=1.0)
    layer = KRVFLLayer(kernel, activation='mish', C=1.0)
    model = KRVFLModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['KRVFL'] = acc
    print(f"  KRVFL:                  {acc:.4f}")
except Exception as e:
    print(f"  KRVFL: ERROR - {e}")

# 7. Deep ELM
try:
    from Models.DeepELMModel import DeepELMModel
    model = DeepELMModel(classification=True)
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['DeepELM'] = acc
    print(f"  DeepELM:                {acc:.4f}")
except Exception as e:
    print(f"  DeepELM: ERROR - {e}")

# 8. Deep RVFL
try:
    from Layers.DeepRVFLLayer import DeepRVFLLayer
    from Models.DeepRVFLModel import DeepRVFLModel
    layer = DeepRVFLLayer(number_neurons=100, n_layers=3, activation='relu', C=1.0)
    model = DeepRVFLModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['DeepRVFL'] = acc
    print(f"  DeepRVFL:               {acc:.4f}")
except Exception as e:
    print(f"  DeepRVFL: ERROR - {e}")

# 9. Ensemble Deep RVFL
try:
    from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer
    from Models.EnsembleDeepRVFLModel import EnsembleDeepRVFLModel
    layer = EnsembleDeepRVFLLayer(number_neurons=100, n_layers=5, activation='relu', C=1.0)
    model = EnsembleDeepRVFLModel(layer, classification=True, ensemble_method='vote')
    model.fit(X_train_clf, y_train_clf)
    eval_results = model.evaluate(X_test_clf, y_test_clf)
    clf_results['EnsembleRVFL_Vote'] = eval_results['vote_accuracy']
    clf_results['EnsembleRVFL_Add'] = eval_results['addition_accuracy']
    print(f"  EnsembleRVFL (Vote):    {eval_results['vote_accuracy']:.4f}")
    print(f"  EnsembleRVFL (Add):     {eval_results['addition_accuracy']:.4f}")
except Exception as e:
    print(f"  EnsembleRVFL: ERROR - {e}")

# 10. OS-ELM (Online Sequential)
try:
    from Layers.OSELMLayer import OSELMLayer
    from Models.OSELMModel import OSELMModel
    layer = OSELMLayer(500, 'tanh', C=0.001)
    model = OSELMModel(layer, prefetch_size=100, batch_size=32, classification=True)
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['OS-ELM'] = acc
    print(f"  OS-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  OS-ELM: ERROR - {e}")

# 11. OS-RVFL (Online Sequential)
try:
    from Layers.OSRVFLLayer import OSRVFLLayer
    from Models.OSRVFLModel import OSRVFLModel
    layer = OSRVFLLayer(500, 'tanh', C=0.001)
    model = OSRVFLModel(layer, prefetch_size=100, batch_size=32, classification=True)
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['OS-RVFL'] = acc
    print(f"  OS-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  OS-RVFL: ERROR - {e}")

# 12. SS-ELM (Semi-Supervised)
try:
    from Layers.SSELMLayer import SSELMLayer
    from Models.SSELMModel import SSELMModel
    layer = SSELMLayer(number_neurons=500, lam=0.001, C=1.0)
    model = SSELMModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['SS-ELM'] = acc
    print(f"  SS-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  SS-ELM: ERROR - {e}")

# 13. SS-RVFL (Semi-Supervised)
try:
    from Layers.SSRVFLLayer import SSRVFLLayer
    from Models.SSRVFLModel import SSRVFLModel
    layer = SSRVFLLayer(number_neurons=500, lam=0.001, C=1.0)
    model = SSRVFLModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['SS-RVFL'] = acc
    print(f"  SS-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  SS-RVFL: ERROR - {e}")

# 14. SSK-ELM (Semi-Supervised Kernel)
try:
    from Layers.SSKELMLayer import SSKELMLayer
    from Models.SSKELMModel import SSKELMModel
    kernel = Kernel("rbf", param=1.0)
    layer = SSKELMLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKELMModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['SSK-ELM'] = acc
    print(f"  SSK-ELM:                {acc:.4f}")
except Exception as e:
    print(f"  SSK-ELM: ERROR - {e}")

# 15. SSK-RVFL (Semi-Supervised Kernel)
try:
    from Layers.SSKRVFLLayer import SSKRVFLLayer
    from Models.SSKRVFLModel import SSKRVFLModel
    kernel = Kernel("rbf", param=1.0)
    layer = SSKRVFLLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKRVFLModel(layer, classification=True)
    model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['SSK-RVFL'] = acc
    print(f"  SSK-RVFL:               {acc:.4f}")
except Exception as e:
    print(f"  SSK-RVFL: ERROR - {e}")

# 16. ML-ELM (Multi-Layer)
try:
    from Models.ML_ELMModel import ML_ELMModel
    from Layers.GELM_AE_Layer import GELM_AE_Layer
    model = ML_ELMModel(verbose=0)
    model.add(GELM_AE_Layer(number_neurons=50))
    model.add(GELM_AE_Layer(number_neurons=100))
    model.add(ELMLayer(number_neurons=500))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['ML-ELM'] = acc
    print(f"  ML-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  ML-ELM: ERROR - {e}")

# 17. ML-RVFL (Multi-Layer)
try:
    from Models.ML_RVFLModel import ML_RVFLModel
    from Layers.GRVFL_AE_Layer import GRVFL_AE_Layer
    model = ML_RVFLModel(verbose=0)
    model.add(GRVFL_AE_Layer(number_neurons=50))
    model.add(GRVFL_AE_Layer(number_neurons=100))
    model.add(RVFLLayer(number_neurons=500))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['ML-RVFL'] = acc
    print(f"  ML-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  ML-RVFL: ERROR - {e}")

# 18. Dr-ELM (Deep Representation)
try:
    from Models.DrELMModel import DrELMModel
    model = DrELMModel(activation='mish', verbose=0)
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['Dr-ELM'] = acc
    print(f"  Dr-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  Dr-ELM: ERROR - {e}")

# 19. Dr-RVFL (Deep Representation)
try:
    from Models.DrRVFLModel import DrRVFLModel
    model = DrRVFLModel(activation='mish', verbose=0)
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['Dr-RVFL'] = acc
    print(f"  Dr-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  Dr-RVFL: ERROR - {e}")

# 20. EHDr-ELM (Enhanced Deep Representation)
try:
    from Models.EHDrELMModel import EHDrELMModel
    model = EHDrELMModel(verbose=0)
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['EHDr-ELM'] = acc
    print(f"  EHDr-ELM:               {acc:.4f}")
except Exception as e:
    print(f"  EHDr-ELM: ERROR - {e}")

# 21. EHDr-RVFL (Enhanced Deep Representation)
try:
    from Models.EHDrRVFLModel import EHDrRVFLModel
    model = EHDrRVFLModel(verbose=0)
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['EHDr-RVFL'] = acc
    print(f"  EHDr-RVFL:              {acc:.4f}")
except Exception as e:
    print(f"  EHDr-RVFL: ERROR - {e}")

# 22. RC-ELM (Residual Compensation)
try:
    from Models.RCELMModel import RCELMModel
    model = RCELMModel(verbose=0)
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.fit(X_train_clf, y_train_clf)
    pred = model.predict(X_test_clf)
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test_clf, pred)
    clf_results['RC-ELM'] = acc
    print(f"  RC-ELM:                 {acc:.4f}")
except Exception as e:
    print(f"  RC-ELM: ERROR - {e}")

# 23. RC-RVFL (Residual Compensation)
try:
    from Models.RCRVFLModel import RCRVFLModel
    model = RCRVFLModel(verbose=0)
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.fit(X_train_clf, y_train_clf)
    pred = model.predict(X_test_clf)
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test_clf, pred)
    clf_results['RC-RVFL'] = acc
    print(f"  RC-RVFL:                {acc:.4f}")
except Exception as e:
    print(f"  RC-RVFL: ERROR - {e}")

# 24. SubELM (Random Subspace)
try:
    from Layers.SubELMLayer import SubELMLayer
    layer = SubELMLayer(number_neurons=500, number_subnets=50, neurons_subnets=20, activation='mish')
    model = ELMModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['SubELM'] = acc
    print(f"  SubELM:                 {acc:.4f}")
except Exception as e:
    print(f"  SubELM: ERROR - {e}")

# 27. SubRVFL (Random Subspace)
try:
    from Layers.SubRVFLLayer import SubRVFLLayer
    layer = SubRVFLLayer(number_neurons=500, activation='mish', feature_ratio=0.8)
    model = RVFLModel(layer, task='classification')
    model.fit(X_train_clf, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf))
    clf_results['SubRVFL'] = acc
    print(f"  SubRVFL:                {acc:.4f}")
except Exception as e:
    print(f"  SubRVFL: ERROR - {e}")

# 28. LRF-ELM (Local Receptive Field)
try:
    from Models.LRFELMModel import LRFELMModel
    layer = ELMLayer(number_neurons=500, activation='mish', C=1.0)
    elm_model = ELMModel(layer, task='classification')
    model = LRFELMModel(elm_model=elm_model, num_feature_maps=16, filter_size=3,
                        num_input_channels=1, pool_size=2)
    model.fit(X_train_img, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_img))
    clf_results['LRF-ELM'] = acc
    print(f"  LRF-ELM:                {acc:.4f}")
except Exception as e:
    print(f"  LRF-ELM: ERROR - {e}")

# 29. LRF-RVFL (Local Receptive Field)
try:
    from Models.LRFRVFLModel import LRFRVFLModel
    layer = RVFLLayer(number_neurons=500, activation='mish', C=1.0)
    rvfl_model = RVFLModel(layer, task='classification')
    model = LRFRVFLModel(rvfl_model=rvfl_model, num_feature_maps=16, filter_size=3,
                         num_input_channels=1, pool_size=2)
    model.fit(X_train_img, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_img))
    clf_results['LRF-RVFL'] = acc
    print(f"  LRF-RVFL:               {acc:.4f}")
except Exception as e:
    print(f"  LRF-RVFL: ERROR - {e}")


# ============================================================================
# REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("REGRESSION - All models")
print("=" * 80)

# Data semi-supervised for regression
n_labeled_reg = len(y_train_reg) // 2
X_labeled_reg = X_train_reg[:n_labeled_reg]
y_labeled_reg = y_train_reg[:n_labeled_reg]
X_unlabeled_reg = X_train_reg[n_labeled_reg:]
y_unlabeled_reg = y_train_reg[n_labeled_reg:]

# 1. ELM Regression
try:
    layer = ELMLayer(number_neurons=500, activation='mish', C=1.0)
    model = ELMModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['ELM'] = mse
    print(f"  ELM:                    {mse:.2f}")
except Exception as e:
    print(f"  ELM: ERROR - {e}")

# 2. RVFL Regression
try:
    layer = RVFLLayer(number_neurons=500, activation='mish', C=1.0)
    model = RVFLModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['RVFL'] = mse
    print(f"  RVFL:                   {mse:.2f}")
except Exception as e:
    print(f"  RVFL: ERROR - {e}")

# NOTA: WELM/WRVFL NO se prueban for regression
# According to literature (Zong et al., 2013), WELM was specifically designed
# for classification for imbalanced data. The weighting mechanism is based
# on class distribution, which is not applicable to regression.
print("  WELM:                   SKIP (classification only - literature)")
print("  WRVFL:                  SKIP (classification only - literature)")

# 5. KELM Regression
try:
    kernel = Kernel("rbf", param=1.0)
    layer = KELMLayer(kernel, activation='mish', C=1.0)
    model = KELMModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['KELM'] = mse
    print(f"  KELM:                   {mse:.2f}")
except Exception as e:
    print(f"  KELM: ERROR - {e}")

# 6. KRVFL Regression
try:
    kernel = Kernel("rbf", param=1.0)
    layer = KRVFLLayer(kernel, activation='mish', C=1.0)
    model = KRVFLModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['KRVFL'] = mse
    print(f"  KRVFL:                  {mse:.2f}")
except Exception as e:
    print(f"  KRVFL: ERROR - {e}")

# 7. DeepELM Regression
try:
    model = DeepELMModel(classification=False)
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.add(ELMLayer(number_neurons=100, activation='relu'))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['DeepELM'] = mse
    print(f"  DeepELM:                {mse:.2f}")
except Exception as e:
    print(f"  DeepELM: ERROR - {e}")

# 8. DeepRVFL Regression
try:
    layer = DeepRVFLLayer(number_neurons=100, n_layers=3, activation='relu', C=1.0)
    model = DeepRVFLModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['DeepRVFL'] = mse
    print(f"  DeepRVFL:               {mse:.2f}")
except Exception as e:
    print(f"  DeepRVFL: ERROR - {e}")

# 9. OS-ELM Regression
try:
    layer = OSELMLayer(500, 'tanh', C=0.001)
    model = OSELMModel(layer, prefetch_size=100, batch_size=32, classification=False)
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['OS-ELM'] = mse
    print(f"  OS-ELM:                 {mse:.2f}")
except Exception as e:
    print(f"  OS-ELM: ERROR - {e}")

# 10. OS-RVFL Regression
try:
    layer = OSRVFLLayer(500, 'tanh', C=0.001)
    model = OSRVFLModel(layer, prefetch_size=100, batch_size=32, classification=False)
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['OS-RVFL'] = mse
    print(f"  OS-RVFL:                {mse:.2f}")
except Exception as e:
    print(f"  OS-RVFL: ERROR - {e}")

# 11. SS-ELM Regression
try:
    layer = SSELMLayer(number_neurons=500, lam=0.001, C=1.0)
    model = SSELMModel(layer, classification=False)
    model.fit(X_labeled_reg, X_unlabeled_reg, y_labeled_reg, y_unlabeled_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['SS-ELM'] = mse
    print(f"  SS-ELM:                 {mse:.2f}")
except Exception as e:
    print(f"  SS-ELM: ERROR - {e}")

# 12. SS-RVFL Regression
try:
    layer = SSRVFLLayer(number_neurons=500, lam=0.001, C=1.0)
    model = SSRVFLModel(layer, classification=False)
    model.fit(X_labeled_reg, X_unlabeled_reg, y_labeled_reg, y_unlabeled_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['SS-RVFL'] = mse
    print(f"  SS-RVFL:                {mse:.2f}")
except Exception as e:
    print(f"  SS-RVFL: ERROR - {e}")

# 13. SSK-ELM Regression
try:
    kernel = Kernel("rbf", param=1.0)
    layer = SSKELMLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKELMModel(layer, classification=False)
    model.fit(X_labeled_reg, X_unlabeled_reg, y_labeled_reg, y_unlabeled_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['SSK-ELM'] = mse
    print(f"  SSK-ELM:                {mse:.2f}")
except Exception as e:
    print(f"  SSK-ELM: ERROR - {e}")

# 14. SSK-RVFL Regression
try:
    kernel = Kernel("rbf", param=1.0)
    layer = SSKRVFLLayer(kernel=kernel, lam=0.001, C=1.0)
    model = SSKRVFLModel(layer, classification=False)
    model.fit(X_labeled_reg, X_unlabeled_reg, y_labeled_reg, y_unlabeled_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['SSK-RVFL'] = mse
    print(f"  SSK-RVFL:               {mse:.2f}")
except Exception as e:
    print(f"  SSK-RVFL: ERROR - {e}")

# 15. ML-ELM Regression
try:
    model = ML_ELMModel(verbose=0)
    model.add(GELM_AE_Layer(number_neurons=50))
    model.add(GELM_AE_Layer(number_neurons=100))
    model.add(ELMLayer(number_neurons=500))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['ML-ELM'] = mse
    print(f"  ML-ELM:                 {mse:.2f}")
except Exception as e:
    print(f"  ML-ELM: ERROR - {e}")

# 16. ML-RVFL Regression
try:
    model = ML_RVFLModel(verbose=0)
    model.add(GRVFL_AE_Layer(number_neurons=50))
    model.add(GRVFL_AE_Layer(number_neurons=100))
    model.add(RVFLLayer(number_neurons=500))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['ML-RVFL'] = mse
    print(f"  ML-RVFL:                {mse:.2f}")
except Exception as e:
    print(f"  ML-RVFL: ERROR - {e}")

# 17. Dr-ELM Regression
try:
    model = DrELMModel(activation='mish', verbose=0)
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.add(ELMLayer(number_neurons=100, activation='identity'))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['Dr-ELM'] = mse
    print(f"  Dr-ELM:                 {mse:.2f}")
except Exception as e:
    print(f"  Dr-ELM: ERROR - {e}")

# 18. Dr-RVFL Regression
try:
    model = DrRVFLModel(activation='mish', verbose=0)
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.add(RVFLLayer(number_neurons=100, activation='identity'))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['Dr-RVFL'] = mse
    print(f"  Dr-RVFL:                {mse:.2f}")
except Exception as e:
    print(f"  Dr-RVFL: ERROR - {e}")

# 19. EHDr-ELM Regression
try:
    model = EHDrELMModel(verbose=0)
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(ELMLayer(number_neurons=100, activation='mish', C=1.8))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['EHDr-ELM'] = mse
    print(f"  EHDr-ELM:               {mse:.2f}")
except Exception as e:
    print(f"  EHDr-ELM: ERROR - {e}")

# 20. EHDr-RVFL Regression
try:
    model = EHDrRVFLModel(verbose=0)
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.add(RVFLLayer(number_neurons=100, activation='mish', C=1.8))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['EHDr-RVFL'] = mse
    print(f"  EHDr-RVFL:              {mse:.2f}")
except Exception as e:
    print(f"  EHDr-RVFL: ERROR - {e}")

# 21. RC-ELM Regression
try:
    model = RCELMModel(verbose=0)
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(ELMLayer(number_neurons=500, activation='sigmoid', C=10))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['RC-ELM'] = mse
    print(f"  RC-ELM:                 {mse:.2f}")
except Exception as e:
    print(f"  RC-ELM: ERROR - {e}")

# 22. RC-RVFL Regression
try:
    model = RCRVFLModel(verbose=0)
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.add(RVFLLayer(number_neurons=500, activation='sigmoid', C=10))
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['RC-RVFL'] = mse
    print(f"  RC-RVFL:                {mse:.2f}")
except Exception as e:
    print(f"  RC-RVFL: ERROR - {e}")

# 23. SubELM Regression
try:
    layer = SubELMLayer(number_neurons=500, number_subnets=50, neurons_subnets=20, activation='mish')
    model = ELMModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['SubELM'] = mse
    print(f"  SubELM:                 {mse:.2f}")
except Exception as e:
    print(f"  SubELM: ERROR - {e}")

# 26. SubRVFL Regression
try:
    layer = SubRVFLLayer(number_neurons=500, activation='mish', feature_ratio=0.8)
    model = RVFLModel(layer, task='regression')
    model.fit(X_train_reg, y_train_reg)
    mse = mean_squared_error(y_test_reg, model.predict(X_test_reg))
    reg_results['SubRVFL'] = mse
    print(f"  SubRVFL:                {mse:.2f}")
except Exception as e:
    print(f"  SubRVFL: ERROR - {e}")

# 27. EnsembleRVFL Regression
try:
    layer = EnsembleDeepRVFLLayer(number_neurons=100, n_layers=5, activation='relu', C=1.0)
    model = EnsembleDeepRVFLModel(layer, classification=False, ensemble_method='add')
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg, method='addition')
    if len(pred.shape) > 1:
        pred = pred.flatten()
    mse = mean_squared_error(y_test_reg, pred)
    reg_results['EnsembleRVFL'] = mse
    print(f"  EnsembleRVFL:           {mse:.2f}")
except Exception as e:
    print(f"  EnsembleRVFL: ERROR - {e}")

# NOTA: LRF-ELM/LRF-RVFL NO se prueban for regression
# According to literature (Huang et al., 2015), LRF-ELM was specifically designed
# for IMAGE CLASSIFICATION. Uses convolutional filters to extract
# image features, which is not appropriate for tabular regression.
print("  LRF-ELM:                SKIP (image classification only - literature)")
print("  LRF-RVFL:               SKIP (image classification only - literature)")


# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN DE RESULTADOS")
print("=" * 80)

print("\n" + "-" * 60)
print("CLASSIFICATION - Sorted by Accuracy (higher is better)")
print("-" * 60)
sorted_clf = sorted(clf_results.items(), key=lambda x: x[1], reverse=True)
print(f"\n{'Rank':<6} {'Modelo':<25} {'Accuracy':<10}")
print("-" * 45)
for i, (model, acc) in enumerate(sorted_clf, 1):
    print(f"{i:<6} {model:<25} {acc:.4f}")
print(f"\n  Total models: {len(clf_results)}")
print(f"  Mejor: {sorted_clf[0][0]} ({sorted_clf[0][1]:.4f})")
print(f"  Promedio: {np.mean(list(clf_results.values())):.4f}")

print("\n" + "-" * 60)
print("REGRESSION - Sorted by MSE (lower is better)")
print("-" * 60)
sorted_reg = sorted(reg_results.items(), key=lambda x: x[1])
print(f"\n{'Rank':<6} {'Modelo':<25} {'MSE':<15}")
print("-" * 45)
for i, (model, mse) in enumerate(sorted_reg, 1):
    print(f"{i:<6} {model:<25} {mse:.2f}")
print(f"\n  Total models: {len(reg_results)}")
print(f"  Mejor: {sorted_reg[0][0]} (MSE={sorted_reg[0][1]:.2f})")

# Comparison table
print("\n" + "-" * 60)
print("COMPARISON TABLE: CLASSIFICATION vs REGRESSION")
print("-" * 60)

all_models = sorted(set(list(clf_results.keys()) + list(reg_results.keys())))
print(f"\n{'Modelo':<25} {'Clasif.':<12} {'Regression':<12} {'Estado':<10}")
print("-" * 60)

working_both = 0
working_clf_only = 0
working_reg_only = 0

for model in all_models:
    clf_val = f"{clf_results.get(model, 0):.4f}" if model in clf_results else "ERROR"
    reg_val = f"{reg_results.get(model, 0):.2f}" if model in reg_results else "ERROR"

    if model in clf_results and model in reg_results:
        status = "OK"
        working_both += 1
    elif model in clf_results:
        status = "Solo Clf"
        working_clf_only += 1
    else:
        status = "Solo Reg"
        working_reg_only += 1

    print(f"{model:<25} {clf_val:<12} {reg_val:<12} {status:<10}")

print("-" * 60)
print(f"\nResumen:")
print(f"  - Funcionan en AMBAS tareas: {working_both}")
print(f"  - Classification only: {working_clf_only}")
print(f"  - Regression only: {working_reg_only}")
print(f"  - Total unique models: {len(all_models)}")

print("\n" + "=" * 80)
print("Benchmark completado!")
print("=" * 80)
