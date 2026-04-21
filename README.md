# TfELM-RVFL

A comprehensive TensorFlow-based framework implementing **Extreme Learning Machines (ELM)** and **Random Vector Functional Link (RVFL)** networks with all their variants.

## Overview

TfELM-RVFL extends the original TfELM framework by adding RVFL counterparts for each ELM variant. RVFL networks include direct connections from input to output, often improving generalization performance.

### Key Features

- **27 model implementations** (13 ELM + 14 RVFL variants)
- **22 layer implementations** with consistent API
- **108 activation functions** supported
- **Full scikit-learn compatibility** (cross_val_score, GridSearchCV, etc.)
- **Multiple learning paradigms**: supervised, semi-supervised, unsupervised, online

## Installation

Install from [PyPI](https://pypi.org/project/tfelm-rvfl/):

```bash
pip install tfelm-rvfl
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/pabhenriquez/Tf-ELM-RVFL.git
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/pabhenriquez/Tf-ELM-RVFL.git
cd Tf-ELM-RVFL
pip install -e .
```

## Quick Start

### Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Layers.RVFLLayer import RVFLLayer
from Models.RVFLModel import RVFLModel

# Load and prepare data
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train model
layer = RVFLLayer(number_neurons=100, activation='sigmoid', C=0.001)
model = RVFLModel(layer, task='classification')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Layers.ELMLayer import ELMLayer
from Models.ELMModel import ELMModel

# Load and prepare data
X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train model
layer = ELMLayer(number_neurons=100, activation='sigmoid', C=0.001)
model = ELMModel(layer, task='regression')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
r2_score = model.score(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")
```

### Clustering (Unsupervised)

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from Layers.USRVFLLayer import USRVFLLayer
from Models.USRVFLModel import USRVFLModel

# Load and prepare data
X, y_true = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)

# Create unsupervised model for clustering
layer = USRVFLLayer(number_neurons=500, embedding_size=10, lam=0.1, activation='sigmoid')
model = USRVFLModel(layer, task='clustering', n_clusters=3)
model.fit(X)

# Get cluster assignments
labels = model.predict(X)

# Evaluate clustering
ari = adjusted_rand_score(y_true, labels)
sil = silhouette_score(X, labels)
print(f"ARI: {ari:.4f}, Silhouette: {sil:.4f}")
```

### Embedding (Dimensionality Reduction)

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from Layers.USELMLayer import USELMLayer
from Models.USELMModel import USELMModel

# Load and prepare data
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

# Create unsupervised model for embedding
layer = USELMLayer(number_neurons=500, embedding_size=2, lam=0.1, activation='sigmoid')
model = USELMModel(layer, task='embedding')
model.fit(X)

# Get 2D embeddings
embeddings = model.predict(X)
print(f"Embedding shape: {embeddings.shape}")  # (150, 2)
```

### Cross-Validation with scikit-learn

```python
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from Layers.RVFLLayer import RVFLLayer
from Models.RVFLModel import RVFLModel

# Load data
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

# Create model
layer = RVFLLayer(number_neurons=100, activation='sigmoid', C=0.001)
model = RVFLModel(layer, task='classification')

# Cross-validation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## ELM vs RVFL

| Aspect | ELM | RVFL |
|--------|-----|------|
| **Output** | `y = H × β` | `y = [H, X] × β` |
| **Direct links** | No | Yes |
| **Preserves linear info** | No | Yes |

## Available Models

### Supervised Learning
| ELM | RVFL | Description |
|-----|------|-------------|
| ELMModel | RVFLModel | Basic single hidden layer |
| KELMModel | KRVFLModel | Kernel-based |
| DeepELMModel | DeepRVFLModel | Deep stacked architecture |
| ML_ELMModel | ML_RVFLModel | Multi-layer with autoencoders |
| RCELMModel | RCRVFLModel | Residual compensation |
| OSELMModel | OSRVFLModel | Online sequential learning |
| SSELMModel | SSRVFLModel | Semi-supervised |
| SSKELMModel | SSKRVFLModel | Semi-supervised kernel |
| - | EnsembleDeepRVFLModel | Ensemble with voting |

### Unsupervised Learning
| ELM | RVFL | Description |
|-----|------|-------------|
| USELMModel | USRVFLModel | Embedding and clustering |
| USKELMModel | USKRVFLModel | Kernel-based unsupervised |

## Project Structure

```
TfELM-RVFL/
├── Data/           # 20+ UCI benchmark datasets
├── Examples/       # 5 annotated code examples
├── Layers/         # 22 layer implementations
├── Models/         # 27 model implementations
├── Optimizers/     # 6 optimizer implementations
├── Resources/      # 20 supporting functions
└── requirements.txt
```

## Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `number_neurons` | Hidden layer size | 100-1000 |
| `activation` | Activation function | 'sigmoid', 'relu', 'tanh' |
| `C` | Regularization (supervised) | 0.001-10 |
| `lam` | Regularization (unsupervised) | 0.001-10 |

## License

MIT License
