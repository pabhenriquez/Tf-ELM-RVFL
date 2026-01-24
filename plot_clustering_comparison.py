"""
Clustering Comparison: US-ELM vs US-RVFL
Generates 2D embedding visualization
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from Models.USELMModel import USELMModel
from Models.USRVFLModel import USRVFLModel
from Layers.USELMLayer import USELMLayer
from Layers.USRVFLLayer import USRVFLLayer

# Load Wine dataset
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Color map for classes
colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green
class_names = ['Class 0', 'Class 1', 'Class 2']

# 1. Ground Truth (PCA for visualization)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ax = axes[0]
for i, (color, name) in enumerate(zip(colors, class_names)):
    mask = y == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
               alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_title('(a) Ground Truth (PCA) - Wine', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# 2. US-ELM Embedding (500 neurons, sigmoid, lam=0.1)
layer_elm = USELMLayer(number_neurons=500, embedding_size=2, activation='sigmoid', lam=0.1)
model_elm = USELMModel(layer_elm, task='embedding')
model_elm.fit(X_scaled)
X_elm = model_elm.predict(X_scaled)

ax = axes[1]
for i, (color, name) in enumerate(zip(colors, class_names)):
    mask = y == i
    ax.scatter(X_elm[mask, 0], X_elm[mask, 1], c=color, label=name,
               alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel('Embedding 1')
ax.set_ylabel('Embedding 2')
ax.set_title('(b) US-ELM Embedding', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# 3. US-RVFL Embedding (500 neurons, sigmoid, lam=0.1)
layer_rvfl = USRVFLLayer(number_neurons=500, embedding_size=2, activation='sigmoid', lam=0.1)
model_rvfl = USRVFLModel(layer_rvfl, task='embedding')
model_rvfl.fit(X_scaled)
X_rvfl = model_rvfl.predict(X_scaled)

ax = axes[2]
for i, (color, name) in enumerate(zip(colors, class_names)):
    mask = y == i
    ax.scatter(X_rvfl[mask, 0], X_rvfl[mask, 1], c=color, label=name,
               alpha=0.7, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel('Embedding 1')
ax.set_ylabel('Embedding 2')
ax.set_title('(c) US-RVFL Embedding', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/clustering_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/clustering_comparison.png', dpi=300, bbox_inches='tight')
print("Figures saved to paper/figures/clustering_comparison.pdf and .png")

plt.show()
