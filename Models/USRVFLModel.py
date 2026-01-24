import h5py
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from Layers.USRVFLLayer import USRVFLLayer


class USRVFLModel(BaseEstimator, TransformerMixin):
    """
    Unsupervised Random Vector Functional Link (US-RVFL) Model for UNSUPERVISED learning.

    Task Type: EMBEDDING (dimensionality reduction) or CLUSTERING

    NOT FOR CLASSIFICATION OR REGRESSION.
    For classification/regression, use RVFLModel instead.

    This model performs unsupervised learning for dimensionality reduction
    and data embedding using RVFL with Laplacian graph regularization.

    Parameters:
    -----------
    layer : USRVFLLayer
        The US-RVFL layer with embedding_size parameter.
    task : str, default='embedding'
        The type of task: 'embedding' or 'clustering'.
    n_clusters : int, default=None
        Number of clusters (required if task='clustering').
    random_weights : bool, default=True
        Whether to initialize the model with random weights.

    Examples:
    -----------
    Dimensionality reduction (embedding):

    >>> layer = USRVFLLayer(number_neurons=500, embedding_size=3, lam=0.001)
    >>> model = USRVFLModel(layer, task='embedding')
    >>> model.fit(X_train)
    >>> embeddings = model.predict(X_test)  # Returns 3D embeddings

    Clustering:

    >>> layer = USRVFLLayer(number_neurons=500, embedding_size=10, lam=0.001)
    >>> model = USRVFLModel(layer, task='clustering', n_clusters=5)
    >>> model.fit(X_train)
    >>> cluster_labels = model.predict(X_test)  # Returns cluster assignments
    """

    def __init__(self, layer: USRVFLLayer, task='embedding', n_clusters=None, random_weights=True):
        self.classes_ = None
        self.task = task
        self.n_clusters = n_clusters
        self.layer = layer
        self.random_weights = random_weights

        if task == 'clustering' and n_clusters is None:
            raise ValueError("n_clusters must be specified when task='clustering'")

    def fit(self, X, y=None):
        """
        Fit the US-RVFL model (unsupervised).

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : Ignored
            Not used, present for API consistency.
        """
        if self.random_weights:
            self.layer.build(X.shape)
        self.classes_ = np.zeros(np.shape(X)[0])
        self.layer.fit(X)
        return self

    def transform(self, X):
        """
        Transform the data into the embedding space.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        embeddings : array-like, shape (n_samples, embedding_size)
            The embedded data.
        """
        return self.layer.predict(X).numpy()

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data.
        """
        self.fit(X)
        return self.transform(X)

    def predict(self, X, clustering=None, k=None):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X : array-like
            Input data for prediction.
        clustering : bool, optional
            Whether to perform clustering. If None, uses self.task setting.
        k : int, optional
            Number of clusters. If None, uses self.n_clusters.

        Returns:
        -----------
        If task='embedding': Returns embeddings array.
        If task='clustering': Returns cluster labels array.
        If clustering=True explicitly: Returns tuple (embeddings, cluster_labels).

        Examples:
        -----------
        >>> model = USRVFLModel(layer, task='embedding')
        >>> embeddings = model.predict(X)

        >>> model = USRVFLModel(layer, task='clustering', n_clusters=5)
        >>> cluster_labels = model.predict(X)
        """
        # Determine clustering mode
        if clustering is None:
            do_clustering = self.task == 'clustering'
        else:
            do_clustering = clustering

        # Determine number of clusters
        n_clusters = k if k is not None else self.n_clusters

        if do_clustering:
            result = self.layer.predict(X, clustering=True, k=n_clusters)
            embeddings, cluster_labels = result
            # If task is clustering, return only labels
            if self.task == 'clustering' and clustering is None:
                return cluster_labels
            # If explicitly requested, return both
            return embeddings, cluster_labels
        else:
            result = self.layer.predict(X, clustering=False, k=None)
            return result.numpy()

    def save(self, file_path):
        """
        Save the model to an HDF5 file.
        """
        try:
            with h5py.File(file_path, 'w') as h5file:
                for key, value in self.to_dict().items():
                    if value is None:
                        value = 'None'
                    elif hasattr(value, 'numpy'):
                        value = value.numpy()
                    h5file.create_dataset(key, data=value)
                h5file.close()
        except Exception as e:
            print(f"Error saving to HDF5: {e}")

    @classmethod
    def load(cls, file_path: str):
        """
        Load a model from an HDF5 file.
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                if "name" in attributes:
                    l_type = attributes.pop("name")

                layer = USRVFLLayer.load(attributes)
                model = cls(layer)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None

    def to_dict(self):
        """
        Convert the model to a dictionary.
        """
        attributes = self.layer.to_dict()
        attributes["task"] = self.task
        attributes["n_clusters"] = self.n_clusters
        attributes["random_weights"] = self.random_weights
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes
