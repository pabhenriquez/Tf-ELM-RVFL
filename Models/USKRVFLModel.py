import h5py
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from Layers.USKRVFLLayer import USKRVFLLayer
from Resources.Kernel import Kernel


class USKRVFLModel(BaseEstimator, TransformerMixin):
    """
    Unsupervised Kernel Random Vector Functional Link (USKRVFL) Model.

    This model implements unsupervised learning (dimensionality reduction) using
    kernel methods with direct input connections (RVFL characteristic).

    Parameters:
    -----------
    layer : USKRVFLLayer
        The USKRVFL layer to use.

    Example:
    -----------
    >>> from Resources.Kernel import Kernel
    >>> kernel = Kernel("rbf", param=1.0)
    >>> layer = USKRVFLLayer(kernel=kernel, embedding_size=10, lam=0.001, include_direct_link=True)
    >>> model = USKRVFLModel(layer)
    >>> model.fit(X_train)
    >>> embeddings, clusters = model.predict(X_test, clustering=True, k=2)
    """
    def __init__(self, layer):
        self.layer = layer

    def fit(self, x, y=None):
        """
        Fit the USKRVFL model to the input data.

        Parameters:
        -----------
        x : array-like
            Input data.
        y : array-like, optional
            Ignored (for sklearn compatibility).
        """
        x = tf.cast(x, dtype=tf.float32)
        self.layer.build(x.shape)
        self.layer.fit(x)
        return self

    def transform(self, x):
        """
        Transform the input data to the embedded space.

        Parameters:
        -----------
        x : array-like
            Input data.

        Returns:
        -----------
        array-like
            Transformed data in the embedded space.
        """
        x = tf.cast(x, dtype=tf.float32)
        return self.layer.predict(x)

    def fit_transform(self, x, y=None):
        """
        Fit and transform the input data.

        Parameters:
        -----------
        x : array-like
            Input data.
        y : array-like, optional
            Ignored (for sklearn compatibility).

        Returns:
        -----------
        array-like
            Transformed data in the embedded space.
        """
        self.fit(x)
        return self.transform(x)

    def predict(self, x, clustering=False, k=None):
        """
        Predict embeddings and optionally perform clustering.

        Parameters:
        -----------
        x : array-like
            Input data.
        clustering : bool
            Whether to perform clustering. Defaults to False.
        k : int
            Number of clusters if clustering is True.

        Returns:
        -----------
        If clustering is False:
            array-like: Embeddings.
        If clustering is True:
            tuple: (embeddings, cluster_labels)
        """
        x = tf.cast(x, dtype=tf.float32)
        return self.layer.predict(x, clustering=clustering, k=k)

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
        except Exception as e:
            print(f"Error saving to HDF5: {e}")

    @classmethod
    def load(cls, file_path):
        """
        Load a model from an HDF5 file.
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                attributes = {key: h5file[key][()] for key in h5file.keys()}
                for key, value in attributes.items():
                    if isinstance(value, bytes):
                        attributes[key] = value.decode('utf-8')

                layer = USKRVFLLayer.load(attributes)
                model = cls(layer=layer)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None

    def to_dict(self):
        """
        Convert the model to a dictionary.
        """
        return self.layer.to_dict()
