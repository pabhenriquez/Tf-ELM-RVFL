import h5py
import numpy as np
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.OSRVFLLayer import OSRVFLLayer


class OSRVFLModel(BaseEstimator, ClassifierMixin):
    """
    Online Sequential Random Vector Functional Link (OS-RVFL) Model.

    This model implements online/incremental learning with RVFL, allowing
    for continuous model updates as new data arrives.

    Parameters:
    -----------
    layer : OSRVFLLayer
        The OS-RVFL layer.
    prefetch_size : int
        Size of the initial batch for initialization. Defaults to 100.
    batch_size : int
        Size of sequential batches. Defaults to 32.
    verbose : int
        Verbosity level. Defaults to 0.
    classification : bool
        Whether the task is classification. Defaults to True.

    Example:
    -----------
    >>> layer = OSRVFLLayer(1000, 'tanh')
    >>> model = OSRVFLModel(layer, prefetch_size=120, batch_size=64)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_test)
    """

    def __init__(self, layer: OSRVFLLayer, prefetch_size=100, batch_size=32, verbose=0, classification=True):
        self.classes_ = None
        self.classification = classification
        self.layer = layer
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the OS-RVFL model using online sequential learning.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input training data.
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The target values.
        """
        self.layer.build(X.shape)

        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            if self.classification:
                y = to_categorical(y)
            else:
                y = np.reshape(y, (-1, 1))

        # Initial batch for initialization
        X_init = X[:self.prefetch_size]
        y_init = y[:self.prefetch_size]
        self.layer.fit_initialize(X_init, y_init)

        # Sequential batches
        remaining_samples = X.shape[0] - self.prefetch_size
        n_batches = remaining_samples // self.batch_size

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=n_batches, desc='OS-RVFL: Sequential learning')

        for i in range(n_batches):
            start_idx = self.prefetch_size + i * self.batch_size
            end_idx = start_idx + self.batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            self.layer.fit_seq(X_batch, y_batch)

            if self.verbose == 1:
                pbar.update(1)

        # Handle remaining samples
        if remaining_samples % self.batch_size != 0:
            start_idx = self.prefetch_size + n_batches * self.batch_size
            X_batch = X[start_idx:]
            y_batch = y[start_idx:]
            if X_batch.shape[0] > 0:
                self.layer.fit_seq(X_batch, y_batch)

        if self.verbose == 1:
            pbar.close()

        return self

    def partial_fit(self, X, y):
        """
        Incrementally update the model with new data.

        Parameters:
        -----------
        X : array-like
            New input samples.
        y : array-like
            New target values.
        """
        if len(np.shape(y)) == 1:
            if self.classification:
                y = to_categorical(y, num_classes=len(self.classes_))
            else:
                y = np.reshape(y, (-1, 1))

        self.layer.fit_seq(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels or regression values.
        """
        pred = self.layer.predict(X)
        if self.classification:
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            return pred.numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        return self.layer.predict_proba(X).numpy()

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

                if "classification" in attributes:
                    c = attributes.pop("classification")
                if "name" in attributes:
                    l_type = attributes.pop("name")

                layer = OSRVFLLayer.load(attributes)
                model = cls(layer, classification=c)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None

    def to_dict(self):
        """
        Convert the model to a dictionary.
        """
        attributes = self.layer.to_dict()
        attributes["classification"] = self.classification
        attributes["prefetch_size"] = self.prefetch_size
        attributes["batch_size"] = self.batch_size

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes
