import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.SSRVFLLayer import SSRVFLLayer


class SSRVFLModel(BaseEstimator, ClassifierMixin):
    """
    Semi-Supervised Random Vector Functional Link (SS-RVFL) Model.

    This model uses both labeled and unlabeled data for training through
    Laplacian graph regularization, combined with RVFL's direct input links.

    Parameters:
    -----------
    layer : SSRVFLLayer
        The SS-RVFL layer.
    classification : bool
        Whether the task is classification. Defaults to True.

    Example:
    -----------
    >>> layer = SSRVFLLayer(number_neurons=1000, lam=0.001)
    >>> model = SSRVFLModel(layer)
    >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    >>> predictions = model.predict(X_test)
    """

    def __init__(self, layer: SSRVFLLayer, classification=True):
        self.classes_ = None
        self.classification = classification
        self.layer = layer

    def fit(self, X_labeled, X_unlabeled, y_labeled, y_unlabeled):
        """
        Fit the SS-RVFL model using labeled and unlabeled data.

        Parameters:
        -----------
        X_labeled : array-like
            Labeled input samples.
        X_unlabeled : array-like
            Unlabeled input samples.
        y_labeled : array-like
            Labels for labeled samples.
        y_unlabeled : array-like
            Placeholder labels for unlabeled samples (used for shape).
        """
        # Build layer with combined data shape
        combined_shape = (X_labeled.shape[0] + X_unlabeled.shape[0], X_labeled.shape[1])
        self.layer.build(combined_shape)

        if self.classification:
            self.classes_ = unique_labels(y_labeled)
        else:
            self.classes_ = [0]

        if len(np.shape(y_labeled)) == 1:
            if self.classification:
                y_labeled = to_categorical(y_labeled)
                y_unlabeled = to_categorical(y_unlabeled, num_classes=y_labeled.shape[1])
            else:
                y_labeled = np.reshape(y_labeled, (-1, 1))
                y_unlabeled = np.reshape(y_unlabeled, (-1, 1))

        self.layer.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
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

                layer = SSRVFLLayer.load(attributes)
                model = cls(layer, c)
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

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes
