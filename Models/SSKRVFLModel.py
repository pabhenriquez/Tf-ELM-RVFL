import h5py
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from Layers.SSKRVFLLayer import SSKRVFLLayer
from Resources.Kernel import Kernel


class SSKRVFLModel(BaseEstimator, ClassifierMixin):
    """
    Semi-Supervised Kernel Random Vector Functional Link (SSKRVFL) Model.

    This model implements semi-supervised learning using kernel methods with
    direct input connections (RVFL characteristic). It can leverage both labeled
    and unlabeled data for improved generalization.

    Parameters:
    -----------
    layer : SSKRVFLLayer
        The SSKRVFL layer to use.
    classification : bool
        Whether the task is classification. Defaults to True.

    Example:
    -----------
    >>> from Resources.Kernel import Kernel
    >>> kernel = Kernel("rbf", param=1.0)
    >>> layer = SSKRVFLLayer(kernel=kernel, lam=0.001, include_direct_link=True)
    >>> model = SSKRVFLModel(layer, classification=True)
    >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    >>> predictions = model.predict(X_test)
    """
    def __init__(self, layer, classification=True):
        self.layer = layer
        self.classification = classification
        self.classes_ = None

    def fit(self, x_labeled, x_unlabeled, y_labeled, y_unlabeled):
        """
        Fit the SSKRVFL model using labeled and unlabeled data.

        Parameters:
        -----------
        x_labeled : array-like
            Labeled input data.
        x_unlabeled : array-like
            Unlabeled input data.
        y_labeled : array-like
            Labels for the labeled data.
        y_unlabeled : array-like
            Placeholder for unlabeled data (used for shape).
        """
        if self.classification:
            self.classes_ = unique_labels(y_labeled)
        else:
            self.classes_ = [0]

        if len(np.shape(y_labeled)) == 1:
            if self.classification:
                y_labeled = to_categorical(y_labeled)
            else:
                y_labeled = np.reshape(y_labeled, (-1, 1))

        if len(np.shape(y_unlabeled)) == 1:
            if self.classification:
                y_unlabeled = to_categorical(y_unlabeled, num_classes=len(self.classes_))
            else:
                y_unlabeled = np.reshape(y_unlabeled, (-1, 1))

        x_combined = np.vstack([x_labeled, x_unlabeled])
        self.layer.build(x_combined.shape)
        self.layer.fit(x_labeled, x_unlabeled, y_labeled, y_unlabeled)
        return self

    def predict(self, x):
        """
        Predict class labels for the input data.

        Parameters:
        -----------
        x : array-like
            Input data.

        Returns:
        -----------
        array-like
            Predicted class labels.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.layer.predict(x)
        if self.classification:
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            return pred.numpy()

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Parameters:
        -----------
        x : array-like
            Input data.

        Returns:
        -----------
        array-like
            Predicted class probabilities.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.layer.predict(x)
        return tf.keras.activations.softmax(pred).numpy()

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

                classification = attributes.pop('classification')
                classes = attributes.pop('classes_')

                layer = SSKRVFLLayer.load(attributes)
                model = cls(layer=layer, classification=classification)
                model.classes_ = classes
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None

    def to_dict(self):
        """
        Convert the model to a dictionary.
        """
        attributes = {
            'classification': self.classification,
            'classes_': self.classes_
        }
        attributes.update(self.layer.to_dict())
        return {k: v for k, v in attributes.items() if v is not None}
