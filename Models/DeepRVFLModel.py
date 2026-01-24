import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.DeepRVFLLayer import DeepRVFLLayer


class DeepRVFLModel(BaseEstimator, ClassifierMixin):
    """
    Deep Random Vector Functional Link (Deep RVFL) Model for SUPERVISED learning.

    Task Type: CLASSIFICATION or REGRESSION

    This class implements a Deep RVFL model with multiple hidden layers and direct
    connections from input to output.

    Parameters:
    -----------
    layer : DeepRVFLLayer
        The Deep RVFL layer of the model.
    task : str, default='classification'
        The type of task: 'classification' or 'regression'.
    random_weights : bool, default=True
        Indicates whether to randomly initialize the weights.
    classification : bool, optional
        DEPRECATED. Use task parameter instead.

    Attributes:
    -----------
    classes_ : array-like, shape (n_classes,)
        The unique class labels.

    Examples:
    -----------
    >>> layer = DeepRVFLLayer(number_neurons=100, n_layers=3, activation='relu')
    >>> model = DeepRVFLModel(layer)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    def __init__(self, layer, task='classification', random_weights=True, classification=None):
        self.classes_ = None
        # Backward compatibility: if classification is passed, convert to task
        if classification is not None:
            self.task = 'classification' if classification else 'regression'
        else:
            self.task = task
        self.classification = self.task == 'classification'  # For backward compatibility
        self.layer = layer
        self.random_weights = random_weights

    def fit(self, X, y):
        """
        Fit the Deep RVFL model to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input training data.
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The target values.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        if self.random_weights:
            self.layer.build(X.shape)

        if self.task == 'classification':
            self.classes_ = unique_labels(y)
            if len(np.shape(y)) == 1:
                y = to_categorical(y)
        else:
            # Regression: ensure y is 2D
            self.classes_ = None
            y = np.array(y)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        self.layer.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels or regression values.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted values.
        """
        pred = self.layer.predict(X)
        if self.task == 'classification':
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            # Regression: return raw predictions
            result = pred.numpy()
            if result.shape[1] == 1:
                return result.flatten()
            return result

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_proba : array-like, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return self.layer.predict_proba(x)

    def save(self, file_path):
        """
        Save the model to an HDF5 file.
        """
        try:
            with h5py.File(file_path, 'w') as h5file:
                for key, value in self.to_dict().items():
                    if value is None:
                        value = 'None'
                    elif isinstance(value, list):
                        # Handle lists of tensors
                        for i, v in enumerate(value):
                            if hasattr(v, 'numpy'):
                                h5file.create_dataset(f"{key}_{i}", data=v.numpy())
                            else:
                                h5file.create_dataset(f"{key}_{i}", data=v)
                        continue
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
                if "random_weights" in attributes:
                    r = attributes.pop("random_weights")

                layer = DeepRVFLLayer.load(attributes)
                model = cls(layer, c, r)
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
        attributes["classification"] = self.classification  # For backward compatibility
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    def summary(self):
        """
        Print a summary of the model.
        """
        print("=" * 60)
        print("Deep RVFL Model Summary")
        print("=" * 60)
        print(f"Number of hidden layers: {self.layer.n_layers}")
        print(f"Neurons per layer: {self.layer.number_neurons}")
        print(f"Activation: {self.layer.activation_name}")
        print(f"Regularization (C): {self.layer.C}")
        print(f"Classification: {self.classification}")
        params = self.layer.count_params()
        print(f"Total parameters: {params['all']}")
        print(f"  - Trainable: {params['trainable']}")
        print(f"  - Non-trainable: {params['non_trainable']}")
        print("=" * 60)
