import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.RVFLLayer import RVFLLayer


class RVFLModel(BaseEstimator, ClassifierMixin):
    """
    Random Vector Functional Link (RVFL) Model for SUPERVISED learning.

    Task Type: CLASSIFICATION or REGRESSION

    This class implements an RVFL model, which is a single-hidden-layer feedforward neural
    network with direct connections from input to output (direct link). The model can be used
    for both classification and regression tasks.

    Parameters:
    -----------
    layer : RVFLLayer
        The RVFL layer of the model.
    task : str, default='classification'
        The type of task: 'classification' or 'regression'.
    random_weights : bool, default=True
        Indicates whether to randomly initialize the weights.

    Attributes:
    -----------
    classes_ : array-like, shape (n_classes,)
        The unique class labels in the training data (only for classification).

    Examples:
    -----------
    Classification example:

    >>> layer = RVFLLayer(number_neurons=1000, activation='mish')
    >>> model = RVFLModel(layer, task='classification')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> accuracy = accuracy_score(y_test, predictions)

    Regression example:

    >>> layer = RVFLLayer(number_neurons=1000, activation='mish')
    >>> model = RVFLModel(layer, task='regression')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> mse = mean_squared_error(y_test, predictions)
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
        Fit the RVFL model to the training data.

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
            # Convert to one-hot encoding for classification
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
            The predicted class labels or regression values.
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
            The predicted class probabilities.
        """
        return self.layer.predict_proba(x)

    def save(self, file_path):
        """
        Save the model to an HDF5 file.

        Parameters:
        -----------
        file_path : str
            The path to the HDF5 file.
        """
        try:
            with h5py.File(file_path, 'w') as h5file:
                for key, value in self.to_dict().items():
                    if value is None:
                        value = 'None'
                    h5file.create_dataset(key, data=value)
                h5file.close()
        except Exception as e:
            print(f"Error saving to HDF5: {e}")

    @classmethod
    def load(cls, file_path: str):
        """
        Load a model from an HDF5 file.

        Parameters:
        -----------
        file_path : str
            The path to the HDF5 file.

        Returns:
        --------
        model : RVFLModel
            The loaded model.
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

                layer = eval(f"{l_type}(**attributes)")
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
        attributes["classification"] = self.classification
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes
