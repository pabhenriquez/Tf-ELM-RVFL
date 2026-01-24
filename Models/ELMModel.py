import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.ELMLayer import ELMLayer
from Layers.WELMLayer import WELMLayer
from Layers.KELMLayer import KELMLayer
from Layers.SubELMLayer import SubELMLayer


class ELMModel(BaseEstimator, ClassifierMixin):
    """
    Extreme Learning Machine model for SUPERVISED learning.

    Task Type: CLASSIFICATION or REGRESSION

    This class implements an Extreme Learning Machine (ELM) model, which is a single-hidden-layer feedforward neural
    network. The model can be used for both classification and regression tasks.

    Parameters:
    -----------
    layer : ELMLayer
        The hidden layer of the ELM model.
    task : str, default='classification'
        The type of task: 'classification' or 'regression'.
    random_weights : bool, default=True
        Indicates whether to randomly initialize the weights of the hidden layer.

    Attributes:
    -----------
    classes_ : array-like, shape (n_classes,)
        The unique class labels in the training data (only for classification).

    Examples:
    -----------
    Classification example:

    >>> layer = ELMLayer(number_neurons=1000, activation='mish')
    >>> model = ELMModel(layer, task='classification')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> accuracy = accuracy_score(y_test, predictions)

    Regression example:

    >>> layer = ELMLayer(number_neurons=1000, activation='mish')
    >>> model = ELMModel(layer, task='regression')
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
            Fit the ELM model to the training data.

            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
                The input training data.
            y : array-like, shape (n_samples,) or (n_samples, n_classes)
                The target values for classification or regression tasks.

            Returns:
            --------
            self : object
                Returns the instance itself.

            Example:
            -----------
            Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

            >>> elm = ELMLayer(number_neurons=1000, activation='mish')

            Create an ELM model using the trained ELM layer

            >>> model = ELMModel(elm)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)
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

    def predict(self, X):
        """
            Predict class labels or regression values for the input data.

            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
                The input data.

            Returns:
            --------
            y_pred : array-like, shape (n_samples,)
                The predicted class labels or regression values.

            Example:
            -----------
            Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

            >>> elm = ELMLayer(number_neurons=1000, activation='mish')

            Create an ELM model using the trained ELM layer

            >>> model = ELMModel(elm)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> acc = accuracy_score(model.predict(X), y)
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
            Predict class probabilities for the input data.

            Parameters:
            -----------
            X : array-like, shape (n_samples, n_features)
                The input data.

            Returns:
            --------
            y_proba : array-like, shape (n_samples, n_classes)
                The predicted class probabilities.

            Example:
            -----------
            Initialize an Extreme Learning Machine (ELM) layer with 1000 neurons

            >>> elm = ELMLayer(number_neurons=1000, activation='mish')

            Create an ELM model using the trained ELM layer

            >>> model = ELMModel(elm)

            Fit the ELM model to the entire dataset

            >>> model.fit(X, y)

            Evaluate the accuracy of the model on the training data

            >>> pred_proba = model.predict_proba(X)
        """
        return self.layer.predict_proba(x)

    def save(self, file_path):
        """
            Save the model to an HDF5 file.

            Parameters:
            -----------
            file_path : str
                The path to the HDF5 file where the model will be saved.

            Example:
            -----------
            Save the trained model to a file

            >>> model.save("Saved Models/ELM_Model.h5")
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
                The path to the HDF5 file containing the model.

            Returns:
            --------
            model : ELMModel
                The loaded ELMModel instance.

            Example:
            -----------
            Load the saved model from the file

            >>> model = ELMModel.load("Saved Models/ELM_Model.h5")
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Extract attributes from the HDF5 file
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
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Convert the model to a dictionary of attributes.

            Returns:
            --------
            attributes : dict
                A dictionary containing the attributes of the model.
        """
        attributes = self.layer.to_dict()
        attributes["task"] = self.task
        attributes["classification"] = self.classification  # For backward compatibility
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes


