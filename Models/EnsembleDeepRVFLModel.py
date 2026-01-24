import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from Layers.EnsembleDeepRVFLLayer import EnsembleDeepRVFLLayer


class EnsembleDeepRVFLModel(BaseEstimator, ClassifierMixin):
    """
    Ensemble Deep Random Vector Functional Link (Ensemble Deep RVFL) Model.

    This model combines multiple RVFL layers in an ensemble fashion. Each layer
    trains its own output weights, and predictions are combined through voting
    or averaging.

    Parameters:
    -----------
    layer : EnsembleDeepRVFLLayer
        The Ensemble Deep RVFL layer.
    classification : bool, default=True
        Indicates whether the model is for classification (True) or regression (False).
    random_weights : bool, default=True
        Indicates whether to randomly initialize the weights.
    ensemble_method : str, default='vote'
        Method for combining predictions: 'vote', 'addition', or 'mean'.

    Attributes:
    -----------
    classes_ : array-like, shape (n_classes,)
        The unique class labels.

    Examples:
    -----------
    >>> layer = EnsembleDeepRVFLLayer(number_neurons=100, n_layers=5, activation='relu')
    >>> model = EnsembleDeepRVFLModel(layer)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    def __init__(self, layer, classification=True, random_weights=True, ensemble_method='vote'):
        self.classes_ = None
        self.classification = classification
        self.layer = layer
        self.random_weights = random_weights
        self.ensemble_method = ensemble_method

    def fit(self, X, y):
        """
        Fit the Ensemble Deep RVFL model to the training data.

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
        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            if self.classification:
                y = to_categorical(y)
            else:
                y = np.reshape(y, (-1, 1))

        self.layer.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels or regression values.

        For classification, uses the specified ensemble_method ('vote' or 'addition').
        For regression, uses mean averaging.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted values.
        """
        if self.classification:
            if self.ensemble_method == 'vote':
                return self.layer.predict_vote(X).numpy()
            else:  # 'addition'
                result, _ = self.layer.predict_addition(X)
                return result.numpy()
        else:
            return self.layer.predict_mean(X).numpy()

    def predict_all(self, X):
        """
        Returns both voting and addition predictions for classification.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        vote_pred : array-like
            Predictions using majority voting.
        add_pred : tuple (result, probabilities)
            Predictions using addition method.
        """
        if not self.classification:
            raise ValueError("predict_all is only available for classification tasks")

        vote_pred = self.layer.predict_vote(X)
        add_result, add_proba = self.layer.predict_addition(X)

        return vote_pred.numpy(), (add_result.numpy(), add_proba.numpy())

    def predict_proba(self, x):
        """
        Predict class probabilities using the addition method.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_proba : array-like, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return self.layer.predict_proba(x).numpy()

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
                if "ensemble_method" in attributes:
                    em = attributes.pop("ensemble_method")
                else:
                    em = 'vote'

                layer = EnsembleDeepRVFLLayer.load(attributes)
                model = cls(layer, c, r, em)
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
        attributes["ensemble_method"] = self.ensemble_method

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    def summary(self):
        """
        Print a summary of the model.
        """
        print("=" * 60)
        print("Ensemble Deep RVFL Model Summary")
        print("=" * 60)
        print(f"Number of ensemble members (layers): {self.layer.n_layers}")
        print(f"Neurons per layer: {self.layer.number_neurons}")
        print(f"Activation: {self.layer.activation_name}")
        print(f"Regularization (C): {self.layer.C}")
        print(f"Ensemble method: {self.ensemble_method}")
        print(f"Classification: {self.classification}")
        params = self.layer.count_params()
        print(f"Total parameters: {params['all']}")
        print(f"  - Trainable: {params['trainable']}")
        print(f"  - Non-trainable: {params['non_trainable']}")
        print("=" * 60)

    def evaluate(self, X, y):
        """
        Evaluate the model on test data.

        For classification, returns accuracy for both voting and addition methods.
        For regression, returns mean absolute error.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The test data.
        y : array-like, shape (n_samples,)
            The true labels.

        Returns:
        --------
        For classification:
            dict with 'vote_accuracy' and 'addition_accuracy'
        For regression:
            float: mean absolute error
        """
        if self.classification:
            vote_pred = self.layer.predict_vote(X).numpy()
            add_pred, _ = self.layer.predict_addition(X)
            add_pred = add_pred.numpy()

            vote_acc = np.mean(vote_pred == y)
            add_acc = np.mean(add_pred == y)

            return {
                'vote_accuracy': vote_acc,
                'addition_accuracy': add_acc
            }
        else:
            pred = self.layer.predict_mean(X).numpy()
            mae = np.mean(np.abs(pred.flatten() - y))
            return mae
