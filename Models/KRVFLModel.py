import h5py
import numpy as np
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
from Layers.KRVFLLayer import KRVFLLayer
from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel


class KRVFLModel(BaseEstimator, ClassifierMixin):
    """
        Kernel Random Vector Functional Link (KRVFL) Model for SUPERVISED learning.

        Task Type: CLASSIFICATION or REGRESSION

        This class implements a Kernel RVFL model that combines kernel methods
        with direct input links for classification or regression tasks.

        Parameters:
        -----------
            krvfl (KRVFLLayer): Instance of a Kernel RVFL layer.
            task (str): The type of task: 'classification' or 'regression'. Default is 'classification'.
            classification (bool, optional): DEPRECATED. Use task parameter instead.

        Attributes:
        -----------
            classes_ (array-like): Unique class labels.
            task (str): The type of task being performed.
            krvfl (KRVFLLayer): Instance of a Kernel RVFL layer.

        Examples:
        -----------
        >>> kernel = Kernel("rbf", param=1.0)
        >>> layer = KRVFLLayer(kernel, activation='mish')
        >>> model = KRVFLModel(layer)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    def __init__(self, krvfl: KRVFLLayer, task='classification', classification=None):
        self.classes_ = None
        # Backward compatibility: if classification is passed, convert to task
        if classification is not None:
            self.task = 'classification' if classification else 'regression'
        else:
            self.task = task
        self.classification = self.task == 'classification'  # For backward compatibility
        self.krvfl = krvfl

    def fit(self, X, y):
        """
        Fit the KRVFL model to training data.

        Parameters:
        -----------
            X (array-like): Training input samples.
            y (array-like): Target values.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        self.krvfl.build(X.shape)

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

        self.krvfl.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels or regression values.

        Parameters:
        -----------
            X (array-like): Input samples.

        Returns:
        -----------
            array-like: Predicted class labels or regression values.
        """
        pred = self.krvfl.predict(X)
        if self.task == 'classification':
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            # Regression: return raw predictions
            result = pred.numpy()
            if result.shape[1] == 1:
                return result.flatten()
            return result

    def to_dict(self):
        """
        Convert the model to a dictionary of attributes.
        """
        attributes = self.krvfl.to_dict()
        attributes["task"] = self.task
        attributes["classification"] = self.classification  # For backward compatibility

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, file_path: str):
        """
        Load a model from an HDF5 file.

        Parameters:
        -----------
            file_path (str): The file path to load from.

        Returns:
        -----------
            KRVFLModel: An instance of the model.
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

                layer = eval(f"{l_type}.load(attributes)")
                model = cls(layer, c)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None

    def save(self, file_path):
        """
        Save the model to an HDF5 file.

        Parameters:
        -----------
            file_path (str): The file path to save to.
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

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        -----------
            X (array-like): Input samples.

        Returns:
        -----------
            array-like: Predicted class probabilities.
        """
        pred = self.krvfl.predict(X)
        return tf.keras.activations.softmax(pred).numpy()

    def summary(self):
        """
        Print a summary of the model.
        """
        print("=" * 60)
        print("Kernel RVFL Model Summary")
        print("=" * 60)
        print(f"Kernel: {self.krvfl.kernel.kernel_name}")
        print(f"Kernel parameter: {self.krvfl.kernel.kernel_param}")
        print(f"Direct link: {self.krvfl.include_direct_link}")
        print(f"Regularization (C): {self.krvfl.C}")
        print(f"Classification: {self.classification}")
        params = self.krvfl.count_params()
        print(f"Total parameters: {params['all']}")
        print("=" * 60)
