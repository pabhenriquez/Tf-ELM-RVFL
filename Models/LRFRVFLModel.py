import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from Resources.generate_random_filters import generate_random_filters
from Resources.sqrt_pooling import sqrt_pooling


class LRFRVFLModel(BaseEstimator, ClassifierMixin):
    """
    A Local Receptive Field Random Vector Functional Link (LRF-RVFL) model.

    Task Type: IMAGE CLASSIFICATION only

    NOT FOR REGRESSION OR NON-IMAGE DATA.
    This model is designed specifically for image classification tasks.
    It requires 4D input data in the format (N, Height, Width, Channels).

    According to the literature, LRF-ELM/LRF-RVFL was designed for IMAGE CLASSIFICATION tasks,
    combining random convolutional filters with RVFL for efficient image recognition.

    Reference:
    - Huang et al. "Local Receptive Fields Based Extreme Learning Machine"
      IEEE Computational Intelligence Magazine, 2015. DOI: 10.1109/MCI.2015.2405316

    This model combines random CNN filters with a Random Vector Functional Link (RVFL)
    for classification tasks. The RVFL maintains direct input connections.

    The key difference from LRFELM:
    - LRFELM: Uses ELMModel with hidden layer only
    - LRFRVFL: Uses RVFLModel with direct input connections

    Parameters:
    -----------
    rvfl_model: The RVFL model to be used for training and prediction (RVFLModel).
    num_feature_maps (int): Number of feature maps in the random CNN filters.
    filter_size (int): Size of the filters used in the random CNN.
    num_input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    pool_size (int): Size of the pooling window for sqrt pooling.
    classification (bool): Whether the task is classification (default is True).
    random_weights (bool): Whether to use random weights for the CNN filters.

    Example:
    -----------
    >>> layer = RVFLLayer(number_neurons=5000, C=10)
    >>> rvfl_model = RVFLModel(layer)
    >>> model = LRFRVFLModel(rvfl_model=rvfl_model)
    >>> model.fit(X_train, y_train)  # X_train should be (N, H, W, C) shape
    >>> pred = model.predict(X_test)
    """
    def __init__(self, rvfl_model, num_feature_maps=48, filter_size=4, num_input_channels=1, pool_size=3,
                 classification=True, random_weights=True, **args):
        self.classes_ = None
        self.classification = classification
        self.num_feature_maps = num_feature_maps
        self.filter_size = filter_size
        self.num_input_channels = num_input_channels
        self.pool_size = pool_size
        self.rvfl_model = rvfl_model
        self.random_weights = random_weights
        if "kernels" in args:
            self.kernels = args["kernels"]
        else:
            self.kernels = None

    def fit(self, X, y):
        """
        Fit the LRF-RVFL model to the training data.

        Parameters:
        -----------
        X (array-like): The input data (should be 4D: N x H x W x C).
        y (array-like): The target labels.
        """
        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            y = to_categorical(y)

        N = X.shape[0]
        X = tf.cast(X, dtype=tf.float32)
        self.kernels = generate_random_filters(self.filter_size, self.num_feature_maps, self.num_input_channels)
        conv_output = tf.nn.conv2d(X, self.kernels, strides=[1, 1, 1, 1], padding='VALID')
        pooled_output = sqrt_pooling(conv_output, self.pool_size)
        flattened_output = tf.reshape(pooled_output, [N, -1])
        self.rvfl_model.fit(flattened_output, y)

    def predict(self, X):
        """
        Predict the labels for the input data.

        Parameters:
        -----------
        X (array-like): The input data (should be 4D: N x H x W x C).

        Returns:
        -----------
        array: Predicted labels.
        """
        N = X.shape[0]
        X = tf.cast(X, dtype=tf.float32)
        conv_output = tf.nn.conv2d(X, self.kernels, strides=[1, 1, 1, 1], padding='VALID')
        pooled_output = sqrt_pooling(conv_output, self.pool_size)
        flattened_output = tf.reshape(pooled_output, [N, -1])
        return self.rvfl_model.predict(flattened_output)

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        -----------
        X (array-like): The input data (should be 4D: N x H x W x C).

        Returns:
        -----------
        array: Predicted class probabilities.
        """
        N = X.shape[0]
        X = tf.cast(X, dtype=tf.float32)
        conv_output = tf.nn.conv2d(X, self.kernels, strides=[1, 1, 1, 1], padding='VALID')
        pooled_output = sqrt_pooling(conv_output, self.pool_size)
        flattened_output = tf.reshape(pooled_output, [N, -1])
        return self.rvfl_model.predict_proba(flattened_output)

    def save(self, file_path):
        """
        Save the model to an HDF5 file.
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
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                if "classification" in attributes:
                    classification = attributes.pop("classification")
                if "model_name" in attributes:
                    m_name = attributes.pop("model_name")
                if "random_weights" in attributes:
                    random_weights = attributes.pop("random_weights")
                if "kernels" in attributes:
                    kernels = attributes.pop("kernels")
                if "num_feature_maps" in attributes:
                    num_feature_maps = attributes.pop("num_feature_maps")
                if "filter_size" in attributes:
                    filter_size = attributes.pop("filter_size")
                if "num_input_channels" in attributes:
                    num_input_channels = attributes.pop("num_input_channels")
                if "pool_size" in attributes:
                    pool_size = attributes.pop("pool_size")
                if "classes_" in attributes:
                    c = attributes.pop("classes_")

                model = eval(f"{m_name}.load(file_path)")
                m = cls(rvfl_model=model, num_feature_maps=num_feature_maps, filter_size=filter_size,
                        num_input_channels=num_input_channels, pool_size=pool_size, classification=classification,
                        random_weights=random_weights, kernels=kernels)
                m.classes_ = c
                return m
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None

    def to_dict(self):
        """
        Serialize the current instance into a dictionary.
        """
        attributes = self.rvfl_model.to_dict()
        attributes["model_name"] = self.rvfl_model.__class__.__name__
        attributes["classification"] = self.classification
        attributes["random_weights"] = self.random_weights
        attributes["kernels"] = self.kernels
        attributes["num_feature_maps"] = self.num_feature_maps
        attributes["filter_size"] = self.filter_size
        attributes["num_input_channels"] = self.num_input_channels
        attributes["pool_size"] = self.pool_size
        attributes["classes_"] = self.classes_

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes
