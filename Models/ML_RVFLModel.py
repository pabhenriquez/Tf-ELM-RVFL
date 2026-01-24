import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import numpy as np

from Layers.GRVFL_AE_Layer import GRVFL_AE_Layer
from Layers.RVFLLayer import RVFLLayer
from Layers.KRVFLLayer import KRVFLLayer
from Resources.apply_denoising import apply_denoising
from Resources.get_layers import get_layers


class ML_RVFLModel(BaseEstimator, ClassifierMixin):
    """
    Multilayer Random Vector Functional Link (ML-RVFL) Model.

    This model consists of multiple RVFL layers for feature extraction followed
    by a final RVFL layer for classification/regression. Each layer maintains
    direct input connections characteristic of RVFL.

    Parameters:
    -----------
    classification : bool
        Whether the task is classification. Defaults to True.
    layers : list
        List of RVFL layers.
    verbose : int
        Verbosity level. Defaults to 0.

    Example:
    -----------
    >>> model = ML_RVFLModel(verbose=1)
    >>> model.add(GRVFL_AE_Layer(number_neurons=50))
    >>> model.add(GRVFL_AE_Layer(number_neurons=60))
    >>> model.add(RVFLLayer(number_neurons=1000))
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_test)
    """
    def __init__(self, classification=True, layers=None, verbose=0):
        self.classes_ = None
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.classification = classification
        self.verbose = verbose

    def add(self, layer):
        """
        Add an RVFL layer to the model.

        Parameters:
        -----------
        layer : RVFLLayer or compatible layer
            The layer to add.
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
        Fit the ML-RVFL model.

        The model is trained layer by layer:
        1. Feature extraction layers (autoencoder style)
        2. Final classification/regression layer

        Parameters:
        -----------
        x : array-like
            Training input data.
        y : array-like
            Target values.
        """
        if self.classification:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0]

        if len(np.shape(y)) == 1:
            if self.classification:
                y_encoded = to_categorical(y)
            else:
                y_encoded = np.reshape(y, (-1, 1))
        else:
            y_encoded = y

        y_encoded = tf.cast(y_encoded, dtype=tf.float32)

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc='ML-RVFL: Training layers')

        x = tf.cast(x, dtype=tf.float32)
        feature_map = x

        # Train each layer
        for i, layer in enumerate(self.layers):
            layer.build(feature_map.shape)

            if i < len(self.layers) - 1:
                # Feature extraction layers (autoencoder style)
                if hasattr(layer, 'denoising') and layer.denoising is not None:
                    feature_map_noised = apply_denoising(feature_map, layer.denoising, layer.denoising_param)
                    layer.fit(feature_map_noised, feature_map)
                else:
                    layer.fit(feature_map, feature_map)

                # Get hidden representation for next layer
                if hasattr(layer, 'transform'):
                    feature_map = layer.transform(feature_map)
                else:
                    feature_map = layer.output
            else:
                # Final classification/regression layer
                layer.fit(feature_map, y_encoded)

            if self.verbose == 1:
                pbar.update(1)

        if self.verbose == 1:
            pbar.close()

        return self

    def predict(self, x):
        """
        Predict class labels or regression values.

        Parameters:
        -----------
        x : array-like
            Input data.

        Returns:
        --------
        array-like
            Predictions.
        """
        x = tf.cast(x, dtype=tf.float32)

        feature_map = x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                # Feature extraction
                if hasattr(layer, 'transform'):
                    feature_map = layer.transform(feature_map)
                else:
                    feature_map = layer.predict(feature_map)
            else:
                # Final prediction
                pred = layer.predict(feature_map)

        if self.classification:
            return tf.math.argmax(pred, axis=1).numpy()
        else:
            return pred.numpy()

    def predict_proba(self, x):
        """
        Predict class probabilities.
        """
        x = tf.cast(x, dtype=tf.float32)

        feature_map = x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                if hasattr(layer, 'transform'):
                    feature_map = layer.transform(feature_map)
                else:
                    feature_map = layer.predict(feature_map)
            else:
                pred = layer.predict(feature_map)

        return tf.keras.activations.softmax(pred).numpy()

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        total = 0
        trainable = 0
        non_trainable = 0
        print("_" * 65)
        print("ML-RVFL Model Summary")
        print("=" * 65)
        print(f" {'Layer':<20} {'Type':<20} {'Params':>10}")
        print("-" * 65)

        for i, layer in enumerate(self.layers):
            params = layer.count_params()
            layer_type = layer.__class__.__name__
            print(f" {i}_{layer.name:<17} {layer_type:<20} {params['all']:>10}")
            total += params['all']
            trainable += params['trainable']
            non_trainable += params['non_trainable']

        print("=" * 65)
        print(f"Total params: {total}")
        print(f"Trainable params: {trainable}")
        print(f"Non-trainable params: {non_trainable}")
        print("_" * 65)

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

                c = attributes.pop('classification')
                v = attributes.pop('verbose')
                cl = attributes.pop('classes')

                model = cls(classification=c, verbose=v)
                layers = get_layers(attributes)
                model.layers = layers
                model.classes_ = cl
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
            'verbose': self.verbose,
            'classes': self.classes_
        }
        for i, layer in enumerate(self.layers):
            key_prefix = f'layer.{i}.'
            for key, value in layer.to_dict().items():
                k = key_prefix + key
                attributes.update({k: value})
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes
