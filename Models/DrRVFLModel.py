import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin

from Models.DeepELMModel import apply_denoising
from Resources.ActivationFunction import ActivationFunction
from Resources.get_layers import get_layers


class DrRVFLModel(BaseEstimator, ClassifierMixin):
    """
    Deep Representation Random Vector Functional Link (DrRVFL) Model.

    This class implements a Deep Representation RVFL model for classification tasks.
    It uses multiple RVFL layers with weighted input connections for deep feature learning.

    The key difference from DrELM:
    - DrELM: Uses ELMLayer with hidden layer only
    - DrRVFL: Uses RVFLLayer with direct input connections

    Parameters:
    -----------
    activation (str): Activation function for the hidden layers. Default is 'tanh'.
    act_params (dict): Parameters for the activation function.
    alpha (float): Weight factor for the weighted input connections. Default is 10.0.
    layers (list): List of layers to be added to the model.
    w2_weights (list): List of weights matrices for the second layer.
    verbose (int): Verbosity mode. Default is 0.

    Example:
    -----------
    >>> model = DrRVFLModel(activation='mish')
    >>> model.add(RVFLLayer(number_neurons=num_neurons, activation='identity'))
    >>> model.add(RVFLLayer(number_neurons=num_neurons, activation='identity'))
    >>> model.add(RVFLLayer(number_neurons=num_neurons, activation='identity'))
    >>> model.fit(X, y)
    >>> predictions = model.predict(X_test)
    """
    def __init__(self, activation='tanh', act_params=None, alpha=10.0, layers=None, w2_weights=None, verbose=0):
        self.classes_ = None
        self.alpha = alpha
        self.act_params = act_params
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        if w2_weights is None:
            self.w2_weights = []
        else:
            self.w2_weights = w2_weights

        self.verbose = verbose
        if act_params is None:
            act = ActivationFunction(1.0)
        elif "act_param" in act_params and "act_param2" in act_params:
            act = ActivationFunction(act_param=act_params["act_param"], act_param2=act_params["act_param2"])
        elif "act_param" in act_params:
            act = ActivationFunction(act_param=act_params["act_param"])
        elif "knots" in act_params:
            act = ActivationFunction(knots=act_params["knots"])
        else:
            raise Exception("TypeError: Wrong specified activation function parameters")
        self.activation = activation
        self.act = eval("act." + activation)

    def add(self, layer):
        """
        Add a layer to the model.

        Parameters:
        -----------
        layer (object): The layer to be added (should be RVFLLayer or compatible).
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        x (array-like): The input data.
        y (array-like): The target labels.
        """
        if len(y.shape) == 1:
            y = to_categorical(y)

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc='DrRVFL')

        for layer in self.layers:
            layer.build(x.shape)

        if self.verbose == 1:
            pbar.set_description(' DrRVFL : Modules training step ')

        i = 1
        X_i = x
        m, d = x.shape
        _, c = y.shape
        for layer in self.layers[:-1]:
            if layer.denoising is None:
                layer.fit(X_i, y)
            else:
                x_noised = apply_denoising(X_i, layer.denoising, layer.denoising_param)
                layer.fit(x_noised, y)
            O_i = layer.output

            W_initializer = tf.random_uniform_initializer(-1, 1)
            W2 = tf.Variable(
                W_initializer(shape=(c, d)),
                dtype=tf.float32,
                trainable=False
            )
            self.w2_weights.append(W2)
            out = x + self.alpha * tf.matmul(O_i, W2)
            X_i = self.act(out)

            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1

        self.layers[-1].fit(X_i, y)

        if self.verbose == 1:
            pbar.update(n=i+1)
            pbar.close()

    def predict(self, x):
        """
        Predict classes for input data.

        Parameters:
        -----------
        x (array-like): The input data.

        Returns:
        -----------
        array: Predicted class labels.
        """
        X_i = x
        for i in range(0, len(self.layers) - 1):
            O_i = self.layers[i].predict(X_i)
            out = x + self.alpha * tf.matmul(O_i, self.w2_weights[i])
            X_i = self.act(out)
        O_i = self.layers[-1].predict(X_i)
        return tf.argmax(O_i, axis=1).numpy()

    def predict_proba(self, x):
        """
        Predict class probabilities for input data.

        Parameters:
        -----------
        x (array-like): The input data.

        Returns:
        -----------
        array: Predicted class probabilities.
        """
        X_i = x
        for i in range(0, len(self.layers) - 1):
            O_i = self.layers[i].predict_proba(X_i)
            out = x + self.alpha * tf.matmul(O_i, self.w2_weights[i])
            X_i = out * tf.math.tanh(tf.math.softplus(out))
        O_i = self.layers[-1].predict_proba(X_i)
        return tf.keras.activations.softmax(O_i).numpy()

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        total = 0
        trainable = 0
        non_trainable = 0
        i = 0
        prev = None
        print("_________________________________________________________________")
        print(" Layer (type)                Output Shape              Param #   ")
        print("=================================================================")
        for layer in self.layers:
            if layer.__class__ is not prev.__class__:
                i = 0
            if layer.output is None:
                sh = "Unknown"
            else:
                sh = layer.output.shape
            print(f"{layer}_{i} ({layer.__class__.__name__})         {sh}                  {layer.count_params()['all']}      ")
            total = total + layer.count_params()['all']
            trainable = trainable + layer.count_params()['trainable']
            non_trainable = non_trainable + layer.count_params()['non_trainable']
            i = i + 1
            prev = layer
        print("=================================================================")
        print(f"Total params: {total}")
        print(f"Trainable params: {trainable}")
        print(f"Non-trainable params: {non_trainable}")
        print("_________________________________________________________________")

    def to_dict(self):
        """
        Convert the model to a dictionary.
        """
        attributes = {
            'verbose': self.verbose,
            'w2_weights': self.w2_weights,
            'alpha': self.alpha,
            'activation': self.activation
        }
        for i, layer in enumerate(self.layers):
            key_prefix = f'layer.{i}.'
            for key, value in layer.to_dict().items():
                k = key_prefix + key
                attributes.update({k: value})
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

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

                v = attributes.pop('verbose')
                w2 = attributes.pop('w2_weights')
                a = attributes.pop('alpha')
                act = attributes.pop('activation')

                model = cls(verbose=v, w2_weights=w2, alpha=a, activation=act)

                layers = get_layers(attributes)
                model.layers = layers
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None
