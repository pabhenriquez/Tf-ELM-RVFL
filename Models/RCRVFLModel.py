import h5py
import tensorflow as tf
import tqdm
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.multiclass import unique_labels

from Resources.apply_denoising import apply_denoising
from Resources.get_layers import get_layers


class RCRVFLModel(BaseEstimator, RegressorMixin):
    """
    Residual Compensation Random Vector Functional Link (RC-RVFL) model.

    Task Type: REGRESSION (primary), CLASSIFICATION (secondary)

    According to the literature, RC-ELM/RC-RVFL was originally designed for REGRESSION
    tasks where the prediction error is compensated layer by layer.

    Reference:
    - Chen et al. "Residual compensation extreme learning machine for regression"
      Neurocomputing, 2018. DOI: 10.1016/j.neucom.2018.05.052

    This model consists of a series of layers, where the first layer acts as a baseline model
    (e.g., an RVFL), and subsequent layers are used to compensate for the residuals of the
    previous layers. Each layer maintains direct input connections (RVFL characteristic).

    The key difference from RCELM:
    - RCELM: Uses ELMLayer with hidden layer only
    - RCRVFL: Uses RVFLLayer with direct input connections

    Parameters:
    -----------
    layers (list): List of layers to be added to the model. Default is None.
    verbose (int): Verbosity mode. 0 = silent, 1 = verbose. Default is 0.

    Example:
    -----------
    >>> model = RCRVFLModel()
    >>> model.add(RVFLLayer(number_neurons=1000, activation='sigmoid', C=10))
    >>> model.add(RVFLLayer(number_neurons=2000, activation='sigmoid', C=10))
    >>> model.add(RVFLLayer(number_neurons=1000, activation='sigmoid', C=10))
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X_test)
    """
    def __init__(self, layers=None, verbose=0, task='regression'):
        self.lambdas = None
        self.errors = None
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.verbose = verbose
        self.task = task  # 'regression' (primary) or 'classification'

    def add(self, layer):
        """
        Add a layer to the model.

        Parameters:
        -----------
        layer: Layer to be added to the model (RVFLLayer or compatible).
        """
        self.layers.append(layer)

    def fit(self, x, y):
        """
        Fit the RC-RVFL model to the given input-output pairs.

        Parameters:
        -----------
        x (numpy.ndarray): Input data tensor.
        y (numpy.ndarray): Output labels.
        """
        import numpy as np
        y = np.array(y)

        # Only convert to categorical for classification tasks
        if self.task == 'classification' and len(y.shape) == 1:
            from keras.utils import to_categorical
            y = to_categorical(y)
        elif self.task == 'regression' and len(y.shape) == 1:
            # Keep y as 2D for consistent processing
            y = y.reshape(-1, 1)

        if self.verbose == 1:
            pbar = tqdm.tqdm(total=len(self.layers), desc=' RC-RVFL : Baseline layer step ')

        for layer in self.layers:
            layer.build(x.shape)

        rvfl_baseline = self.layers[0]
        rvfl_baseline.fit(x, y)
        y_hat = rvfl_baseline.predict(x)
        e_k = y - y_hat

        if self.verbose == 1:
            pbar.set_description(' RC-RVFL : Compensation layer step ')

        self.errors = []
        self.lambdas = []

        i = 1
        for layer in self.layers[1:]:
            rvfl_k = layer
            if layer.denoising is None:
                rvfl_k.fit(x, e_k)
            else:
                x_noised = apply_denoising(x, layer.denoising, layer.denoising_param)
                rvfl_k.fit(x_noised, e_k)
            e_k1_hat = rvfl_k.predict(x)
            e_k = e_k - e_k1_hat
            self.errors.append(e_k)
            # Lambda is inverse of mean squared error (scalar per layer)
            mse = float(tf.reduce_mean(e_k ** 2).numpy())
            self.lambdas.append(1.0 / (mse + 1e-10))

            if self.verbose == 1:
                pbar.update(n=i)
                i = i + 1

        # Normalize lambdas to sum to 1
        total_lambda = sum(self.lambdas)
        self.lambdas = [l / total_lambda for l in self.lambdas]
        self.lambdas = tf.constant(self.lambdas, dtype=tf.float32)
        self.errors = tf.stack(self.errors, axis=0)

        if self.verbose == 1:
            pbar.update(n=i+1)
            pbar.close()

    def predict(self, x):
        """
        Predict the output for the given input data.

        Parameters:
        -----------
        x (numpy.ndarray): Input data tensor.

        Returns:
        -----------
        numpy.ndarray: Predicted output tensor.
        """
        rvfl_baseline = self.layers[0]
        y_hat = rvfl_baseline.predict(x)

        preds = []
        for layer in self.layers[1:]:
            y_pred = layer.predict(x)
            preds.append(y_pred)

        preds = tf.stack(preds, axis=0)

        # Handle both 1D (regression) and 2D (classification) outputs
        if len(preds.shape) == 3:
            # 2D output: [num_layers, samples, features]
            lambdas_reshaped = tf.reshape(self.lambdas, [-1, 1, 1])
        else:
            # 1D output: [num_layers, samples]
            lambdas_reshaped = tf.reshape(self.lambdas, [-1, 1])

        product = lambdas_reshaped * preds
        stacked_errors = tf.reduce_sum(product, axis=0)

        y_pred = y_hat + stacked_errors
        result = y_pred.numpy()

        # For regression, flatten if output is 1D
        if self.task == 'regression' and len(result.shape) > 1 and result.shape[1] == 1:
            result = result.flatten()

        return result

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
        Convert the model into a dictionary representation.
        """
        attributes = {
            'verbose': self.verbose,
            'lambdas': self.lambdas,
            'errors': self.errors
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
                l = attributes.pop('lambdas')
                e = attributes.pop('errors')

                model = cls(verbose=v)
                model.lambdas = l
                model.errors = e

                layers = get_layers(attributes)
                model.layers = layers
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None
