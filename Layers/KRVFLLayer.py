import numpy as np
from sklearn.cluster import KMeans

from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel
from Resources.kernel_distances import calculate_pairwise_distances_vector, calculate_pairwise_distances


def proceed_kernel(attributes):
    if "kernel" in attributes:
        k_n = attributes.pop("kernel")
        k_p = attributes.pop("kernel_param")
        k_t = attributes.pop("kernel_type")
        if k_t == "Kernel":
            k = Kernel(kernel_name=k_n, param=k_p)
        elif k_t == "CombinedSumKernel":
            kernels = []
            for n, p in zip(k_n, k_p):
                kernels.append(Kernel(kernel_name=n.decode('utf-8'), param=p))
            k = CombinedSumKernel(kernels)
        else:
            kernels = []
            for n, p in zip(k_n, k_p):
                kernels.append(Kernel(kernel_name=n.decode('utf-8'), param=p))
            k = CombinedProductKernel(kernels)
    else:
        k = Kernel()
    return k


class KRVFLLayer:
    """
        Kernel Random Vector Functional Link (KRVFL) Layer.

        This class implements a Kernel RVFL layer that uses kernel functions to map
        the input data to a higher-dimensional space, combined with direct input links.

        The key difference from KELM is the inclusion of the original input in the
        feature matrix, following the RVFL paradigm.

        Parameters:
        -----------
            kernel (Kernel): Instance of a kernel function.
            activation (str, optional): Name of the activation function. Defaults to 'tanh'.
            act_params (dict, optional): Parameters for the activation function.
            C (float, optional): Regularization parameter. Defaults to 1.0.
            include_direct_link (bool, optional): Whether to include direct input link.
                Defaults to True.
            nystrom_approximation (bool, optional): Whether to use Nystrom approximation.
                Defaults to False.
            landmark_selection_method (str, optional): Method for landmark selection.
                Defaults to 'random'.

        Attributes:
        -----------
            K (tensor): Kernel matrix.
            beta (tensor): Weights of the layer.
            input (tensor): Input data.
            output (tensor): Output data.

        Examples:
        -----------
        >>> kernel = Kernel("rbf", param=1.0)
        >>> layer = KRVFLLayer(kernel, activation='mish')
        >>> model = KRVFLModel(layer)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X_test)
    """
    def __init__(self, kernel: Kernel, activation='tanh', act_params=None, C=1.0,
                 include_direct_link=True, nystrom_approximation=False,
                 landmark_selection_method='random', random_pct=0.1, **params):
        self.K = None
        self.error_history = None
        self.feature_map = None
        self.name = "krvfl"
        self.beta = None
        self.input = None
        self.output = None
        self.include_direct_link = include_direct_link
        self.nystrom_approximation = nystrom_approximation
        self.landmark_selection_method = landmark_selection_method
        self.random_pct = random_pct
        self.C = C

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

        self.activation = eval("act." + activation)
        self.activation_name = activation

        if "beta" in params:
            self.beta = params.pop("beta")
        if "input" in params:
            self.input = params.pop("input")
        if "K" in params:
            self.K = params.pop("K")

        if "denoising" in params:
            self.denoising = params.pop("denoising")
        else:
            self.denoising = None

        if "denoising_param" in params:
            self.denoising_param = params.pop("denoising_param")
        else:
            self.denoising_param = None

        if "kernel_param" in params:
            params.update({'kernel': kernel})
            self.kernel = proceed_kernel(params)
        else:
            self.kernel = kernel

    def build(self, input_shape):
        """
        Build the layer with the given input shape.

        Parameters:
        -----------
            input_shape (tuple): Shape of the input data.
        """
        observations = input_shape[0]
        self.K = tf.Variable(
            tf.zeros(shape=(observations, observations)),
            dtype=tf.float32,
            trainable=False
        )

    def fit(self, x, y):
        """
        Fit the KRVFL layer to the input-output pairs.

        The feature matrix includes both the kernel matrix and the original input
        (direct link): D = [K, X, 1]

        Parameters:
        -----------
            x (tensor): Input data.
            y (tensor): Target values.
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x

        n_samples = int(x.shape[0])
        n_landmark = int(self.random_pct * n_samples)

        if self.nystrom_approximation:
            L = random_sampling(x, n_landmark)
            C = calculate_pairwise_distances_vector(x, L, self.kernel.ev)
            W = calculate_pairwise_distances(L, self.kernel.ev)
            diagonal = tf.linalg.diag_part(W)
            diagonal_with_small_value = diagonal + 0.00001
            W = tf.linalg.set_diag(W, diagonal_with_small_value)
            K = tf.matmul(tf.matmul(C, tf.linalg.inv(W)), C, transpose_b=True)
        else:
            K = calculate_pairwise_distances(x, self.kernel.ev)

        diagonal = tf.linalg.diag_part(K)
        diagonal_with_small_value = diagonal + 0.1
        K = tf.linalg.set_diag(K, diagonal_with_small_value)

        # KRVFL: Include direct link from input
        if self.include_direct_link:
            D = tf.concat([K, x], axis=1)
        else:
            D = K

        # Add bias column
        ones = tf.ones([n_samples, 1], dtype=tf.float32)
        D = tf.concat([D, ones], axis=1)

        # Compute beta using regularized least squares
        d_cols = int(D.shape[1])

        if n_samples > d_cols:
            DTD = tf.matmul(D, D, transpose_a=True)
            reg_matrix = self.C * tf.eye(d_cols, dtype=tf.float32)
            self.beta = tf.matmul(
                tf.matmul(tf.linalg.inv(reg_matrix + DTD), D, transpose_b=True),
                y
            )
        else:
            DDT = tf.matmul(D, D, transpose_b=True)
            reg_matrix = self.C * tf.eye(n_samples, dtype=tf.float32)
            self.beta = tf.matmul(
                tf.matmul(D, tf.linalg.inv(reg_matrix + DDT), transpose_a=True),
                y
            )

        self.K = K
        self.feature_map = D

    def predict(self, x):
        """
        Predict the output for the input data.

        Parameters:
        -----------
            x (tensor): Input data.

        Returns:
        -----------
            tensor: Predicted output.
        """
        x = tf.cast(x, dtype=tf.float32)
        n_samples = int(x.shape[0])

        # Compute kernel matrix between x and training data
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)

        # Include direct link if specified
        if self.include_direct_link:
            D = tf.concat([k, x], axis=1)
        else:
            D = k

        # Add bias column
        ones = tf.ones([n_samples, 1], dtype=tf.float32)
        D = tf.concat([D, ones], axis=1)

        output = tf.matmul(D, self.beta)
        self.output = output
        return output

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Parameters:
        -----------
            x (tensor): Input data.

        Returns:
        -----------
            numpy.ndarray: Predicted class probabilities.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred).numpy()

    def __str__(self):
        return f"{self.name}, kernel: {self.kernel.kernel_name}"

    def count_params(self):
        """
        Count the number of parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]
        non_trainable = 0
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
        Convert the layer attributes to a dictionary.
        """
        attributes = {
            'name': 'KRVFLLayer',
            'C': self.C,
            "beta": self.beta,
            "kernel": self.kernel.kernel_name,
            "kernel_param": self.kernel.kernel_param,
            "kernel_type": self.kernel.__class__.__name__,
            "include_direct_link": self.include_direct_link,
            "nystrom_approximation": self.nystrom_approximation,
            "landmark_selection_method": self.landmark_selection_method,
            "input": self.input,
            "K": self.K,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load the layer from a dictionary of attributes.
        """
        k = proceed_kernel(attributes)
        attributes.update({"kernel": k})
        layer = cls(**attributes)
        return layer


# Random Sampling
def random_sampling(data, n_samples):
    num_rows = tf.shape(data)[0]
    selected_indices = tf.random.shuffle(tf.range(num_rows))[:n_samples]
    sampled_data = tf.gather(data, selected_indices)
    return sampled_data
