import h5py
import numpy as np

from Layers.USKELMLayer import calculate_pairwise_distances
from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.Kernel import CombinedProductKernel, Kernel, CombinedSumKernel
from Resources.gram_schmidt import gram_schmidt
from Resources.kernel_distances import calculate_pairwise_distances_vector


class SSKRVFLLayer:
    """
        Semi-Supervised Kernel Random Vector Functional Link (SSKRVFL) layer.

        This layer implements a semi-supervised version of the Kernel RVFL algorithm. It is
        capable of handling both labeled and unlabeled data for training, utilizing kernel-based
        feature mapping and Laplacian graph regularization with direct input connections.

        The key difference from SSKELM is the inclusion of direct input links (RVFL characteristic):
        - SSKELM: Uses only kernel features
        - SSKRVFL: Uses kernel features + direct input link

        Parameters:
        -----------
        - kernel (Kernel): Kernel function to be used for feature mapping.
        - activation (str): Name of the activation function. Default is 'tanh'.
        - act_params (dict): Parameters for the activation function. Default is None.
        - C (float): Regularization parameter. Default is 1.0.
        - lam (float): Laplacian graph regularization parameter. Default is 0.5.
        - include_direct_link (bool): Whether to include direct input connections. Default is True.
        - nystrom_approximation (bool): Whether to use Nystrom approximation. Default is False.
        - landmark_selection_method (str): Method for landmark selection. Default is 'random'.

        Example:
        -----------
        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
        >>> layer = SSKRVFLLayer(kernel=kernel, lam=0.001, include_direct_link=True)
        >>> model = SSKRVFLModel(layer)
    """
    def __init__(self,
                 kernel,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 lam=0.5,
                 include_direct_link=True,
                 nystrom_approximation=False,
                 landmark_selection_method='random',
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "sskrvfl"
        self.beta = None
        self.input = None
        self.output = None
        self.lam = lam
        self.act_params = act_params
        self.include_direct_link = include_direct_link
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
        self.activation_name = activation
        self.activation = eval("act." + activation)
        self.C = C
        self.kernel = kernel
        self.nystrom_approximation = nystrom_approximation
        self.landmark_selection_method = landmark_selection_method

        if "K" in params:
            self.K = params.pop("K")
        if "input" in params:
            self.input = params.pop("input")
        if "beta" in params:
            self.beta = params.pop("beta")

        if "denoising" in params:
            self.denoising = params.pop("denoising")
        else:
            self.denoising = None

        if "denoising_param" in params:
            self.denoising_param = params.pop("denoising_param")
        else:
            self.denoising_param = None

    def build(self, input_shape):
        """
        Build the layer by initializing necessary variables.

        Parameters:
        -----------
        - input_shape (tuple): Shape of the input data.
        """
        observations = input_shape[0]
        self.K = tf.Variable(
            tf.zeros(shape=(observations, observations)),
            dtype=tf.float32,
            trainable=False
        )

    def _compute_feature_matrix(self, K, X):
        """
        Compute the RVFL feature matrix D = [K, X] (kernel features + direct link).

        Parameters:
        -----------
        - K (tf.Tensor): Kernel matrix.
        - X (tf.Tensor): Original input data.

        Returns:
        -----------
        - D (tf.Tensor): Feature matrix with kernel features and direct link.
        """
        if self.include_direct_link:
            D = tf.concat([K, X], axis=1)
        else:
            D = K
        return D

    def fit(self, x_labeled, x_unlabeled, y_labeled, y_unlabeled):
        """
        Train the layer on labeled and unlabeled data.

        Parameters:
        -----------
        - x_labeled (np.ndarray or tf.Tensor): Labeled input data.
        - x_unlabeled (np.ndarray or tf.Tensor): Unlabeled input data.
        - y_labeled (np.ndarray or tf.Tensor): Labeled target data.
        - y_unlabeled (np.ndarray or tf.Tensor): Unlabeled target data.
        """
        y_un_zero = np.zeros(shape=np.shape(y_unlabeled))
        n_labeled = np.shape(x_labeled)[0]
        X_combined = np.vstack([x_labeled, x_unlabeled])
        Y_combined = np.vstack([y_labeled, y_un_zero])

        X_combined = tf.cast(X_combined, dtype=tf.float32)
        Y_combined = tf.cast(Y_combined, dtype=tf.float32)

        self.input = X_combined

        # Laplacian Graph
        squared_norms = tf.reduce_sum(tf.square(X_combined), axis=1, keepdims=True)
        dot_product = tf.matmul(X_combined, X_combined, transpose_b=True)
        distances = squared_norms - 2 * dot_product + tf.transpose(squared_norms)
        distances = tf.maximum(distances, 0.0)
        sigma = 1.0
        W = tf.exp(-distances / (2.0 * sigma ** 2))
        D_diag = tf.linalg.diag(tf.reduce_sum(W, axis=1))
        L_unnormalized = D_diag - W
        D_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(D_diag))
        L = tf.matmul(tf.matmul(D_sqrt_inv, L_unnormalized), D_sqrt_inv)

        ni = self.C / tf.reduce_sum(Y_combined, axis=0)
        C = tf.linalg.diag(tf.reduce_sum(tf.multiply(Y_combined, ni), axis=1))
        m, n = tf.shape(C)[0], tf.shape(C)[1]
        mask = tf.math.logical_and(tf.range(m) > n_labeled, tf.range(n) > n_labeled)
        C = tf.where(mask, tf.zeros_like(C), C)

        if self.nystrom_approximation:
            num_rows = tf.shape(X_combined)[0]
            shuffled_indices = tf.random.shuffle(tf.range(num_rows))
            selected_indices = shuffled_indices[:100]
            L_land = tf.gather(X_combined, selected_indices)
            C_nys = calculate_pairwise_distances_vector(X_combined, L_land, self.kernel.ev)
            W_nys = calculate_pairwise_distances(L_land, self.kernel.ev)
            K = tf.matmul(tf.matmul(C_nys, tf.linalg.inv(W_nys)), C_nys, transpose_b=True)
        else:
            K = calculate_pairwise_distances(X_combined, self.kernel.ev)

        # RVFL: Create feature matrix with direct link
        D = self._compute_feature_matrix(K, X_combined)

        # Compute beta using regularized least squares with Laplacian
        CHHt = tf.matmul(tf.matmul(C, D), D, transpose_b=True)
        lLHHt = self.lam * tf.matmul(tf.matmul(L, D), D, transpose_b=True)
        i = tf.eye(tf.shape(CHHt)[0])
        inv = tf.linalg.inv(i + CHHt + lLHHt)
        beta = tf.matmul(tf.matmul(tf.matmul(D, inv, transpose_a=True), C), Y_combined)

        self.beta = beta
        self.K = K

    def predict(self, x):
        """
        Predict output for the given input data.

        Parameters:
        -----------
        - x (np.ndarray or tf.Tensor): Input data.

        Returns:
        -----------
        tf.Tensor: Predicted output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)

        # RVFL: Create feature matrix with direct link
        D = self._compute_feature_matrix(k, x)

        output = tf.matmul(D, self.beta)
        self.output = output
        return output

    def predict_proba(self, x):
        """
        Predict class probabilities for the given input data.

        Parameters:
        -----------
        - x (np.ndarray or tf.Tensor): Input data.

        Returns:
        -----------
        tf.Tensor: Predicted class probabilities' tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        """
        Calculate the output of the layer for the given input data.
        """
        x = tf.cast(x, dtype=tf.float32)
        k = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)
        D = self._compute_feature_matrix(k, x)
        out = self.activation(tf.matmul(D, self.beta))
        self.output = out
        return out

    def __str__(self):
        return f"{self.name}, kernel: {self.kernel.__class__.__name__}"

    def count_params(self):
        """
        Counts the number of trainable and non-trainable parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]

        non_trainable = 0
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
        Convert the layer's attributes to a dictionary.
        """
        attributes = {
            'name': 'SSKRVFLLayer',
            'C': self.C,
            "beta": self.beta,
            "kernel": self.kernel.kernel_name,
            "kernel_param": self.kernel.kernel_param,
            "kernel_type": self.kernel.__class__.__name__,
            "nystrom_approximation": self.nystrom_approximation,
            "landmark_selection_method": self.landmark_selection_method,
            "include_direct_link": self.include_direct_link,
            "input": self.input,
            "K": self.K,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param,
            "lam": self.lam
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load a layer instance from a dictionary of attributes.
        """
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

        attributes.update({"kernel": k})
        layer = cls(**attributes)
        return layer
