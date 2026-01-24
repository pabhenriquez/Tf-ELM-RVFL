import numpy as np

from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class SSRVFLLayer:
    """
        Semi-Supervised Random Vector Functional Link (SS-RVFL) Layer.

        This layer implements Semi-Supervised learning with RVFL, utilizing both labeled
        and unlabeled data through Laplacian graph regularization, while maintaining
        direct input connections characteristic of RVFL.

        Parameters:
        -----------
        - number_neurons (int): Number of neurons in the hidden layer.
        - activation (str): Activation function. Defaults to 'tanh'.
        - act_params (dict or None): Parameters for the activation function.
        - C (float): Regularization parameter. Defaults to 1.0.
        - beta_optimizer (ELMOptimizer or None): Optimizer for beta parameters.
        - is_orthogonalized (bool): Whether to orthogonalize weights. Defaults to False.
        - lam (float): Trade-off parameter for Laplacian regularization. Defaults to 0.5.
        - include_bias (bool): Whether to include bias term. Defaults to True.

        Example:
        -----------
        >>> layer = SSRVFLLayer(number_neurons=1000, lam=0.001)
        >>> model = SSRVFLModel(layer)
        >>> model.fit(X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    """
    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 lam=0.5,
                 include_bias=True,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "ssrvfl"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.lam = lam
        self.act_params = act_params
        self.beta_optimizer = beta_optimizer
        self.is_orthogonalized = is_orthogonalized
        self.include_bias = include_bias
        self.n_features = None

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
        self.number_neurons = number_neurons
        self.C = C

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "bias" in params:
            self.bias = params.pop("bias")
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
        Builds the layer with the given input shape.

        Parameters:
        -----------
        - input_shape (tuple): Shape of the input data.
        """
        self.n_features = input_shape[-1]
        alpha_initializer = tf.random_uniform_initializer(-1, 1)
        self.alpha = tf.Variable(
            alpha_initializer(shape=(self.n_features, self.number_neurons)),
            dtype=tf.float32,
            trainable=False
        )
        bias_initializer = tf.random_uniform_initializer(0, 1)
        self.bias = tf.Variable(
            bias_initializer(shape=(self.number_neurons,)),
            dtype=tf.float32,
            trainable=False
        )
        if self.is_orthogonalized:
            self.alpha = gram_schmidt(self.alpha)
            self.bias = self.bias / tf.norm(self.bias)

    def _compute_feature_matrix(self, x):
        """
        Compute the RVFL feature matrix D = [H, X, 1].
        """
        n_sample = int(x.shape[0])

        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        # RVFL: Concatenate with original input
        D = tf.concat([H, x], axis=1)

        if self.include_bias:
            ones = tf.ones([n_sample, 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        return D

    def fit(self, x_labeled, x_unlabeled, y_labeled, y_unlabeled):
        """
        Fits the layer to labeled and unlabeled data.

        Parameters:
        -----------
        - x_labeled (numpy.ndarray): Labeled input features.
        - x_unlabeled (numpy.ndarray): Unlabeled input features.
        - y_labeled (numpy.ndarray): Labeled target labels.
        - y_unlabeled (numpy.ndarray): Unlabeled target labels (used for shape).
        """
        y_un_zero = np.zeros(shape=np.shape(y_unlabeled))
        n_labeled = np.shape(x_labeled)[0]
        X_combined = np.vstack([x_labeled, x_unlabeled])
        Y_combined = np.vstack([y_labeled, y_un_zero])

        X_combined = tf.cast(X_combined, dtype=tf.float32)
        Y_combined = tf.cast(Y_combined, dtype=tf.float32)

        self.input = X_combined
        n_sample = int(X_combined.shape[0])

        # Compute Laplacian Graph
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

        # Compute sample weights
        ni = self.C / tf.reduce_sum(Y_combined, axis=0)
        C_mat = tf.linalg.diag(tf.reduce_sum(tf.multiply(Y_combined, ni), axis=1))
        m, n = tf.shape(C_mat)[0], tf.shape(C_mat)[1]
        mask = tf.math.logical_and(tf.range(m) > n_labeled, tf.range(n) > n_labeled)
        C_mat = tf.where(mask, tf.zeros_like(C_mat), C_mat)

        # Compute RVFL feature matrix
        D = self._compute_feature_matrix(X_combined)
        d_cols = int(D.shape[1])

        # Compute beta with Laplacian regularization
        if n_sample > d_cols:
            DtCD = tf.matmul(tf.matmul(D, C_mat, transpose_a=True), D)
            lDtLD = self.lam * tf.matmul(tf.matmul(D, L, transpose_a=True), D)
            I = tf.eye(d_cols, dtype=tf.float32)
            inv = tf.linalg.inv(I + DtCD + lDtLD)
            beta = tf.matmul(tf.matmul(tf.matmul(inv, D, transpose_b=True), C_mat), Y_combined)
        else:
            CDDt = tf.matmul(tf.matmul(C_mat, D), D, transpose_b=True)
            lLDDt = self.lam * tf.matmul(tf.matmul(L, D), D, transpose_b=True)
            I = tf.eye(n_sample, dtype=tf.float32)
            inv = tf.linalg.inv(I + CDDt + lLDDt)
            beta = tf.matmul(tf.matmul(tf.matmul(D, inv, transpose_a=True), C_mat), Y_combined)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, D, Y_combined)
        else:
            self.beta = beta

        self.feature_map = D
        self.output = tf.matmul(D, self.beta)

    def predict(self, x):
        """
        Predicts the output for the given input data.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.

        Returns:
        -----------
        tf.Tensor: Predicted output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        D = self._compute_feature_matrix(x)
        output = tf.matmul(D, self.beta)
        return output

    def predict_proba(self, x):
        """
        Predicts class probabilities.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        """
        Calculates the output of the layer.
        """
        x = tf.cast(x, dtype=tf.float32)
        out = self.activation(tf.matmul(x, self.beta, transpose_b=True))
        self.output = out
        return out

    def __str__(self):
        return f"{self.name}, neurons: {self.number_neurons}"

    def count_params(self):
        """
        Counts the number of parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]
        if self.alpha is None or self.bias is None:
            non_trainable = 0
        else:
            non_trainable = self.alpha.shape[0] * self.alpha.shape[1] + self.bias.shape[0]
        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
        Converts the layer's attributes to a dictionary.
        """
        attributes = {
            'name': 'SSRVFLLayer',
            'number_neurons': self.number_neurons,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            'include_bias': self.include_bias,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param,
            "lam": self.lam
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Loads an instance of the layer.
        """
        return cls(**attributes)
