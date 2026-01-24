from Optimizers import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf


class OSRVFLLayer:
    """
        Online Sequential Random Vector Functional Link (OS-RVFL) Layer.

        This layer implements Online Sequential RVFL for incremental/online learning.
        It uses the Woodbury matrix identity for efficient sequential updates while
        maintaining direct input connections (RVFL characteristic).

        Parameters:
        -----------
        - number_neurons (int): The number of neurons in the hidden layer.
        - activation (str): Activation function. Defaults to 'tanh'.
        - act_params (dict): Activation function parameters.
        - C (float): Regularization parameter. Defaults to 0.001.
        - beta_optimizer (ELMOptimizer): Optimizer for beta coefficients.
        - is_orthogonalized (bool): Whether to orthogonalize weights.
        - include_bias (bool): Whether to include bias term. Defaults to True.

        Attributes:
        -----------
        - P (tf.Tensor): P matrix used in sequential learning.
        - beta (tf.Tensor): Output weights.
        - alpha (tf.Tensor): Input weights to hidden layer.
        - bias (tf.Tensor): Bias of hidden layer.

        Example:
        -----------
        >>> layer = OSRVFLLayer(1000, 'tanh')
        >>> model = OSRVFLModel(layer, prefetch_size=120, batch_size=64)
        >>> model.fit(X, y)
    """

    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=0.001,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 include_bias=True,
                 **params):
        self.y = None
        self.P = None
        self.error_history = None
        self.feature_map = None
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.name = "OS-RVFL"
        self.is_orthogonalized = is_orthogonalized
        self.include_bias = include_bias
        self.n_features = None  # Store input features for direct link

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

        self.act_params = act_params
        self.activation_name = activation
        self.activation = eval("act." + activation)
        self.number_neurons = number_neurons
        self.C = C
        self.beta_optimizer = beta_optimizer

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "bias" in params:
            self.bias = params.pop("bias")

    def build(self, input_shape):
        """
        Build the layer based on the input shape.

        Parameters:
        -----------
        - input_shape: Shape of the input data.
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
            self.alpha, _ = tf.linalg.qr(self.alpha)
            self.bias = self.bias / tf.norm(self.bias)

    def _compute_feature_matrix(self, x):
        """
        Compute the RVFL feature matrix D = [H, X, 1].

        Parameters:
        -----------
        - x (tf.Tensor): Input data.

        Returns:
        -----------
        - D (tf.Tensor): Feature matrix with hidden output, direct link, and bias.
        """
        n_sample = int(x.shape[0])

        # Hidden layer output
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        # RVFL: Concatenate with original input (direct link)
        D = tf.concat([H, x], axis=1)

        # Add bias column
        if self.include_bias:
            ones = tf.ones([n_sample, 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        return D

    def fit_initialize(self, x, y):
        """
        Initialize the layer and fit it to the initial data batch.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.
        - y (tf.Tensor): Output labels (one-hot encoded for classification).
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        n = int(x.shape[0])

        # Compute RVFL feature matrix
        D = self._compute_feature_matrix(x)
        d_cols = int(D.shape[1])

        # For online sequential learning, P must always be d_cols x d_cols
        # P = (D'D + C*I)^(-1)
        DTD = tf.matmul(D, D, transpose_a=True)
        P = tf.linalg.set_diag(DTD, tf.linalg.diag_part(DTD) + self.C)
        P = tf.linalg.inv(P)
        beta = tf.matmul(tf.matmul(P, D, transpose_b=True), y)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, D, y)
        else:
            self.beta = beta

        self.y = y
        self.input = x
        self.P = P
        self.feature_map = D

    def fit_seq(self, x, y):
        """
        Fit the layer sequentially to new data using Woodbury identity.

        Parameters:
        -----------
        - x (tf.Tensor): New input data.
        - y (tf.Tensor): New output labels.
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        # Compute feature matrix for new data
        D = self._compute_feature_matrix(x)

        # Update P using Woodbury matrix identity
        # P_new = P - P * D' * (I + D * P * D')^(-1) * D * P
        I = tf.eye(int(D.shape[0]), dtype=tf.float32)
        DPDt = tf.matmul(tf.matmul(D, self.P), D, transpose_b=True)
        inv_term = tf.linalg.inv(I + DPDt)
        update = tf.matmul(tf.matmul(tf.matmul(tf.matmul(self.P, D, transpose_b=True), inv_term), D), self.P)
        self.P = self.P - update

        # Update beta
        # beta_new = beta + P * D' * (y - D * beta)
        residual = y - tf.matmul(D, self.beta)
        beta = self.beta + tf.matmul(tf.matmul(self.P, D, transpose_b=True), residual)

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, D, y)
        else:
            self.beta = beta

        self.input = x
        self.feature_map = D

    def predict(self, x):
        """
        Predict the output for the given input data.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.

        Returns:
        -----------
        - tf.Tensor: Predicted output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        D = self._compute_feature_matrix(x)
        output = tf.matmul(D, self.beta)
        self.output = output
        return output

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.

        Returns:
        -----------
        - tf.Tensor: Predicted probabilities.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        """
        Calculate the output for the given input data.
        """
        return self.predict(x)

    def __str__(self):
        return self.name

    def count_params(self):
        """
        Count the number of trainable and non-trainable parameters.
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
        Convert the layer's attributes to a dictionary.
        """
        attributes = {
            'name': 'OSRVFLLayer',
            'number_neurons': self.number_neurons,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            'include_bias': self.include_bias,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load the layer from a dictionary of attributes.
        """
        return cls(**attributes)
