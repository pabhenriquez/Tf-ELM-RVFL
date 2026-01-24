from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.gram_schmidt import gram_schmidt


class DeepRVFLLayer:
    """
        Deep Random Vector Functional Link (Deep RVFL) Layer with TensorFlow.

        Deep RVFL extends the basic RVFL by stacking multiple hidden layers while maintaining
        direct connections from the input to the output. Each hidden layer's output is concatenated
        with all previous layers' outputs plus the original input.

        Parameters:
        -----------
        number_neurons : int
            The number of neurons in each hidden layer.
        n_layers : int
            The number of hidden layers.
        activation : str, default='tanh'
            The name of the activation function to be applied to the neurons.
        act_params : dict, default=None
            Additional parameters for the activation function.
        C : float, default=1.0
            Regularization parameter (lambda) to control overfitting.
        is_orthogonalized : bool, default=False
            Indicates whether the input weights are orthogonalized.
        include_bias : bool, default=True
            Whether to include a bias term.
        **params : dict
            Additional parameters.

        Attributes:
        -----------
        feature_map : tensor
            The concatenated feature map from all layers plus original input.
        name : str, default="deep_rvfl"
            The name of the layer.
        beta : tensor
            The output weights matrix.
        alphas : list
            List of input weight matrices for each hidden layer.
        biases : list
            List of bias vectors for each hidden layer.

        Example:
        -----------
        >>> deep_rvfl = DeepRVFLLayer(number_neurons=100, n_layers=3, activation='relu')
        >>> deep_rvfl.build(x.shape)
        >>> deep_rvfl.fit(X_train, y_train)
        >>> predictions = deep_rvfl.predict(X_test)
    """
    def __init__(self,
                 number_neurons,
                 n_layers=2,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 is_orthogonalized=False,
                 include_bias=True,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "deep_rvfl"
        self.beta = None
        self.alphas = []
        self.biases = []
        self.input = None
        self.output = None
        self.act_params = act_params
        self.is_orthogonalized = is_orthogonalized
        self.include_bias = include_bias
        self.C = C
        self.n_layers = n_layers
        self.number_neurons = number_neurons

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

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alphas" in params:
            self.alphas = params.pop("alphas")
        if "biases" in params:
            self.biases = params.pop("biases")

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
        Builds the Deep RVFL layer by initializing weights and biases for all hidden layers.

        Parameters:
        -----------
        - input_shape (tuple): The shape of the input data.

        Returns:
        -----------
        None
        """
        n_features = input_shape[-1]
        current_input_size = n_features

        self.alphas = []
        self.biases = []

        for i in range(self.n_layers):
            alpha_initializer = tf.random_uniform_initializer(-1, 1)
            alpha = tf.Variable(
                alpha_initializer(shape=(current_input_size, self.number_neurons)),
                dtype=tf.float32,
                trainable=False
            )
            bias_initializer = tf.random_uniform_initializer(0, 1)
            bias = tf.Variable(
                bias_initializer(shape=(self.number_neurons,)),
                dtype=tf.float32,
                trainable=False
            )

            if self.is_orthogonalized:
                alpha = gram_schmidt(alpha)
                bias = bias / tf.norm(bias)

            self.alphas.append(alpha)
            self.biases.append(bias)

            # Next layer input is the current hidden output
            current_input_size = self.number_neurons

    def fit(self, x, y):
        """
        Fits the Deep RVFL model to the given training data.

        In Deep RVFL, all hidden layer outputs are concatenated with the original
        input to form the final feature matrix D = [H1, H2, ..., Hn, X, 1].

        Parameters:
        -----------
            x (tf.Tensor): The input training data of shape (N, D).
            y (tf.Tensor): The target training data of shape (N, C).

        Returns:
        -----------
            None
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x

        n_sample = int(x.shape[0])

        # Start with original input for direct link
        D = x

        # Current hidden layer input
        H = x

        # Process each hidden layer
        for i in range(self.n_layers):
            # Compute hidden layer output
            H = tf.matmul(H, self.alphas[i]) + self.biases[i]
            H = self.activation(H)

            # Concatenate hidden output with accumulated features (Deep RVFL style)
            D = tf.concat([H, D], axis=1)

        # Add bias column if specified
        if self.include_bias:
            ones = tf.ones([n_sample, 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        # Compute beta using regularized least squares
        d_cols = int(D.shape[1])

        if n_sample > d_cols:
            DTD = tf.matmul(D, D, transpose_a=True)
            reg_matrix = self.C * tf.eye(d_cols, dtype=tf.float32)
            self.beta = tf.matmul(
                tf.matmul(tf.linalg.inv(reg_matrix + DTD), D, transpose_b=True),
                y
            )
        else:
            DDT = tf.matmul(D, D, transpose_b=True)
            reg_matrix = self.C * tf.eye(n_sample, dtype=tf.float32)
            self.beta = tf.matmul(
                tf.matmul(D, tf.linalg.inv(reg_matrix + DDT), transpose_a=True),
                y
            )

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
        n_sample = int(x.shape[0])

        # Start with original input
        D = x
        H = x

        # Process each hidden layer
        for i in range(self.n_layers):
            H = tf.matmul(H, self.alphas[i]) + self.biases[i]
            H = self.activation(H)
            D = tf.concat([H, D], axis=1)

        # Add bias if specified
        if self.include_bias:
            ones = tf.ones([n_sample, 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        output = tf.matmul(D, self.beta)
        return output

    def predict_proba(self, x):
        """
        Predicts the probabilities output for the given input data.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def __str__(self):
        return f"{self.name}, neurons: {self.number_neurons}, layers: {self.n_layers}"

    def count_params(self):
        """
        Counts the number of trainable and non-trainable parameters.
        """
        if self.beta is None:
            trainable = 0
        else:
            trainable = self.beta.shape[0] * self.beta.shape[1]

        non_trainable = 0
        for alpha, bias in zip(self.alphas, self.biases):
            non_trainable += alpha.shape[0] * alpha.shape[1] + bias.shape[0]

        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
        Convert the Deep RVFL layer attributes to a dictionary.
        """
        attributes = {
            'name': 'DeepRVFLLayer',
            'number_neurons': self.number_neurons,
            'n_layers': self.n_layers,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            'include_bias': self.include_bias,
            "beta": self.beta,
            "alphas": self.alphas,
            "biases": self.biases,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param,
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load a Deep RVFL layer from a dictionary of attributes.
        """
        return cls(**attributes)
