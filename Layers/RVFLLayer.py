from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.generate_contrainted_weights import generate_contrainted_weights
from Resources.gram_schmidt import gram_schmidt
from Resources.ReceptiveFieldGenerator import ReceptiveFieldGenerator
from Resources.ReceptiveFieldGaussianGenerator import ReceptiveFieldGaussianGenerator


class RVFLLayer:
    """
        Random Vector Functional Link (RVFL) Layer with TensorFlow.

        RVFL differs from ELM by including a direct connection from the input layer to the output layer.
        The output is computed as: D = [H, X] where H is the hidden layer output and X is the original input.
        This direct link often improves generalization performance.

        Parameters:
        -----------
        number_neurons : int
            The number of neurons in the hidden layer.
        activation : str, default='tanh'
            The name of the activation function to be applied to the neurons.
        act_params : dict, default=None
            Additional parameters for the activation function.
        C : float, default=None
            Regularization parameter (lambda) to control overfitting.
        beta_optimizer : ELMOptimizer, default=None
            An optimizer to optimize the output weights (beta) of the layer.
        is_orthogonalized : bool, default=False
            Indicates whether the input weights of the hidden neurons are orthogonalized.
        receptive_field_generator : ReceptiveFieldGenerator, default=None
            An object for generating receptive fields to constrain the input weights.
        **params : dict
            Additional parameters to be passed to the layer.

        Attributes:
        -----------
        feature_map : tensor, shape (n_samples, number_neurons + n_features)
            The feature map matrix including both hidden layer output and direct input.
        name : str, default="rvfl"
            The name of the layer.
        beta : tensor, shape (number_neurons + n_features + 1, n_outputs) or None
            The output weights matrix of the layer (includes bias term).
        bias : tensor, shape (number_neurons,) or None
            The bias vector of the hidden layer.
        alpha : tensor, shape (n_features, number_neurons) or None
            The input weights matrix of the layer.
        include_bias : bool, default=True
            Whether to include a bias term in the direct link.

        Example:
        -----------
        Initialize a Random Vector Functional Link (RVFL) layer with 1000 neurons:

        >>> rvfl = RVFLLayer(number_neurons=1000, activation='mish')

        Create an RVFL model using the RVFL layer:

        >>> model = RVFLModel(rvfl)

        Perform cross-validation:

        >>> cv = RepeatedKFold(n_splits=10, n_repeats=50)
        >>> scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', error_score='raise')
        >>> print(np.mean(scores))
    """
    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 receptive_field_generator=None,
                 include_bias=True,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "rvfl"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.act_params = act_params
        self.beta_optimizer = beta_optimizer
        self.is_orthogonalized = is_orthogonalized
        self.include_bias = include_bias
        self.C = C  # Regularization parameter (lambda)

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
        self.receptive_field_generator = receptive_field_generator

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

        if 'constrained' in params and params['constrained'] is True:
            self.constrained = True
        else:
            self.constrained = False

        if 'rf_name' in params:
            rf = eval(f"{params['rf_name']}.load(params)")
            self.receptive_field_generator = rf

    def build(self, input_shape):
        """
        Builds the RVFL layer by initializing weights and biases.

        Parameters:
        -----------
        - input_shape (tuple): The shape of the input data.

        Returns:
        -----------
        None

        Example:
        -----------
            >>> rvfl = RVFLLayer(number_neurons=1000, activation='mish')
            >>> rvfl.build(x.shape)
        """
        n_features = input_shape[-1]
        alpha_initializer = tf.random_uniform_initializer(-1, 1)
        self.alpha = tf.Variable(
            alpha_initializer(shape=(n_features, self.number_neurons)),
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

    def fit(self, x, y):
        """
        Fits the RVFL model to the given training data.

        The key difference from ELM is that RVFL concatenates the original input
        with the hidden layer output: D = [H, X, 1] (with optional bias term).

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

        n_sample = tf.shape(x)[0]
        n_feature = tf.shape(x)[1]

        if self.constrained:
            generate_contrainted_weights(x, y, self.number_neurons)
        if self.receptive_field_generator is not None:
            self.receptive_field_generator.generate_receptive_fields(self.alpha)

        # Compute hidden layer output: H = g(X * alpha + bias)
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        # RVFL: Concatenate hidden layer with original input (direct link)
        # D = [H, X]
        D = tf.concat([H, x], axis=1)

        # Add bias column if specified
        if self.include_bias:
            ones = tf.ones([tf.shape(D)[0], 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        # Compute beta using regularized least squares
        # beta = (D^T * D + lambda * I)^(-1) * D^T * y  (when n_samples > n_features)
        # beta = D^T * (D * D^T + lambda * I)^(-1) * y  (when n_features > n_samples)

        n_sample_int = int(D.shape[0])
        d_cols = int(D.shape[1])

        if n_sample_int > d_cols:
            # More samples than features
            DTD = tf.matmul(D, D, transpose_a=True)
            reg_matrix = self.C * tf.eye(d_cols, dtype=tf.float32)
            DTy = tf.matmul(D, y, transpose_a=True)
            self.beta = tf.linalg.solve(reg_matrix + DTD, DTy)
        else:
            # More features than samples
            DDT = tf.matmul(D, D, transpose_b=True)
            reg_matrix = self.C * tf.eye(n_sample_int, dtype=tf.float32)
            self.beta = tf.matmul(
                D, tf.linalg.solve(reg_matrix + DDT, y), transpose_a=True
            )

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(self.beta, D, y)

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

        Example:
        -----------
            >>> rvfl = RVFLLayer(number_neurons=1000, activation='mish')
            >>> rvfl.build(x.shape)
            >>> rvfl.fit(train_data, train_targets)
            >>> pred = rvfl.predict(test_data)
        """
        x = tf.cast(x, dtype=tf.float32)

        # Compute hidden layer
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        # RVFL: Concatenate with original input
        D = tf.concat([H, x], axis=1)

        # Add bias if specified
        if self.include_bias:
            ones = tf.ones([tf.shape(D)[0], 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        output = tf.matmul(D, self.beta)
        return output

    def predict_proba(self, x):
        """
        Predicts the probabilities output for the given input data.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.

        Returns:
        -----------
        tf.Tensor: Predicted probability tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def calc_output(self, x):
        """
        Calculates the output of the RVFL layer for the given input data.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.

        Returns:
        -----------
        tf.Tensor: Output tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        out = self.activation(tf.matmul(x, self.beta, transpose_b=True))
        self.output = out
        return out

    def apply_activation(self, x):
        """
        Applies activation function for the given input data.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.

        Returns:
        -----------
        tf.Tensor: Output tensor.
        """
        return self.activation(x)

    def __str__(self):
        """
        Returns a string representation of the RVFL layer.
        """
        return f"{self.name}, neurons: {self.number_neurons}"

    def count_params(self):
        """
        Counts the number of trainable and non-trainable parameters.
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
        Convert the RVFL layer attributes to a dictionary.
        """
        attributes = {
            'name': 'RVFLLayer',
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
        }
        if self.receptive_field_generator is not None:
            attributes.update(self.receptive_field_generator.to_dict())
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load an RVFL layer from a dictionary of attributes.
        """
        return cls(attributes)
