from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class SubRVFLLayer:
    """
        Sub-RVFL Layer for ensemble methods.

        This layer implements a sub-RVFL network that can be used as a component
        in ensemble models like bagging or boosting. Each sub-RVFL uses a random
        subset of features and/or samples.

        Parameters:
        -----------
        number_neurons : int
            Number of neurons in the hidden layer.
        activation : str
            Activation function. Defaults to 'tanh'.
        act_params : dict
            Parameters for the activation function.
        C : float
            Regularization parameter. Defaults to 1.0.
        beta_optimizer : ELMOptimizer
            Optimizer for beta parameters.
        is_orthogonalized : bool
            Whether to orthogonalize weights.
        feature_ratio : float
            Ratio of features to use (0.0 to 1.0). Defaults to 1.0.
        sample_ratio : float
            Ratio of samples to use (0.0 to 1.0). Defaults to 1.0.
        include_bias : bool
            Whether to include bias term. Defaults to True.

        Example:
        -----------
        >>> layer = SubRVFLLayer(number_neurons=500, feature_ratio=0.8)
        >>> model = RVFLModel(layer)
        >>> model.fit(X, y)
    """
    def __init__(self,
                 number_neurons,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 beta_optimizer: ELMOptimizer = None,
                 is_orthogonalized=False,
                 feature_ratio=1.0,
                 sample_ratio=1.0,
                 include_bias=True,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "sub_rvfl"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.act_params = act_params
        self.beta_optimizer = beta_optimizer
        self.is_orthogonalized = is_orthogonalized
        self.feature_ratio = feature_ratio
        self.sample_ratio = sample_ratio
        self.include_bias = include_bias
        self.C = C
        self.n_features = None
        self.selected_features = None

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

        if "beta" in params:
            self.beta = params.pop("beta")
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "bias" in params:
            self.bias = params.pop("bias")
        if "selected_features" in params:
            self.selected_features = params.pop("selected_features")
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
        Build the layer with random feature selection.

        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data.
        """
        total_features = input_shape[-1]
        n_selected = max(1, int(total_features * self.feature_ratio))

        # Random feature selection
        all_indices = tf.range(total_features)
        shuffled = tf.random.shuffle(all_indices)
        self.selected_features = shuffled[:n_selected]
        self.n_features = n_selected

        alpha_initializer = tf.random_uniform_initializer(-1, 1)
        self.alpha = tf.Variable(
            alpha_initializer(shape=(n_selected, self.number_neurons)),
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

    def _select_features(self, x):
        """
        Select the random subset of features.
        """
        return tf.gather(x, self.selected_features, axis=1)

    def _compute_feature_matrix(self, x):
        """
        Compute the RVFL feature matrix D = [H, X_selected, 1].
        """
        x_selected = self._select_features(x)
        n_sample = int(x.shape[0])

        H = tf.matmul(x_selected, self.alpha) + self.bias
        H = self.activation(H)

        # RVFL: Concatenate with selected input features
        D = tf.concat([H, x_selected], axis=1)

        if self.include_bias:
            ones = tf.ones([n_sample, 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        return D

    def fit(self, x, y):
        """
        Fit the Sub-RVFL layer.

        Parameters:
        -----------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data.
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x

        n_sample = int(x.shape[0])

        # Random sample selection if sample_ratio < 1.0
        if self.sample_ratio < 1.0:
            n_selected_samples = max(1, int(n_sample * self.sample_ratio))
            all_indices = tf.range(n_sample)
            shuffled = tf.random.shuffle(all_indices)
            selected_samples = shuffled[:n_selected_samples]
            x = tf.gather(x, selected_samples)
            y = tf.gather(y, selected_samples)
            n_sample = n_selected_samples

        # Compute RVFL feature matrix
        D = self._compute_feature_matrix(x)
        d_cols = int(D.shape[1])

        # Compute beta
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

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(self.beta, D, y)

        self.feature_map = D
        self.output = tf.matmul(D, self.beta)

    def predict(self, x):
        """
        Predict the output.

        Parameters:
        -----------
        x : tf.Tensor
            Input data.

        Returns:
        --------
        tf.Tensor
            Predicted output.
        """
        x = tf.cast(x, dtype=tf.float32)
        D = self._compute_feature_matrix(x)
        output = tf.matmul(D, self.beta)
        return output

    def predict_proba(self, x):
        """
        Predict class probabilities.
        """
        x = tf.cast(x, dtype=tf.float32)
        pred = self.predict(x)
        return tf.keras.activations.softmax(pred)

    def __str__(self):
        return f"{self.name}, neurons: {self.number_neurons}, features: {self.feature_ratio}"

    def count_params(self):
        """
        Count the number of parameters.
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
        Convert layer attributes to a dictionary.
        """
        attributes = {
            'name': 'SubRVFLLayer',
            'number_neurons': self.number_neurons,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            'feature_ratio': self.feature_ratio,
            'sample_ratio': self.sample_ratio,
            'include_bias': self.include_bias,
            "beta": self.beta,
            "alpha": self.alpha,
            "bias": self.bias,
            "selected_features": self.selected_features,
            "denoising": self.denoising,
            "denoising_param": self.denoising_param
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load layer from a dictionary.
        """
        return cls(**attributes)
