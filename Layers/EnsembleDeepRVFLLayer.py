from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.gram_schmidt import gram_schmidt


class EnsembleDeepRVFLLayer:
    """
        Ensemble Deep Random Vector Functional Link (Ensemble Deep RVFL) Layer with TensorFlow.

        Ensemble Deep RVFL trains a separate output layer (beta) for each hidden layer.
        During prediction, the outputs from all layers are combined through voting (classification)
        or averaging (regression).

        Parameters:
        -----------
        number_neurons : int
            The number of neurons in each hidden layer.
        n_layers : int
            The number of hidden layers (also the number of ensemble members).
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
        betas : list
            List of output weight matrices (one per ensemble member).
        alphas : list
            List of input weight matrices for each hidden layer.
        biases : list
            List of bias vectors for each hidden layer.

        Example:
        -----------
        >>> ensemble_rvfl = EnsembleDeepRVFLLayer(number_neurons=100, n_layers=5, activation='relu')
        >>> ensemble_rvfl.build(x.shape)
        >>> ensemble_rvfl.fit(X_train, y_train)
        >>> vote_pred, add_pred = ensemble_rvfl.predict(X_test)
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
        self.feature_maps = []
        self.name = "ensemble_deep_rvfl"
        self.betas = []  # One beta per layer (ensemble)
        self.alphas = []
        self.biases = []
        self.input = None
        self.outputs = []
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

        if "betas" in params:
            self.betas = params.pop("betas")
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
        Builds the Ensemble Deep RVFL layer.

        Parameters:
        -----------
        - input_shape (tuple): The shape of the input data.
        """
        n_features = input_shape[-1]

        self.alphas = []
        self.biases = []
        self.betas = []

        # First layer takes original input
        # Subsequent layers take: hidden_output + original_input
        current_input_size = n_features

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

            # Next layer input size: hidden neurons + original features
            current_input_size = self.number_neurons + n_features

    def fit(self, x, y):
        """
        Fits the Ensemble Deep RVFL model to the given training data.

        Each layer trains its own beta, allowing ensemble prediction.

        Parameters:
        -----------
            x (tf.Tensor): The input training data of shape (N, D).
            y (tf.Tensor): The target training data of shape (N, C).
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x

        n_sample = int(x.shape[0])
        n_features = int(x.shape[1])

        self.betas = []
        self.feature_maps = []

        # H starts as original input
        H = x

        for i in range(self.n_layers):
            # Compute hidden layer output
            H_new = tf.matmul(H, self.alphas[i]) + self.biases[i]
            H_new = self.activation(H_new)

            # Build feature matrix D = [H_new, original_x]
            D = tf.concat([H_new, x], axis=1)

            # Update H for next layer: D is the new input
            H = D

            # Add bias column if specified
            if self.include_bias:
                ones = tf.ones([n_sample, 1], dtype=tf.float32)
                D_full = tf.concat([D, ones], axis=1)
            else:
                D_full = D

            # Compute beta for this layer using regularized least squares
            d_cols = int(D_full.shape[1])

            if n_sample > d_cols:
                DTD = tf.matmul(D_full, D_full, transpose_a=True)
                reg_matrix = self.C * tf.eye(d_cols, dtype=tf.float32)
                beta = tf.matmul(
                    tf.matmul(tf.linalg.inv(reg_matrix + DTD), D_full, transpose_b=True),
                    y
                )
            else:
                DDT = tf.matmul(D_full, D_full, transpose_b=True)
                reg_matrix = self.C * tf.eye(n_sample, dtype=tf.float32)
                beta = tf.matmul(
                    tf.matmul(D_full, tf.linalg.inv(reg_matrix + DDT), transpose_a=True),
                    y
                )

            self.betas.append(beta)
            self.feature_maps.append(D_full)

    def predict(self, x):
        """
        Predicts the output for the given input data using ensemble methods.

        Returns:
        -----------
        For classification:
            - vote_result: Majority voting result
            - (add_result, add_proba): Addition-based result and probabilities

        For regression:
            - Mean of all ensemble outputs
        """
        x = tf.cast(x, dtype=tf.float32)
        n_sample = int(x.shape[0])

        outputs = []
        H = x

        for i in range(self.n_layers):
            # Compute hidden layer output
            H_new = tf.matmul(H, self.alphas[i]) + self.biases[i]
            H_new = self.activation(H_new)

            # Build feature matrix
            D = tf.concat([H_new, x], axis=1)
            H = D

            # Add bias if specified
            if self.include_bias:
                ones = tf.ones([n_sample, 1], dtype=tf.float32)
                D_full = tf.concat([D, ones], axis=1)
            else:
                D_full = D

            # Compute output for this ensemble member
            output = tf.matmul(D_full, self.betas[i])
            outputs.append(output)

        return outputs

    def predict_vote(self, x):
        """
        Predicts using majority voting (for classification).
        """
        outputs = self.predict(x)

        # Get class predictions from each ensemble member
        predictions = [tf.argmax(output, axis=1) for output in outputs]

        # Stack predictions: shape (n_layers, n_samples)
        predictions_stacked = tf.stack(predictions, axis=0)

        # Transpose to (n_samples, n_layers) for voting
        predictions_stacked = tf.transpose(predictions_stacked)

        # Majority voting
        vote_results = []
        for i in range(predictions_stacked.shape[0]):
            sample_preds = predictions_stacked[i]
            # Count occurrences
            counts = tf.math.bincount(tf.cast(sample_preds, tf.int32))
            vote_results.append(tf.argmax(counts))

        return tf.stack(vote_results)

    def predict_addition(self, x):
        """
        Predicts by summing outputs and applying softmax (for classification).
        """
        outputs = self.predict(x)

        # Sum all outputs
        summed_output = tf.add_n(outputs)

        # Apply softmax
        proba = tf.nn.softmax(summed_output)
        result = tf.argmax(proba, axis=1)

        return result, proba

    def predict_mean(self, x):
        """
        Predicts by averaging outputs (for regression).
        """
        outputs = self.predict(x)

        # Stack and compute mean
        stacked = tf.stack(outputs, axis=0)
        mean_output = tf.reduce_mean(stacked, axis=0)

        return mean_output

    def predict_proba(self, x):
        """
        Predicts probabilities using addition method.
        """
        _, proba = self.predict_addition(x)
        return proba

    def __str__(self):
        return f"{self.name}, neurons: {self.number_neurons}, layers: {self.n_layers}"

    def count_params(self):
        """
        Counts the number of parameters.
        """
        trainable = 0
        for beta in self.betas:
            if beta is not None:
                trainable += beta.shape[0] * beta.shape[1]

        non_trainable = 0
        for alpha, bias in zip(self.alphas, self.biases):
            non_trainable += alpha.shape[0] * alpha.shape[1] + bias.shape[0]

        return {'trainable': trainable, 'non_trainable': non_trainable, 'all': trainable + non_trainable}

    def to_dict(self):
        """
        Convert the Ensemble Deep RVFL layer attributes to a dictionary.
        """
        attributes = {
            'name': 'EnsembleDeepRVFLLayer',
            'number_neurons': self.number_neurons,
            'n_layers': self.n_layers,
            'activation': self.activation_name,
            'act_params': self.act_params,
            'C': self.C,
            'is_orthogonalized': self.is_orthogonalized,
            'include_bias': self.include_bias,
            "betas": self.betas,
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
        Load an Ensemble Deep RVFL layer from a dictionary of attributes.
        """
        return cls(**attributes)
