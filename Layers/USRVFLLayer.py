from sklearn.cluster import KMeans
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class USRVFLLayer:
    """
        Unsupervised Random Vector Functional Link (US-RVFL) Layer.

        Task Type: EMBEDDING (dimensionality reduction) and CLUSTERING only

        NOT FOR CLASSIFICATION OR REGRESSION.
        This is an UNSUPERVISED model that learns representations WITHOUT labels.
        For classification/regression, use RVFLLayer with RVFLModel instead.

        According to the literature, Unsupervised ELM/RVFL was designed for UNSUPERVISED
        learning tasks including dimensionality reduction, data visualization, feature
        extraction, and clustering. It uses graph Laplacian regularization to preserve
        data structure while maintaining direct input connections (RVFL characteristic).

        Reference:
        - Huang et al. "Unsupervised extreme learning machines"
          Neurocomputing, 2014. DOI: 10.1016/j.neucom.2014.03.022

        This layer implements unsupervised learning with RVFL for dimensionality
        reduction and data embedding, while maintaining direct input connections.

        Parameters:
        -----------
        - number_neurons (int): Number of neurons in the hidden layer.
        - embedding_size (int): Size of the embedding space.
        - activation (str): Activation function. Defaults to 'tanh'.
        - act_params (dict): Parameters for the activation function.
        - C (float): Regularization parameter. Defaults to 1.0.
        - is_orthogonalized (bool): Whether to orthogonalize weights.
        - lam (float): Regularization parameter for graph Laplacian. Defaults to 0.5.
        - include_bias (bool): Whether to include bias term. Defaults to True.

        Example:
        -----------
        >>> layer = USRVFLLayer(number_neurons=5000, embedding_size=3, lam=0.001)
        >>> model = USRVFLModel(layer)
        >>> model.fit(X)
        >>> embeddings = model.predict(X)
    """

    def __init__(self,
                 number_neurons,
                 embedding_size,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 is_orthogonalized=False,
                 lam=0.5,
                 include_bias=True,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "usrvfl"
        self.beta = None
        self.bias = None
        self.alpha = None
        self.input = None
        self.output = None
        self.lam = lam
        self.act_params = act_params
        self.is_orthogonalized = is_orthogonalized
        self.embedding_size = embedding_size
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
        Build the layer by initializing weights and biases.

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

    def fit(self, x):
        """
        Fit the layer to the input data (unsupervised).

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.
        """
        x = tf.cast(x, dtype=tf.float32)
        d = int(x.shape[-1])
        self.input = x
        n_sample = int(x.shape[0])

        if self.embedding_size > d + 1:
            raise Exception("Embedding size cannot be larger than input dimensions + 1")

        # Compute Laplacian Graph
        squared_norms = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        dot_product = tf.matmul(x, x, transpose_b=True)
        distances = squared_norms - 2 * dot_product + tf.transpose(squared_norms)
        distances = tf.maximum(distances, 0.0)
        sigma = 1.0
        W = tf.exp(-distances / (2.0 * sigma ** 2))
        D_diag = tf.linalg.diag(tf.reduce_sum(W, axis=1))
        L_unnormalized = D_diag - W
        D_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(D_diag))
        L = tf.matmul(tf.matmul(D_sqrt_inv, L_unnormalized), D_sqrt_inv)

        # Compute RVFL feature matrix
        D = self._compute_feature_matrix(x)
        d_cols = int(D.shape[1])

        # Solve eigenvalue problem for dimensionality reduction
        if n_sample > d_cols:
            eq = tf.eye(d_cols) + self.lam * tf.matmul(tf.matmul(D, L, transpose_a=True), D)
            e, v = tf.linalg.eigh(eq)
            sorted_indices = tf.argsort(e)
            v_sorted = tf.gather(v, sorted_indices, axis=1)
            v_trimmed = v_sorted[:, 1:self.embedding_size + 1]
            norm_factor = tf.norm(tf.matmul(D, v_trimmed), axis=0)
            v_trimmed = v_trimmed / norm_factor
            beta = v_trimmed
        else:
            eq = tf.eye(n_sample) + self.lam * tf.matmul(tf.matmul(L, D), D, transpose_b=True)
            e, v = tf.linalg.eigh(eq)
            sorted_indices = tf.argsort(e)
            v_sorted = tf.gather(v, sorted_indices, axis=1)
            v_trimmed = v_sorted[:, 1:self.embedding_size + 1]
            norm_factor = tf.norm(tf.matmul(tf.matmul(D, tf.transpose(D)), v_trimmed), axis=0)
            v_trimmed = v_trimmed / norm_factor
            beta = tf.matmul(D, v_trimmed, transpose_a=True)

        self.beta = beta
        self.feature_map = D
        self.output = tf.matmul(D, self.beta)

    def predict(self, x, clustering=False, k=None):
        """
        Predicts the embedding for the given input data.

        Parameters:
        -----------
        - x (tf.Tensor): Input data tensor.
        - clustering (bool): Whether to apply K-means clustering.
        - k (int): Number of clusters (required if clustering=True).

        Returns:
        -----------
        tf.Tensor: Embedding output, optionally with cluster labels.
        """
        x = tf.cast(x, dtype=tf.float32)
        D = self._compute_feature_matrix(x)
        output = tf.matmul(D, self.beta)

        if not clustering:
            return output
        else:
            output_np = output.numpy()
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(output_np)
            cluster_labels = kmeans.labels_
            return output, cluster_labels

    def calc_output(self, x):
        """
        Calculates the output of the layer.
        """
        x = tf.cast(x, dtype=tf.float32)
        out = self.activation(tf.matmul(x, self.beta, transpose_b=True))
        self.output = out
        return out

    def __str__(self):
        return f"{self.name}, neurons: {self.number_neurons}, embedding: {self.embedding_size}"

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
        Convert the layer attributes to a dictionary.
        """
        attributes = {
            'name': 'USRVFLLayer',
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
            "lam": self.lam,
            "embedding_size": self.embedding_size
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Load a USRVFLLayer instance from a dictionary.
        """
        return cls(**attributes)
