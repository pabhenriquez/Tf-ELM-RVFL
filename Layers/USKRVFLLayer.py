from sklearn.cluster import KMeans
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf

from Resources.Kernel import Kernel, CombinedSumKernel, CombinedProductKernel
from Resources.kernel_distances import calculate_pairwise_distances_vector, calculate_pairwise_distances


class USKRVFLLayer:
    """
        Unsupervised Kernel RVFL (USK-RVFL) Layer.

        Task Type: EMBEDDING (dimensionality reduction) and CLUSTERING only

        NOT FOR CLASSIFICATION OR REGRESSION.
        This is an UNSUPERVISED model that learns representations WITHOUT labels.
        For classification/regression, use KRVFLLayer with KRVFLModel instead.

        According to the literature, Unsupervised ELM/RVFL was designed for UNSUPERVISED
        learning tasks including dimensionality reduction, data visualization, feature
        extraction, and clustering. It combines kernel methods with graph Laplacian
        regularization while maintaining direct input connections (RVFL characteristic).

        Reference:
        - Huang et al. "Unsupervised extreme learning machines"
          Neurocomputing, 2014. DOI: 10.1016/j.neucom.2014.03.022

        This class represents an Unsupervised Kernel Random Vector Functional Link (USKRVFL) layer.
        It performs dimensionality reduction using kernel methods with direct input connections
        (RVFL characteristic).

        The key difference from USKELM is the inclusion of direct input links:
        - USKELM: Uses only kernel features
        - USKRVFL: Uses kernel features + direct input link

        Parameters:
        -----------
        - kernel: The kernel function to be used.
        - embedding_size (int): The size of the embedded feature space.
        - activation (str): The activation function to be used. Defaults to 'tanh'.
        - act_params (dict): Additional parameters for the activation function.
        - C (float): Regularization parameter C. Defaults to 1.0.
        - include_direct_link (bool): Whether to include direct input connections. Default is True.
        - nystrom_approximation (bool): Whether to use the Nystrom approximation method.
        - landmark_selection_method (str): Method for selecting landmarks. Defaults to 'random'.
        - lam (float): Lambda parameter for Laplacian regularization. Defaults to 0.5.

        Example:
        -----------
        >>> kernel = CombinedProductKernel([Kernel("rational_quadratic"), Kernel("exponential")])
        >>> layer = USKRVFLLayer(kernel=kernel, embedding_size=3, lam=0.001, include_direct_link=True)
        >>> model = USKRVFLModel(layer)
    """
    def __init__(self,
                 kernel,
                 embedding_size,
                 activation='tanh',
                 act_params=None,
                 C=1.0,
                 include_direct_link=True,
                 nystrom_approximation=False,
                 landmark_selection_method='random',
                 lam=0.5,
                 **params):
        self.error_history = None
        self.feature_map = None
        self.name = "uskrvfl"
        self.beta = None
        self.input = None
        self.output = None
        self.lam = lam
        self.act_params = act_params
        self.embedding_size = embedding_size
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
        Builds the USKRVFL layer.

        Parameters:
        -----------
        - input_shape: Shape of the input data.
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

    def fit(self, x):
        """
        Fits the USKRVFL layer to the input data.

        Parameters:
        -----------
        - x: Input data.
        """
        x = tf.cast(x, dtype=tf.float32)
        d = tf.shape(x)[-1]
        self.input = x

        if self.embedding_size > d + 1:
            raise Exception("Embedding size cannot be larger than input dimension + 1")

        # Laplacian Graph
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

        if self.nystrom_approximation:
            num_rows = tf.shape(x)[0]
            shuffled_indices = tf.random.shuffle(tf.range(num_rows))
            selected_indices = shuffled_indices[:100]
            L_land = tf.gather(x, selected_indices)
            C_nys = calculate_pairwise_distances_vector(x, L_land, self.kernel.ev)
            W_nys = calculate_pairwise_distances(L_land, self.kernel.ev)
            K = tf.matmul(tf.matmul(C_nys, tf.linalg.inv(W_nys)), C_nys, transpose_b=True)
        else:
            K = calculate_pairwise_distances(x, self.kernel.ev)

        # RVFL: Create feature matrix with direct link
        D = self._compute_feature_matrix(K, x)

        eq = tf.eye(tf.shape(D)[0]) + self.lam * tf.matmul(tf.matmul(L, D), D, transpose_b=True)
        e, v = tf.linalg.eigh(eq)
        sorted_indices = tf.argsort(e)
        v_sorted = tf.gather(v, sorted_indices, axis=1)
        v_trimmed = v_sorted[:, 1:self.embedding_size+1]
        norm_factor = tf.norm(tf.matmul(tf.matmul(D, tf.transpose(D)), v_trimmed), axis=0)
        v_trimmed = v_trimmed / norm_factor
        beta = tf.matmul(D, v_trimmed, transpose_a=True)

        self.beta = beta
        self.K = K

    def predict(self, x, clustering=False, k=None):
        """
        Makes predictions based on the input data.

        Parameters:
        -----------
        - x: Input data.
        - clustering (bool): Whether to perform clustering. Defaults to False.
        - k: Number of clusters if clustering is True.

        Returns:
        -----------
        - Predicted values (embeddings) and optionally cluster labels.
        """
        x = tf.cast(x, dtype=tf.float32)
        k_mat = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)

        # RVFL: Create feature matrix with direct link
        D = self._compute_feature_matrix(k_mat, x)

        output = tf.matmul(D, self.beta)
        self.output = output

        if not clustering:
            return output
        else:
            output_np = output.numpy()
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(output_np)
            cluster_labels = kmeans.labels_
            return output_np, cluster_labels

    def calc_output(self, x):
        """
        Calculates the output based on the input data.
        """
        x = tf.cast(x, dtype=tf.float32)
        k_mat = calculate_pairwise_distances_vector(x, self.input, self.kernel.ev)
        D = self._compute_feature_matrix(k_mat, x)
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
        Converts the layer to a dictionary representation.
        """
        attributes = {
            'name': 'USKRVFLLayer',
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
            "lam": self.lam,
            "embedding_size": self.embedding_size
        }
        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes

    @classmethod
    def load(cls, attributes):
        """
        Loads the layer from a dictionary of attributes.
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
