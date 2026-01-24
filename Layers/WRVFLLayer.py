from Layers.RVFLLayer import RVFLLayer
from Optimizers.ELMOptimizer import ELMOptimizer
from Resources.ActivationFunction import ActivationFunction
import tensorflow as tf
from Resources.gram_schmidt import gram_schmidt


class WRVFLLayer(RVFLLayer):
    """
        Weighted Random Vector Functional Link (WRVFL) Layer.

        Task Type: CLASSIFICATION only (designed for imbalanced datasets)

        NOT RECOMMENDED FOR REGRESSION.
        The weighting mechanism is based on class distribution and requires categorical labels.

        According to the literature, Weighted ELM/RVFL was designed specifically for CLASSIFICATION
        tasks with imbalanced class distributions. The weighting mechanism adjusts sample importance
        based on class frequencies, which is not applicable to regression problems.

        Reference:
        - Zong et al. "Weighted extreme learning machine for imbalance learning"
          Neurocomputing, 2013. DOI: 10.1016/j.neucom.2012.08.010

        This layer extends the RVFLLayer by incorporating weighted samples during training.
        Useful for handling imbalanced datasets.

        Parameters:
        -----------
            number_neurons (int): The number of neurons in the hidden layer.
            activation (str, optional): The activation function. Defaults to 'tanh'.
            act_params (dict, optional): Parameters for the activation function.
            C (float, optional): Regularization parameter. Defaults to 1.0.
            beta_optimizer (ELMOptimizer, optional): Optimizer for output weights.
            is_orthogonalized (bool, optional): Whether to orthogonalize weights.
            weight_method (str, optional): Method for computing sample weights.
                Options: 'wei-1', 'wei-2', 'ban-1', 'ban-decay'. Defaults to 'wei-1'.
            weight_param (int, optional): Parameter for weight computation. Defaults to 4.
            include_bias (bool, optional): Whether to include bias term. Defaults to True.
            **params: Additional parameters.

        Example:
        -----------
        >>> layer = WRVFLLayer(number_neurons=1000, activation='tanh', weight_method='wei-1')
        >>> model = RVFLModel(layer)
        >>> model.fit(X, y)
    """
    def __init__(self, number_neurons, activation='tanh', act_params=None, C=1.0,
                 beta_optimizer: ELMOptimizer = None, is_orthogonalized=False,
                 weight_method='wei-1', weight_param=4, include_bias=True, **params):
        super().__init__(number_neurons, activation, act_params, C, beta_optimizer,
                         is_orthogonalized, None, include_bias, **params)
        self.name = "wrvfl"
        self.weight_method = weight_method
        self.weight_param = weight_param

    def fit(self, x, y):
        """
        Fit the WRVFLLayer to the input-output pairs with weighted samples.

        Parameters:
        -----------
            x (tf.Tensor): Input data.
            y (tf.Tensor): Output data (one-hot encoded for classification).

        Returns:
        -----------
            None
        """
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        self.input = x

        n_sample = int(x.shape[0])

        # Compute hidden layer output
        H = tf.matmul(x, self.alpha) + self.bias
        H = self.activation(H)

        # RVFL: Concatenate hidden layer with original input
        D = tf.concat([H, x], axis=1)

        # Add bias column if specified
        if self.include_bias:
            ones = tf.ones([n_sample, 1], dtype=tf.float32)
            D = tf.concat([D, ones], axis=1)

        # Compute sample weights based on class distribution
        if self.weight_method == 'wei-1':
            ni = tf.reduce_sum(y, axis=0)
            W = tf.linalg.diag(tf.reduce_sum(tf.multiply(y, ni), axis=1))
        elif self.weight_method == 'wei-2':
            ni = tf.reduce_sum(y, axis=0)
            mean_ni = tf.reduce_mean(ni)
            ni = tf.where(ni <= mean_ni, 1 / ni, 0.618 / ni)
            W = tf.linalg.diag(tf.reduce_sum(tf.multiply(y, ni), axis=1))
        elif self.weight_method == 'ban-1':
            ni = tf.reduce_sum(y, axis=0)
            mean_ni = tf.reduce_mean(ni)
            ni = tf.where(ni > mean_ni, 1 / ni, 0.618 / ni)
            W = tf.linalg.diag(tf.reduce_sum(tf.multiply(y, ni), axis=1))
        elif self.weight_method == 'ban-decay':
            ni = tf.reduce_sum(y, axis=0)
            max_ni = tf.reduce_max(ni)
            ni = tf.pow(ni / max_ni, self.weight_param) / ni
            W = tf.linalg.diag(tf.reduce_sum(tf.multiply(y, ni), axis=1))

        d_cols = int(D.shape[1])

        # Compute beta with weighted samples
        if n_sample < d_cols:
            Dp = tf.matmul(tf.matmul(W, D), D, transpose_b=True)
            Dp = tf.linalg.set_diag(Dp, tf.linalg.diag_part(Dp) + self.C)
            pD = tf.linalg.inv(Dp)
            beta = tf.matmul(tf.matmul(tf.matmul(D, pD, transpose_a=True), W), y)
        else:
            Dp = tf.matmul(tf.matmul(D, W, transpose_a=True), D)
            Dp = tf.linalg.set_diag(Dp, tf.linalg.diag_part(Dp) + self.C)
            pD = tf.linalg.inv(Dp)
            beta = tf.matmul(tf.matmul(tf.matmul(pD, D, transpose_b=True), W), y)

        self.beta = beta

        if self.beta_optimizer is not None:
            self.beta, self.error_history = self.beta_optimizer.optimize(beta, D, y)

        self.feature_map = D
        self.output = tf.matmul(D, self.beta)

    def to_dict(self):
        """
        Convert layer attributes to a dictionary.
        """
        attributes = super().to_dict()
        attributes['name'] = 'WRVFLLayer'
        attributes['weight_method'] = self.weight_method
        attributes['weight_param'] = self.weight_param
        return attributes

    @classmethod
    def load(cls, attributes):
        """
        Load layer attributes from a dictionary.
        """
        return cls(**attributes)
