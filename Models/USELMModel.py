import h5py
import numpy as np

from Layers.USELMLayer import USELMLayer


class USELMModel:
    """
        Unsupervised ELM (US-ELM) Model for UNSUPERVISED learning.

        Task Type: EMBEDDING (dimensionality reduction) or CLUSTERING

        NOT FOR CLASSIFICATION OR REGRESSION.
        For classification/regression, use ELMModel instead.

        This class implements an Unsupervised Extreme Learning Machine that learns
        low-dimensional embeddings without using labels. It can be used for:
        - Dimensionality reduction (like PCA, t-SNE)
        - Clustering (with KMeans on embeddings)
        - Feature extraction for downstream tasks
        - Data visualization

        Parameters:
        -----------
        layer : USELMLayer
            The USELMLayer instance with embedding_size parameter.
        task : str, default='embedding'
            The type of task: 'embedding' or 'clustering'.
        n_clusters : int, default=None
            Number of clusters (required if task='clustering').
        random_weights : bool, default=True
            Whether to initialize the model with random weights.

        Examples:
        -----------
        Dimensionality reduction (embedding):

        >>> layer = USELMLayer(number_neurons=500, embedding_size=3, lam=0.001)
        >>> model = USELMModel(layer, task='embedding')
        >>> model.fit(X_train)
        >>> embeddings = model.predict(X_test)  # Returns 3D embeddings
        >>> # Visualize with matplotlib 3D scatter plot

        Clustering:

        >>> layer = USELMLayer(number_neurons=500, embedding_size=10, lam=0.001)
        >>> model = USELMModel(layer, task='clustering', n_clusters=5)
        >>> model.fit(X_train)
        >>> cluster_labels = model.predict(X_test)  # Returns cluster assignments

        Feature extraction for classification:

        >>> layer = USELMLayer(number_neurons=500, embedding_size=50, lam=0.001)
        >>> model = USELMModel(layer, task='embedding')
        >>> model.fit(X_train)
        >>> features_train = model.predict(X_train)
        >>> features_test = model.predict(X_test)
        >>> # Use features with any classifier (SVM, RandomForest, etc.)
        >>> clf = SVC()
        >>> clf.fit(features_train, y_train)
        >>> predictions = clf.predict(features_test)
    """
    def __init__(self, layer: USELMLayer, task='embedding', n_clusters=None, random_weights=True):
        self.classes_ = None
        self.task = task
        self.n_clusters = n_clusters
        self.number_neurons = layer.number_neurons
        self.activation = layer.activation
        self.act_params = layer.act_params
        self.C = layer.C
        self.is_orthogonalized = layer.is_orthogonalized
        self.layer = layer
        self.random_weights = random_weights

        if task == 'clustering' and n_clusters is None:
            raise ValueError("n_clusters must be specified when task='clustering'")

    def fit(self, x):
        """
           Fit the USELM model to the input data.

           Parameters:
           -----------
           - x: Input data.

           Examples:
           -----------
           >>> layer = USELMLayer(number_neurons=5000, embedding_size=3, lam=0.001)
           >>> model = USELMModel(layer)
           >>> model.fit(X)
       """
        if self.random_weights:
            self.layer.build(np.shape(x))
        self.classes_ = np.zeros(np.shape(x)[0])
        self.layer.fit(x)

    def predict(self, X, clustering=None, k=None):
        """
            Make predictions using the trained model.

            Parameters:
            -----------
            X : array-like
                Input data for prediction.
            clustering : bool, optional
                Whether to perform clustering. If None, uses self.task setting.
            k : int, optional
                Number of clusters. If None, uses self.n_clusters.

            Returns:
            -----------
            If task='embedding': Returns embeddings array.
            If task='clustering': Returns cluster labels array.
            If clustering=True explicitly: Returns tuple (embeddings, cluster_labels).

            Examples:
            -----------
            >>> model = USELMModel(layer, task='embedding')
            >>> embeddings = model.predict(X)

            >>> model = USELMModel(layer, task='clustering', n_clusters=5)
            >>> cluster_labels = model.predict(X)
        """
        # Determine clustering mode
        if clustering is None:
            do_clustering = self.task == 'clustering'
        else:
            do_clustering = clustering

        # Determine number of clusters
        n_clusters = k if k is not None else self.n_clusters

        if do_clustering:
            result = self.layer.predict(X, clustering=True, k=n_clusters)
            embeddings, cluster_labels = result
            # If task is clustering, return only labels
            if self.task == 'clustering' and clustering is None:
                return cluster_labels
            # If explicitly requested, return both
            return embeddings, cluster_labels
        else:
            result = self.layer.predict(X, clustering=False, k=None)
            return result.numpy()

    def save(self, file_path):
        """
            Save the model to an HDF5 file.

            Parameters:
            -----------
            - file_path (str): File path to save the model.

            Examples:
            -----------
            >>> model.save("Saved Models/USELM_Model_1.h5")
        """
        try:
            with h5py.File(file_path, 'w') as h5file:
                for key, value in self.to_dict().items():
                    if value is None:
                        value = 'None'
                    h5file.create_dataset(key, data=value)
                h5file.close()
        except Exception as e:
            print(f"Error saving to HDF5: {e}")

    @classmethod
    def load(cls, file_path: str):
        """
            Load the model from an HDF5 file.

            Parameters:
            -----------
            - file_path (str): File path to load the model from.

            Returns:
            -----------
            - Loaded USELMModel instance.

            Examples:
            -----------
            >>> model = model.load("Saved Models/USELM_Model_1.h5")
        """
        try:
            with h5py.File(file_path, 'r') as h5file:
                # Extract attributes from the HDF5 file
                attributes = {key: h5file[key][()] for key in h5file.keys()}

                for key, value in attributes.items():
                    if type(value) is bytes:
                        v = value.decode('utf-8')
                        attributes[key] = v

                if "name" in attributes:
                    l_type = attributes.pop("name")

                layer = eval(f"{l_type}(**attributes)")
                model = cls(layer)
                return model
        except Exception as e:
            print(f"Error loading from HDF5: {e}")
            return None  # Return None or raise an exception based on your error-handling strategy

    def to_dict(self):
        """
            Convert the model to a dictionary representation.

            Returns:
            -----------
            - Dictionary containing model attributes.
        """
        attributes = self.layer.to_dict()
        attributes["random_weights"] = self.random_weights

        filtered_attributes = {key: value for key, value in attributes.items() if value is not None}
        return filtered_attributes