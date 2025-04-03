import tensorflow as tf


class Layer(object):
    """Base layer class for neighbor samplers"""

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        if not self.name:
            self.name = self.__class__.__name__.lower()

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors from adjacency matrix.

    Args:
        adj_info: Adjacency information matrix of shape [n_nodes, max_degree]
    """

    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        """
        Sample neighbors uniformly.

        Args:
            inputs: Tuple of (node_ids, num_samples)
                - node_ids: Node IDs to sample neighbors for
                - num_samples: Number of neighbors to sample per node

        Returns:
            Sampled neighbor IDs for each node
        """
        node_ids, num_samples = inputs

        # Look up neighbor information for the batch of nodes
        adj_lists = tf.nn.embedding_lookup(self.adj_info, node_ids)

        # Randomly shuffle neighbors
        adj_lists = tf.transpose(tf.random.shuffle(tf.transpose(adj_lists)))

        # Take only the requested number of samples
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])

        return adj_lists