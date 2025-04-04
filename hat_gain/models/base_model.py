import tensorflow as tf


class Model(object):
    """
    Base model class for all models.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model.

        Args:
            name: Name of the model
            logging: Whether to log variables
        """
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, f'Invalid keyword argument: {kwarg}'

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        # Initialize attributes
        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        """
        Build the model.
        Must be implemented by specific models.
        """
        raise NotImplementedError

    def build(self):
        """
        Wrapper for _build().
        """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        """
        Predict function.
        Must be implemented by specific models.
        """
        pass

    def _loss(self):
        """
        Loss function.
        Must be implemented by specific models.
        """
        raise NotImplementedError

    def _accuracy(self):
        """
        Accuracy function.
        Must be implemented by specific models.
        """
        raise NotImplementedError

    def save(self, sess=None, save_path=None):
        """
        Save the model.
        """
        if save_path is None:
            save_path = f"./tmp/{self.name}.ckpt"

        if not sess:
            raise AttributeError("TensorFlow session not provided.")

        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, save_path)
        print(f"Model saved in file: {save_path}")

    def load(self, sess=None, load_path=None):
        """
        Load the model.
        """
        if load_path is None:
            load_path = f"./tmp/{self.name}.ckpt"

        if not sess:
            raise AttributeError("TensorFlow session not provided.")

        saver = tf.train.Saver(self.vars)
        saver.restore(sess, load_path)
        print(f"Model restored from file: {load_path}")