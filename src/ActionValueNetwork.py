# Do not modify this cell!

# Import necessary libraries
# DO NOT IMPORT OTHER LIBRARIES - This will break the autograder.
import numpy as np
from copy import deepcopy



# -----------
# Graded Cell
# -----------

# Work Required: Yes. Fill in the code for layer_sizes in __init__ (~1 Line).
# Also go through the rest of the code to ensure your understanding is correct.
class ActionValueNetwork:
    # Work Required: Yes. Fill in the layer_sizes member variable (~1 Line).
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        self.rand_generator = np.random.RandomState(network_config.get("seed"))

        # Specify self.layer_sizes which shows the number of nodes in each layer
        # your code here
        self.layer_sizes = np.array([self.state_dim, self.num_hidden_units, self.num_actions])

        # Initialize the weights of the neural network
        # self.weights is an array of dictionaries with each dictionary corresponding to
        # the weights from one layer to the next. Each dictionary includes W and b
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))

    # Work Required: No.
    def get_action_values(self, s):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)

        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

        return q_vals

    # Work Required: No.
    def get_TD_update(self, s, delta_mat):
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions). Each row of delta_mat
            correspond to one state in the batch. Each row has only one non-zero element
            which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']

        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        # td_update has the same structure as self.weights, that is an array of dictionaries.
        # td_update[0]["W"], td_update[0]["b"], td_update[1]["W"], and td_update[1]["b"] have the same shape as
        # self.weights[0]["W"], self.weights[0]["b"], self.weights[1]["W"], and self.weights[1]["b"] respectively
        td_update = [dict() for i in range(len(self.weights))]

        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        return td_update

    # Work Required: No. You may wish to read the relevant paper for more information on this weight initialization
    # (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks by Saxe, A et al., 2013)
    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    # Work Required: No.
    def get_weights(self):
        """
        Returns:
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)

    # Work Required: No.
    def set_weights(self, weights):
        """
        Args:
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)

if __name__ == "__main__":
    # --------------
    # Debugging Cell
    # --------------
    # Feel free to make any changes to this cell to debug your code

    network_config = {
        "state_dim": 5,
        "num_hidden_units": 20,
        "num_actions": 3
    }

    test_network = ActionValueNetwork(network_config)
    print("layer_sizes:", test_network.layer_sizes)
    assert (np.allclose(test_network.layer_sizes, np.array([5, 20, 3])))
    print(f'expected output:layer_sizes: [ 5 20  3]')


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.

rand_generator = np.random.RandomState(0)
for _ in range(1000):
    network_config = {
        "state_dim": rand_generator.randint(2, 10),
        "num_hidden_units": rand_generator.randint(2, 1024),
        "num_actions": rand_generator.randint(2, 10)
    }

    test_network = ActionValueNetwork(network_config)

    assert(np.allclose(test_network.layer_sizes, np.array([network_config["state_dim"],
                                                           network_config["num_hidden_units"],
                                                           network_config["num_actions"]])))
