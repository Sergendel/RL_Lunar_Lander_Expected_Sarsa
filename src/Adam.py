from src.ActionValueNetwork import ActionValueNetwork
import numpy as np

### Work Required: Yes. Fill in code in __init__ and update_weights (~9-11 Lines).
class Adam():
    # Work Required: Yes. Fill in the initialization for self.m and self.v (~4 Lines).
    def __init__(self, layer_sizes,
                 optimizer_info):
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]

        for i in range(0, len(self.layer_sizes) - 1):
            # Hint: The initialization for m and v should look very much like the initializations of the weights
            # except for the fact that initialization here is to zeroes (see description above.)
            # Replace the None in each following line

            # your code here
            self.m[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))

        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to
        # the time step t. We can calculate these powers using an incremental product. At initialization then,
        # beta_m_product and beta_v_product should be ...? (Note that timesteps start at 1 and if we were to
        # start from 0, the denominator would be 0.)
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    # Work Required: Yes. Fill in the weight updates (~5-7 lines).
    def update_weights(self, weights, td_errors_times_gradients):
        """
        Args:
            weights (Array of dictionaries): The weights of the neural network.
            td_errors_times_gradients (Array of dictionaries): The gradient of the
            action-values with respect to the network's weights times the TD-error
        Returns:
            The updated weights (Array of dictionaries).
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                # Hint: Follow the equations above. First, you should update m and v and then compute
                # m_hat and v_hat. Finally, compute how much the weights should be incremented by.
                # self.m[i][param] = None
                # self.v[i][param] = None
                # m_hat = None
                # v_hat = None
                weight_update = None

                # your code here
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * td_errors_times_gradients[i][
                    param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * td_errors_times_gradients[i][
                    param] ** 2

                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                weight_update = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

                weights[i][param] = weights[i][param] + weight_update
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights


if __name__ == "__main__":
    # --------------
    # Debugging Cell
    # --------------
    # Feel free to make any changes to this cell to debug your code

    network_config = {"state_dim": 5,
                      "num_hidden_units": 2,
                      "num_actions": 3
                      }

    optimizer_info = {"step_size": 0.1,
                      "beta_m": 0.99,
                      "beta_v": 0.999,
                      "epsilon": 0.0001
                      }

    network = ActionValueNetwork(network_config)
    test_adam = Adam(network.layer_sizes, optimizer_info)

    print("m[0][\"W\"] shape: {}".format(test_adam.m[0]["W"].shape))
    print("m[0][\"b\"] shape: {}".format(test_adam.m[0]["b"].shape))
    print("m[1][\"W\"] shape: {}".format(test_adam.m[1]["W"].shape))
    print("m[1][\"b\"] shape: {}".format(test_adam.m[1]["b"].shape), "\n")

    assert (np.allclose(test_adam.m[0]["W"].shape, np.array([5, 2])))
    assert (np.allclose(test_adam.m[0]["b"].shape, np.array([1, 2])))
    assert (np.allclose(test_adam.m[1]["W"].shape, np.array([2, 3])))
    assert (np.allclose(test_adam.m[1]["b"].shape, np.array([1, 3])))

    print("v[0][\"W\"] shape: {}".format(test_adam.v[0]["W"].shape))
    print("v[0][\"b\"] shape: {}".format(test_adam.v[0]["b"].shape))
    print("v[1][\"W\"] shape: {}".format(test_adam.v[1]["W"].shape))
    print("v[1][\"b\"] shape: {}".format(test_adam.v[1]["b"].shape), "\n")

    assert (np.allclose(test_adam.v[0]["W"].shape, np.array([5, 2])))
    assert (np.allclose(test_adam.v[0]["b"].shape, np.array([1, 2])))
    assert (np.allclose(test_adam.v[1]["W"].shape, np.array([2, 3])))
    assert (np.allclose(test_adam.v[1]["b"].shape, np.array([1, 3])))

    assert (np.all(test_adam.m[0]["W"] == 0))
    assert (np.all(test_adam.m[0]["b"] == 0))
    assert (np.all(test_adam.m[1]["W"] == 0))
    assert (np.all(test_adam.m[1]["b"] == 0))

    assert (np.all(test_adam.v[0]["W"] == 0))
    assert (np.all(test_adam.v[0]["b"] == 0))
    assert (np.all(test_adam.v[1]["W"] == 0))
    assert (np.all(test_adam.v[1]["b"] == 0))