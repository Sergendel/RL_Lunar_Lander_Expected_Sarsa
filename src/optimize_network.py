import numpy as np
from src.get_td_error import get_td_error
from src.ActionValueNetwork import ActionValueNetwork
from src.Adam import Adam

def optimize_network(experiences, discount, optimizer, network, current_q, tau):
    """
    Args:
        experiences (Numpy array): The batch of experiences including the states, actions,
                                   rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                        and particularly, the action-values at the next-states.
    """

    # Get states, action, rewards, terminals, and next_states from experiences
    states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]

    # Compute TD error using the get_td_error function
    # Note that q_vec is a 1D array of shape (batch_size)
    delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)

    # Batch Indices is an array from 0 to the batch_size - 1.
    batch_indices = np.arange(batch_size)

    # Make a td error matrix of shape (batch_size, num_actions)
    # delta_mat has non-zero value only for actions taken
    delta_mat = np.zeros((batch_size, network.num_actions))
    delta_mat[batch_indices, actions] = delta_vec

    # Pass delta_mat to compute the TD errors times the gradients of the network's weights from back-propagation

    ### START CODE HERE
    td_update = network.get_TD_update(states, delta_mat)
    ### END CODE HERE

    # Pass network.get_weights and the td_update to the optimizer to get updated weights
    ### START CODE HERE
    weights = optimizer.update_weights(network.get_weights(), td_update)
    ### END CODE HERE

    network.set_weights(weights)

if __name__ == "__main__":
    # -----------
    # Tested Cell
    # -----------
    # The contents of the cell will be tested by the autograder.
    # If they do not pass here, they will not pass there.

    input_data = np.load("../asserts/optimize_network_input_1.npz", allow_pickle=True)

    experiences = list(input_data["experiences"])
    discount = input_data["discount"]
    tau = 0.001

    network_config = {"state_dim": 8,
                      "num_hidden_units": 512,
                      "num_actions": 4
                      }

    network = ActionValueNetwork(network_config)
    network.set_weights(input_data["network_weights"])

    current_q = ActionValueNetwork(network_config)
    current_q.set_weights(input_data["current_q_weights"])

    optimizer_config = {'step_size': 3e-5,
                        'beta_m': 0.9,
                        'beta_v': 0.999,
                        'epsilon': 1e-8
                        }
    optimizer = Adam(network.layer_sizes, optimizer_config)
    optimizer.m = input_data["optimizer_m"]
    optimizer.v = input_data["optimizer_v"]
    optimizer.beta_m_product = input_data["optimizer_beta_m_product"]
    optimizer.beta_v_product = input_data["optimizer_beta_v_product"]

    optimize_network(experiences, discount, optimizer, network, current_q, tau)
    updated_weights = network.get_weights()

    output_data = np.load("../asserts/optimize_network_output_1.npz", allow_pickle=True)
    answer_updated_weights = output_data["updated_weights"]

    assert (np.allclose(updated_weights[0]["W"], answer_updated_weights[0]["W"]))
    assert (np.allclose(updated_weights[0]["b"], answer_updated_weights[0]["b"]))
    assert (np.allclose(updated_weights[1]["W"], answer_updated_weights[1]["W"]))
    assert (np.allclose(updated_weights[1]["b"], answer_updated_weights[1]["b"]))
    print("Passed the asserts!")