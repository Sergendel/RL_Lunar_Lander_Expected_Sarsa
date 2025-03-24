import numpy as np

def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions).
                       The action-values computed by an action-value network.
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """

    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = None
    # Compute the maximum preference across the actions
    max_preference = None

    # your code here
    preferences = action_values / tau
    # print(f"preferences = {preferences}")
    # print(f"preferences.shape = {preferences.shape}")
    max_preference = np.max(preferences, axis=1)
    # print(f"max_preference = {max_preference}")
    # print(f"max_preference.shape = {max_preference.shape}")

    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))

    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = None
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = None

    # your code here
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)

    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = None

    # your code here
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences

    # squeeze() removes any singleton dimensions. It is used here because this function is used in the
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    # print(f"action_probs = {action_probs}")
    return action_probs


if __name__ == "__main__":
    test_action_values = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    tau = 0.5
    probabilities = softmax(test_action_values, tau)
    print("Action probabilities:\n", probabilities)

    rand_generator = np.random.RandomState(0)
    action_values = rand_generator.normal(0, 1, (2, 4))
    tau = 0.5

    action_probs = softmax(action_values, tau)
    print("action_probs", action_probs)

    assert (np.allclose(action_probs, np.array([
        [0.25849645, 0.01689625, 0.05374514, 0.67086216],
        [0.84699852, 0.00286345, 0.13520063, 0.01493741]
    ])))

    print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")