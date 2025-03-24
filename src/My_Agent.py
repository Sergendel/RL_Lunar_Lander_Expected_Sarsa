### Work Required: Yes. Fill in code in agent_step and agent_end (~7 Lines).
from src.agent import BaseAgent
from src.ReplayBuffer import ReplayBuffer
from src.Adam import Adam
from src.ActionValueNetwork import ActionValueNetwork
from src.softmax import softmax
from src.optimize_network import optimize_network
from copy import deepcopy
import numpy as np

class Agent(BaseAgent):
    def __init__(self):
        self.name = "expected_sarsa_agent"

    # Work Required: No.
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer,
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        self.optimizer = Adam(self.network.layer_sizes, agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

    # Work Required: No.
    def policy(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action.
        """
        action_values = self.network.get_action_values(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    # Work Required: No.
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    # Work Required: Yes. Fill in the action selection, replay-buffer update,
    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        # your code here
        action = self.policy(state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        # your code here
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network (~1 Line)
                # your code here

        # Update the last state and last action.
        ### START CODE HERE (~2 Lines)
        self.last_state = state
        self.last_action = action
        ### END CODE HERE
        # your code here

        return action

    # Work Required: Yes. Fill in the replay-buffer update and
    # update of the weights using optimize_network (~2 lines).
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments

        # your code here
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network
                # your code here
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

if __name__ == "__main__":
    # -----------
    # Tested Cell
    # -----------
    # The contents of the cell will be tested by the autograder.
    # If they do not pass here, they will not pass there.

    agent_info = {
        'network_config': {
            'state_dim': 8,
            'num_hidden_units': 256,
            'num_hidden_layers': 1,
            'num_actions': 4
        },
        'optimizer_config': {
            'step_size': 3e-5,
            'beta_m': 0.9,
            'beta_v': 0.999,
            'epsilon': 1e-8
        },
        'replay_buffer_size': 32,
        'minibatch_sz': 32,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 1000.0,
        'seed': 0}

    # Initialize agent
    agent = Agent()
    agent.agent_init(agent_info)

    # load agent network, optimizer, replay_buffer from the agent_input_1.npz file
    input_data = np.load("../asserts/agent_input_1.npz", allow_pickle=True)
    agent.network.set_weights(input_data["network_weights"])
    agent.optimizer.m = input_data["optimizer_m"]
    agent.optimizer.v = input_data["optimizer_v"]
    agent.optimizer.beta_m_product = input_data["optimizer_beta_m_product"]
    agent.optimizer.beta_v_product = input_data["optimizer_beta_v_product"]
    agent.replay_buffer.rand_generator.seed(int(input_data["replay_buffer_seed"]))
    for experience in input_data["replay_buffer"]:
        agent.replay_buffer.buffer.append(experience)

    # Perform agent_step multiple times
    last_state_array = input_data["last_state_array"]
    last_action_array = input_data["last_action_array"]
    state_array = input_data["state_array"]
    reward_array = input_data["reward_array"]

    for i in range(5):
        agent.last_state = last_state_array[i]
        agent.last_action = last_action_array[i]
        state = state_array[i]
        reward = reward_array[i]

        agent.agent_step(reward, state)

        # Load expected values for last_state, last_action, weights, and replay_buffer
        output_data = np.load("asserts/agent_step_output_{}.npz".format(i), allow_pickle=True)
        answer_last_state = output_data["last_state"]
        answer_last_action = output_data["last_action"]
        answer_updated_weights = output_data["updated_weights"]
        answer_replay_buffer = output_data["replay_buffer"]

        # Asserts for last_state and last_action
        assert (np.allclose(answer_last_state, agent.last_state))
        assert (np.allclose(answer_last_action, agent.last_action))

        # # Asserts for replay_buffer
        # for i in range(answer_replay_buffer.shape[0]):
        #     for j in range(answer_replay_buffer.shape[1]):
        #         assert (np.allclose(np.asarray(agent.replay_buffer.buffer)[i, j], answer_replay_buffer[i, j]))

        # Asserts for network.weights
        assert (np.allclose(agent.network.weights[0]["W"], answer_updated_weights[0]["W"]))
        assert (np.allclose(agent.network.weights[0]["b"], answer_updated_weights[0]["b"]))
        assert (np.allclose(agent.network.weights[1]["W"], answer_updated_weights[1]["W"]))
        assert (np.allclose(agent.network.weights[1]["b"], answer_updated_weights[1]["b"]))

        print("Passed the asserts!")