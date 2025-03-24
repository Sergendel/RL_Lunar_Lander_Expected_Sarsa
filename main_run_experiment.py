import numpy as np
import yaml

from src.rl_glue import RLGlue
from src.lunar_lander import LunarLanderEnvironment
from tqdm import tqdm
import os
import shutil
from utils.plot_script import plot_result

from src.My_Agent import Agent


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"],
                                 experiment_parameters["num_episodes"]))

    env_info = {}

    agent_info = agent_parameters

    # one agent setting
    for run in range(1, experiment_parameters["num_runs"] + 1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)

        for episode in tqdm(range(1, experiment_parameters["num_episodes"] + 1)):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])

            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)
    shutil.make_archive('results', 'zip', 'results')


# Run Experiment

# Load YAML configuration
with open("config/config.yml", "r") as file:
    config = yaml.safe_load(file)

# Access configurations
experiment_parameters = config["experiment_parameters"]
agent_parameters = config["agent_parameters"]

# Verify loaded data
print("Config Params:")
print(experiment_parameters)
print(agent_parameters)

# experiment_parameters = {
#     "num_runs": 1,
#     "num_episodes": 300,
#     # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after
#     # some number of timesteps. Here we use the default of 500.
#     "timeout": 500
# }
#
# # Agent parameters
# agent_parameters = {
#     'network_config': {
#         'state_dim': 8,
#         'num_hidden_units': 256,
#         'num_actions': 4
#     },
#     'optimizer_config': {
#         'step_size': 1e-3,
#         'beta_m': 0.9,
#         'beta_v': 0.999,
#         'epsilon': 1e-8
#     },
#     'replay_buffer_size': 50000,
#     'minibatch_sz': 8,
#     'num_replay_updates_per_step': 4,
#     'gamma': 0.99,
#     'tau': 0.001
# }

# Environment parameters
environment_parameters = {}

current_env = LunarLanderEnvironment

current_agent = Agent

# run experiment
run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
plot_result(["expected_sarsa_agent", "random_agent"])