import configparser
from collections import namedtuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    pass

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

steps_done = 0
FOLDER_NAME = './models'  # where to checkpoint the best models

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 600


config = configparser.ConfigParser()
config.read('config.ini')
WANDB_RUN_ID = config['DEFAULT']['WANDB_RUN_ID']
WANDB_RUN_NAME = config['DEFAULT']['WANDB_RUN_NAME']
WANDB_PROJECT_NAME = config['DEFAULT']['WANDB_PROJECT_NAME']
WANDB_CONFIG = {
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "architecture": "DQN",
    "dataset": "",
    "epochs": num_episodes,
}
SAVE_IN_WANDB = True
