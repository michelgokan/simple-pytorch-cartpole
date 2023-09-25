from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb

import variables as v
from helpers import select_action, optimize_model, episode_durations, plot_durations, checkpoint_model
from memory import ReplayMemory
from qnet import DQN

policy_net = DQN(v.n_observations, v.n_actions).to(v.device)
target_net = DQN(v.n_observations, v.n_actions).to(v.device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=v.LR, amsgrad=True)
memory = ReplayMemory(10000)

# keep track of median path length for model checkpointing
current_min_med_time = float('inf')

if v.SAVE_IN_WANDB:
    wandb.init(project=v.WANDB_PROJECT_NAME, name=v.WANDB_RUN_NAME, config=v.WANDB_CONFIG)

for i_episode in range(v.num_episodes):
    # Initialize the environment and get it's state
    state, info = v.env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=v.device).unsqueeze(0)
    for t in count():
        action = select_action(state, policy_net)
        observation, reward, terminated, truncated, _ = v.env.step(action.item())
        reward = torch.tensor([reward], device=v.device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=v.device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model(memory, policy_net, target_net, optimizer)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*v.TAU + target_net_state_dict[key]*(1-v.TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            med_duration = np.median(episode_durations[-100:])

            if med_duration < current_min_med_time:
                current_min_med_time = t
                checkpoint_model(target_net, optimizer, loss, i_episode, t)
            episode_durations.append(t + 1)
            wandb.log(
                {"loss": (-1 if loss is None else loss), "median duration": np.median(episode_durations[-50:]),
                 "duration": t})

            #     plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
