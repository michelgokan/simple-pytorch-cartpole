# A simple pytorch implementation of the cartpole problem

## 1. Introduction

This is a simple pytorch implementation of the cartpole problem. The goal is to train a neural network to balance a pole on a cart. The cart can move left or right. The pole starts upright, and the goal is to prevent it from falling over. 
The state of the system is represented by four numbers: the position and velocity of the cart, and the angle and angular velocity of the pole. The action is either to move the cart left or right. 

More info: https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space

The reward is 1 for every time step that the pole remains upright. 
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.


### Execution
If you decided to use this repository, follow these steps:

* copy/paste config.ini.sample into config.ini and replace variables in it
* run followings:
  ```bash
  pip3.11 install -r requirements.txt 
    ```
* Follow steps in ["Weights & Biases quickstart page"](https://docs.wandb.ai/quickstart)
    ```bash
  wandb login
  ```
* To run the training:
  ```
  python3.11 train.py
  ```
* To use the trained model to find the shortest path:
  ```bash
  python3.11 execute.py 
  ```
