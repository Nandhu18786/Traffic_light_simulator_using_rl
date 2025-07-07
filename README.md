# DQN-based Traffic Light Control for SUMO

This project implements a Deep Q-Network (DQN) agent to control a complex traffic intersection in a [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) environment. The goal of the agent is to learn an optimal traffic light control policy to minimize vehicle waiting times and improve traffic flow.

## Table of Contents
- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
  - [SUMO Environment](#sumo-environment)
  - [State Representation](#state-representation)
  - [Action Space](#action-space)
  - [Reward Function](#reward-function)
  - [DQN Agent](#dqn-agent)
- [Core Technologies](#core-technologies)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Parameters](#configuration-parameters)
- [Outputs](#outputs)
- [Potential Improvements](#potential-improvements)

## Project Overview

The script `traffic_light_qlearning.py` sets up a Reinforcement Learning pipeline where a DQN agent interacts with a SUMO traffic simulation. The agent observes the traffic state (number of vehicles on key roads) and takes actions (changing traffic light phases) to maximize a cumulative reward, which is designed to be inversely proportional to the total vehicle waiting time.

By training over multiple episodes, the agent learns to make intelligent decisions to manage traffic congestion effectively.

## How It Works

The system is composed of two main components: a custom OpenAI Gym environment that wraps the SUMO simulation and a DQN agent that learns from it.

### SUMO Environment
The `TrafficLightEnv` class creates an interface between the agent and the SUMO simulation.
- It launches the SUMO GUI using a specified configuration file (`real_traffic.sumocfg`).
- It handles the connection to the simulation via the **TraCI (Traffic Control Interface)**.
- It defines the `step`, `reset`, and `close` methods required by the Gym API.

### State Representation
The **state** is a snapshot of the traffic conditions. It is defined as a NumPy array containing the number of vehicles on a predefined set of `CONTROLLED_EDGES`. This gives the agent a quantitative measure of congestion at critical approaches to the intersection.

```python
# State is the number of vehicles on each controlled edge
def get_state(self):
    return np.array(
        [min(99, traci.edge.getLastStepVehicleNumber(e)) for e in CONTROLLED_EDGES],
        dtype=np.float32
    )
```

### Action Space
The **action** determines the phase of the traffic lights. The current implementation uses a simplified action space. The agent selects a single integer, and the phase for *all* controlled traffic lights is set to `action % PHASES`.

```python
# All traffic lights are set to the same phase
def step(self, action):
    for idx, tl_id in enumerate(TL_IDS):
        phase = action % PHASES
        traci.trafficlight.setPhase(tl_id, phase)
```
**Note:** This is a major simplification. See [Potential Improvements](#potential-improvements) for more details.

### Reward Function
The **reward** signal guides the agent's learning process. The reward function is designed to penalize waiting time:
- The base reward is the negative average waiting time across all `CONTROLLED_EDGES`.
- A bonus reward of `+5` is given if the total waiting time decreases compared to the previous action, incentivizing continuous improvement.

```python
# Reward is based on negative waiting time + a bonus for improvement
total_wait = sum([traci.edge.getWaitingTime(e) for e in CONTROLLED_EDGES])
reward = - (total_wait / len(CONTROLLED_EDGES))
if total_wait < self.prev_total_wait:
    reward += 5
```

### DQN Agent
The `DQNAgent` class implements the Deep Q-Learning algorithm.
- **Model**: A simple feed-forward neural network built with PyTorch is used to approximate the Q-function (the expected future reward for taking an action in a given state).
- **Epsilon-Greedy Policy**: To balance exploration and exploitation, the agent sometimes chooses a random action (exploration) and sometimes chooses the best-known action based on its model's prediction (exploitation). The probability of choosing a random action (`epsilon`) decreases over time.
- **Replay Memory**: The agent stores its experiences (state, action, reward, next_state) in a `deque`. During training, it samples random minibatches from this memory to update its neural network. This technique, called experience replay, stabilizes and improves learning.

## Core Technologies
- **Python 3.x**
- **SUMO**: Traffic simulation suite.
- **TraCI**: Python API for controlling SUMO.
- **PyTorch**: Deep learning framework for the DQN model.
- **OpenAI Gym**: Toolkit for developing and comparing reinforcement learning algorithms.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting the training results.

## Prerequisites
1.  **SUMO**: You must have SUMO installed and the `sumo-gui` executable available in your system's PATH. You can find installation instructions on the [official SUMO documentation](https://sumo.dlr.de/docs/Installing/index.html).
2.  **Python 3.8+**
3.  **SUMO Scenario Files**: A valid SUMO configuration file (e.g., `real_traffic.sumocfg`) and all its associated files (network, routes, etc.) must be present in the same directory as the script.

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install numpy torch gym matplotlib
    ```
    You will also need the `traci` Python library, which is typically included with your SUMO installation. Ensure it is accessible by your Python interpreter.

## Usage

To start the training process, run the script from your terminal:
```bash
python traffic_light_qlearning.py
```
This will:
1.  Launch the SUMO GUI and start the simulation.
2.  Initialize the DQN agent.
3.  Run the training loop for the number of episodes specified in `EPISODES`.
4.  Print the total reward for each episode to the console.
5.  Save the trained model and the rewards plot upon completion.

## Configuration Parameters
You can modify the agent's behavior and the simulation setup by changing the parameters at the top of `traffic_light_qlearning.py`:

- `JUNCTION_ID`, `TL_IDS`, `CONTROLLED_EDGES`: SUMO-specific IDs for the intersection and roads being controlled.
- `PHASES`: The number of distinct phases for the traffic lights.
- `SUMO_CONFIG`: The path to your SUMO configuration file.
- `ALPHA`, `GAMMA`, `EPSILON`, etc.: Hyperparameters for the DQN algorithm (learning rate, discount factor, exploration rate, etc.).
- `BATCH_SIZE`, `MEMORY_SIZE`: Parameters for the experience replay memory.
- `EPISODES`: The total number of simulation runs for training.

## Outputs
After the script finishes, it will generate two files:
1.  `dqn_real_model.pth`: The saved state dictionary of the trained PyTorch model. This can be loaded later for inference or further training.
2.  `reward_per_episode_fixed.png`: A plot showing the total reward obtained in each episode, along with a 5-episode moving average to visualize the learning trend.

- **Reward Function**: The reward function could be tuned or expanded to consider other metrics like vehicle throughput, emergency vehicle priority, or fuel consumption.
- **Hyperparameter Tuning**: The performance is sensitive to hyperparameters like `ALPHA`, `GAMMA`, and `EPSILON_DECAY`. Using techniques like Grid Search or Bayesian Optimization could find a better set of values.
- **Advanced RL Algorithms**: More modern algorithms like Double DQN, Dueling DQN, or Proximal Policy Optimization (PPO) could lead to faster and more stable learning.
