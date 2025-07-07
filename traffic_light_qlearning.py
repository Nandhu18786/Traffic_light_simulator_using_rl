


import os
import numpy as np
import traci
import gym
from gym import spaces
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------ Parameters ------------------
JUNCTION_ID = "10016346056"
TL_IDS = [
    "11283488131", "11434604026", "11283488139", "10016346057", "1232963680",
    "12120691272", "6584009910", "6584009911", "4562270105", "4562270106",
    "4562270099", "4562270100", "3305876073", "3305876069"
]
PHASES = 3
CONTROLLED_EDGES = ["766406224#1", "1308654687#1", "915684608#2",":7676373070_5",":3305876071_0"]
SUMO_CONFIG = "real_traffic.sumocfg"

ALPHA = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 3000
EPISODES = 5  # Increased for better learning

WAITING_TIME_THRESHOLD = 30  # Not used now
EXTRA_PENALTY = 0  # Removed extra penalty

# ------------------ DQN Model ------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ SUMO Environment ------------------
class TrafficLightEnv(gym.Env):
    def __init__(self):
        super(TrafficLightEnv, self).__init__()
        self.sumo_binary = "sumo-gui"
        self.sumo_config = SUMO_CONFIG
        self.action_space = spaces.Discrete(PHASES * len(TL_IDS))
        self.observation_space = spaces.Box(low=0, high=100, shape=(len(CONTROLLED_EDGES),), dtype=np.float32)
        self.prev_total_wait = 0
        print(f"Managing junction ID: {JUNCTION_ID}")
        print(f"Controlling {len(TL_IDS)} traffic lights: {TL_IDS}")
        self.start_sumo()

    def start_sumo(self):
        if traci.isLoaded():
            traci.close()
        traci.start([self.sumo_binary, "-c", self.sumo_config])

    def step(self, action):
        for idx, tl_id in enumerate(TL_IDS):
            phase = action % PHASES
            traci.trafficlight.setPhase(tl_id, phase)

        for _ in range(5):  # Reduced to 5 steps per action
            traci.simulationStep()

        state = self.get_state()
        waiting_times = [traci.edge.getWaitingTime(e) for e in CONTROLLED_EDGES]
        total_wait = sum(waiting_times)

        # Normalized reward
        reward = - (total_wait / len(CONTROLLED_EDGES))

        # Bonus for improvement
        if total_wait < self.prev_total_wait:
            reward += 5

        self.prev_total_wait = total_wait
        done = traci.simulation.getMinExpectedNumber() == 0

        return state, reward, done, {}

    def reset(self):
        self.start_sumo()
        self.prev_total_wait = 0
        return self.get_state()

    def get_state(self):
        return np.array(
            [min(99, traci.edge.getLastStepVehicleNumber(e)) for e in CONTROLLED_EDGES],
            dtype=np.float32
        )

    def close(self):
        if traci.isLoaded():
            traci.close()

# ------------------ Agent ------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.loss_fn = nn.MSELoss()

        self.epsilon = EPSILON

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target += GAMMA * torch.max(self.model(torch.FloatTensor(next_state))) #bellman equation

            output = self.model(torch.FloatTensor(state))[action]
            loss = self.loss_fn(output, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save_model(self, path="dqn_real_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="dqn_real_model.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# ------------------ Training ------------------
def moving_average(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    env = TrafficLightEnv()
    agent = DQNAgent(state_size=len(CONTROLLED_EDGES), action_size=PHASES * len(TL_IDS))

    rewards_per_episode = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            episode_reward += reward

        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode}: Total Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

    env.close()
    agent.save_model()
    print("Training completed. Model saved.")

    # Plot rewards with moving average
    plt.plot(rewards_per_episode, label='Reward')
    if len(rewards_per_episode) >= 5:
        plt.plot(moving_average(rewards_per_episode), label='Moving Avg (5)', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_per_episode_fixed.png")
    plt.show()
