import numpy as np
import pandas as pd
import random
import os
from collections import deque
import matplotlib.pyplot as plt

from envs.lstm_trading_env import StockTradingEnv
from models.dqn_model import build_dqn_model

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.97 
gamma = 0.99
batch_size = 32
memory_size = 100000
episodes = 300 
max_steps_per_episode = 1000
model_save_path = "models/dqn_stock_model.keras"

data = pd.read_csv("data/final_dataset.csv").reset_index(drop=True)

env = StockTradingEnv(data)
state_shape = env.observation_space.shape
action_shape = env.action_space.n

print(f"Observation shape: {state_shape}, Number of actions: {action_shape}")

q_model = build_dqn_model(state_shape, action_shape)
target_model = build_dqn_model(state_shape, action_shape)
target_model.set_weights(q_model.get_weights())

memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_batch():
    batch = random.sample(memory, batch_size)
    return map(np.array, zip(*batch))

def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(action_shape)
    q_values = q_model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])

def train_step():
    if len(memory) < batch_size:
        return
    states, actions, rewards, next_states, dones = sample_batch()

    next_q_values = target_model.predict(next_states, verbose=0)
    max_next_q = np.max(next_q_values, axis=1)

    target_q_values = q_model.predict(states, verbose=0)
    for i in range(batch_size):
        target_q_values[i][actions[i]] = (
            rewards[i] if dones[i] else rewards[i] + gamma * max_next_q[i]
        )

    q_model.fit(states, target_q_values, verbose=0)

reward_history = []
global_step = 0

print("Début de l'entraînement")

for episode in range(episodes):
    print(f"\n🎬 Épisode {episode}")
    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps_per_episode:
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        store_transition(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state
        train_step()

        step += 1
        global_step += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_model.set_weights(q_model.get_weights())
        print("🔁 Cible mise à jour")

    if episode % 50 == 0:
        q_model.save(f"models/dqn_model_ep{episode}.keras")
        print("💾 Modèle sauvegardé")

    reward_history.append(total_reward)
    print(f"✅ Episode {episode} terminé | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

os.makedirs("models", exist_ok=True)
q_model.save(model_save_path)
print(f"\n✅ Modèle final sauvegardé dans {model_save_path}")

plt.figure(figsize=(10, 5))
plt.plot(reward_history)
plt.title("Reward par épisode")
plt.xlabel("Épisode")
plt.ylabel("Total Reward")
plt.grid()
plt.tight_layout()
plt.show()
