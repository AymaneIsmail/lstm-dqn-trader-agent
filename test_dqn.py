import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from envs.lstm_trading_env import StockTradingEnv  # ton environnement custom

# Charger les données
data = pd.read_csv("data/final_dataset.csv").reset_index(drop=True)

# Charger le modèle DQN entraîné
model = load_model("models/dqn_stock_model.keras")

env = StockTradingEnv(data)
state = env.reset()

done = False
total_reward = 0
step = 0

while not done:
    q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(q_values[0])  # meilleure action
    next_state, reward, done, info = env.step(action)
    env.render() 
    total_reward += reward
    state = next_state
    step += 1

print(f"\nTest terminé en {step} steps.")
print(f"Total net worth final : {info['net_worth']:.2f}")
print(f"Total reward obtenu : {total_reward:.2f}")
