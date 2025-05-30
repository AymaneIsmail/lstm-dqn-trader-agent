
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.lstm_trading_env import StockTradingEnv
# Création du dataset factice
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'Open': np.random.rand(10),
    'High': np.random.rand(10),
    'Low': np.random.rand(10),
    'Close': np.linspace(100, 110, 10),
    'Volume': np.random.randint(100, 1000, size=10)

})

# Instanciation de l'environnement
env = StockTradingEnv(data=data)

obs = env.reset()
print("Observation initiale :", obs)

done = False
step = 0

while not done and step < 10:
    action = env.action_space.sample()  # Action aléatoire: hold/buy/sell
    obs, reward, done, info = env.step(action)

    print(f"\nStep {step+1}")
    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    step += 1

env.close()
