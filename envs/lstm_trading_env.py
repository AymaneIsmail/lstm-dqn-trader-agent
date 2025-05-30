import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance=10000.0):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

        self.max_steps = len(data) - 1

        self.numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.numeric_cols),),
            dtype=np.float32
        )

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        obs = self.data.loc[self.current_step, self.numeric_cols].values.astype(np.float32)
        return obs

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0

        self.previous_net_worth = self.initial_balance

        return self._get_obs()

    def step(self, action):
        current_price = self.data.loc[self.current_step, "Close"]

        # Achat
        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price

        # Vente
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price

        self.current_step += 1
        done = self.current_step >= self.max_steps or self.current_step >= len(self.data) - 1

        self.net_worth = self.balance + self.shares_held * current_price

        reward = self.net_worth - self.previous_net_worth
        self.previous_net_worth = self.net_worth

        obs = self._get_obs()
        info = {"net_worth": self.net_worth}

        return obs, reward, done, info


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Net worth: {self.net_worth:.2f}")
