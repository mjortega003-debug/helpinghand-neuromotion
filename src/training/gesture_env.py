import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import Counter
from signal_cleaner import SignalCleaner

class HandGestureEnv(gym.Env):
    def __init__(self, csv_path="logs/neuromotion_data.csv", window_size=50, stride=10):
        super(HandGestureEnv, self).__init__()
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['cv_label'] != 'none'].reset_index(drop=True)
        
        if self.df.empty:
            raise ValueError("CSV is empty or has no valid labeled data.")

        self.window_size = window_size  # 50 samples = 200ms
        self.stride = stride            
        self.cleaner = SignalCleaner(fs=250)
        self.feature_cols = [f"emg_ch{i+1}" for i in range(8)] + [f"eeg_ch{i+1}" for i in range(8)]
        self.labels = self.df['cv_label'].unique().tolist()
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.labels))
        
        self.current_step = 0

    def _get_obs(self):
        """Grabs a window of data, cleans it, and extracts features."""
        # Get raw window of 50 rows x 16 columns
        window_df = self.df.iloc[self.current_step : self.current_step + self.window_size]
        raw_window = window_df[self.feature_cols].values
        

        raw_window = raw_window.T 
        features = self.cleaner.preprocess_and_extract(raw_window)
        return features.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        window_labels = self.df.iloc[self.current_step : self.current_step + self.window_size]['cv_label'].tolist()
        true_label_str = Counter(window_labels).most_common(1)[0][0]
        true_action = self.label_map[true_label_str]

        # Reward Calculation
        if action == true_action:
            reward = 1.0
        else:
            reward = -1.0

        # Move forward by the stride
        self.current_step += self.stride
        
        done = (self.current_step + self.window_size) >= len(self.df)
        
        obs = np.zeros(16, dtype=np.float32) if done else self._get_obs()
        
        return obs, reward, done, False, {}