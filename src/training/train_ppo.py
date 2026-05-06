from stable_baselines3 import PPO
from gesture_env import HandGestureEnv
import os

# Entry point for training the PPO gesture classification model.
# Requires a pre-recorded neuromotion CSV and a configured HandGestureEnv.
def train():
    # 1. Initialize the Environment
    # Update this path if the dataset moves — HandGestureEnv wraps the CSV as a Gym env
    env = HandGestureEnv(r"C:\Users\Arshi\Documents\GitHub\helpinghand-neuromotion\logs\neuromotion_data.csv")

    # 2. Define the PPO Model
    # MlpPolicy is used because our data is a flat vector of 16 channels
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu",       # Switch to "cuda" if a compatible GPU is available
        learning_rate=3e-4, # Default PPO lr; lower if training is unstable
        n_steps=2048,       # Rollout buffer size before each policy update
        batch_size=64,      # Minibatch size for gradient updates
        n_epochs=10         # Number of passes over the rollout buffer per update
    )

    # 3. Train
    # Increase total_timesteps for better convergence on larger datasets
    print("Starting PPO training...")
    model.learn(total_timesteps=100000)

    # 4. Save
    # Saves as a .zip file — load later with PPO.load("models/ppo_gesture_hand")
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_gesture_hand")
    print("Model saved to models/ppo_gesture_hand")

if __name__ == "__main__":
    train()