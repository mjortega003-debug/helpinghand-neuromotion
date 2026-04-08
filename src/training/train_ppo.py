from stable_baselines3 import PPO
from gesture_env import HandGestureEnv
import os

def train():
    # 1. Initialize the Environment
    env = HandGestureEnv(r"C:\Users\Arshi\Documents\GitHub\helpinghand-neuromotion\logs\neuromotion_data.csv")

    # 2. Define the PPO Model
    # MlpPolicy is used because our data is a flat vector of 16 channels
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu", # to use gpu change value to cuda
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )

    # 3. Train
    print("Starting PPO training...")
    model.learn(total_timesteps=100000)

    # 4. Save
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_gesture_hand")
    print("Model saved to models/ppo_gesture_hand")

if __name__ == "__main__":
    train()