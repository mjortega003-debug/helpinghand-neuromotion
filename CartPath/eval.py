# eval.py
import time
import torch
import gymnasium as gym
from nets import ActorCritic

def eval_render(episodes=5, seed=0, fps=60):
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCritic(obs_dim, act_dim, hidden=(64, 64))
    net.load_state_dict(torch.load("policy_cartpole.pt", map_location="cpu"))
    net.eval()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return, ep_len = 0.0, 0

        while not done:
            # Deterministic action: argmax over policy probs
            with torch.no_grad():
                dist, _ = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = dist.probs.argmax(dim=-1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_len += 1

            # Keep it human-speed (optional)
            time.sleep(1.0 / fps)

        print(f"Episode {ep+1}: return={ep_return:.0f}, length={ep_len}")
    env.close()

if __name__ == "__main__":
    eval_render()
