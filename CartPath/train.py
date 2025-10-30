# train.py
import time
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from nets import ActorCritic
from buffer import RolloutBuffer
from ppo import PPOConfig, ppo_update

def make_env(seed=0):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def train(
    total_timesteps=200_000,
    rollout_size=2048,
    gamma=0.99,
    lam=0.95,
    seed=1,
):
    device = torch.device("cpu")
    env = make_env(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCritic(obs_dim, act_dim, hidden=(64, 64)).to(device)
    cfg = PPOConfig()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    buf = RolloutBuffer(obs_dim, size=rollout_size, gamma=gamma, lam=lam, device=device)
    writer = SummaryWriter()

    obs, _ = env.reset(seed=seed)
    ep_return, ep_len, global_step, update_idx = 0.0, 0, 0, 0
    start_time = time.time()

    while global_step < total_timesteps:
        # ===== Collect one rollout =====
        last_done = False
        for t in range(rollout_size):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, value = net.act(obs_t)
            action = int(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buf.add(obs, action, reward, done, value.item(), logp.item())

            ep_return += reward
            ep_len += 1
            global_step += 1
            last_done = done
            obs = next_obs

            if done:
                writer.add_scalar("charts/episode_return", ep_return, global_step)
                writer.add_scalar("charts/episode_length", ep_len, global_step)
                obs, _ = env.reset()
                ep_return, ep_len = 0.0, 0

        # Bootstrap value if rollout ended mid-episode
        with torch.no_grad():
            v_boot = 0.0
            if not last_done:
                v_boot = net(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))[1].item()
        buf.finish_path(last_value=v_boot)

        # ===== PPO update =====
        batch = buf.get()
        for k in batch:
            batch[k] = batch[k].to(device)

        stats = ppo_update(net, optimizer, batch, cfg)
        update_idx += 1

        # ===== Logging =====
        writer.add_scalar("losses/policy_loss", stats["policy_loss"], global_step)
        writer.add_scalar("losses/value_loss",  stats["value_loss"],  global_step)
        writer.add_scalar("stats/entropy",      stats["entropy"],     global_step)
        writer.add_scalar("stats/kl",           stats["kl"],          global_step)
        writer.add_scalar("stats/clipfrac",     stats["clipfrac"],    global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        print(f"update {update_idx:04d} | step {global_step:7d} | "
              f"ret(last ep) {ep_return:7.2f} | pol {stats['policy_loss']:.3f} | "
              f"val {stats['value_loss']:.3f} | ent {stats['entropy']:.3f} | kl {stats['kl']:.4f}")
        torch.save(net.state_dict(), "policy_cartpole.pt")
        print("Saved model to policy_cartpole.pt")
    env.close()
    writer.close()

if __name__ == "__main__":
    train()
