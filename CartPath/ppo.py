# ppo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
import math
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2          # ε in the clipped objective
    ent_coef: float = 0.01           # entropy bonus coefficient
    vf_coef: float = 0.5             # value loss coefficient
    max_grad_norm: float = 0.5
    epochs: int = 10                 # K: policy epochs per update
    minibatch_size: int = 128
    value_clip: bool = True          # value clipping as in PPO2
    lr: float = 3e-4                 # Adam lr


def iterate_minibatches(total_size: int, batch_size: int) -> Iterable[torch.Tensor]:
    idx = torch.randperm(total_size)
    for start in range(0, total_size, batch_size):
        yield idx[start : start + batch_size]


def ppo_update(net, optimizer, batch: Dict[str, torch.Tensor], cfg: PPOConfig) -> Dict[str, float]:
    """
    One PPO update over (obs, acts, old_logps, adv, returns, old_values).
    Performs cfg.epochs passes over shuffled minibatches.
    Returns logging stats averaged across all minibatches.
    """
    obs   = batch["obs"]
    acts  = batch["acts"]
    old_logps = batch["logps"]
    adv   = batch["adv"]
    ret   = batch["ret"]
    old_vals = batch["vals"]

    n = obs.shape[0]
    stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0, clipfrac=0.0)

    for _ in range(cfg.epochs):
        for mb_idx in iterate_minibatches(n, cfg.minibatch_size):
            mb_obs   = obs[mb_idx]
            mb_acts  = acts[mb_idx]
            mb_old_logp = old_logps[mb_idx]
            mb_adv   = adv[mb_idx]
            mb_ret   = ret[mb_idx]
            mb_old_v = old_vals[mb_idx]

            # New log-probs, entropy, and values from current policy
            new_logp, entropy, new_v = net.evaluate_actions(mb_obs, mb_acts)

            # Policy ratio and clipped surrogate
            ratio = (new_logp - mb_old_logp).exp()
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # Approx KL for diagnostics
            with torch.no_grad():
                approx_kl = torch.mean(mb_old_logp - new_logp).clamp_min(0.0)

            # Value function loss (optionally clipped)
            if cfg.value_clip:
                v_clipped = mb_old_v + (new_v - mb_old_v).clamp(-cfg.clip_ratio, cfg.clip_ratio)
                v_loss_unclipped = (new_v - mb_ret).pow(2)
                v_loss_clipped = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
            else:
                value_loss = 0.5 * F.mse_loss(new_v, mb_ret)

            # Entropy bonus (maximize entropy -> subtract negative)
            entropy_loss = -torch.mean(entropy)

            loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Bookkeeping
            with torch.no_grad():
                clipped_frac = (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"]  += value_loss.item()
                stats["entropy"]     += (-entropy_loss).item()  # report positive entropy
                stats["kl"]          += approx_kl.item()
                stats["clipfrac"]    += clipped_frac

    # Average over number of minibatches processed
    num_updates = cfg.epochs * math.ceil(n / cfg.minibatch_size)
    for k in stats:
        stats[k] /= num_updates
    return stats
