# buffer.py
from __future__ import annotations
from typing import Dict, Tuple
import torch


def _discount_cumsum(x: torch.Tensor, discount: float) -> torch.Tensor:
    """
    Compute y[t] = x[t] + discount * x[t+1] + discount^2 * x[t+2] + ...
    Works on 1D tensors. Implemented via reverse scan for speed.
    """
    y = torch.zeros_like(x)
    running = 0.0
    for t in reversed(range(x.shape[0])):
        running = x[t] + discount * running
        y[t] = running
    return y


class RolloutBuffer:
    """
    Fixed-horizon on-policy buffer for PPO with GAE(λ).
    Stores one contiguous trajectory (or multiple episodes back-to-back)
    until it reaches `size`, then you call .finish_path(...) and .get().

    Usage:
      buf = RolloutBuffer(obs_dim, size=T, gamma=0.99, lam=0.95, device="cpu")
      buf.add(obs, act, rew, done, val, logp)   # repeat T times
      buf.finish_path(last_value)               # bootstrap if not done
      batch = buf.get()                         # tensors for PPO update
    """
    def __init__(self, obs_dim: int, size: int, gamma: float, lam: float, device="cpu"):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros(size, dtype=torch.long, device=device)
        self.rews = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)  # 1.0 if terminal/truncated
        self.vals = torch.zeros(size, dtype=torch.float32, device=device)
        self.logps = torch.zeros(size, dtype=torch.float32, device=device)

        self.adv = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret = torch.zeros(size, dtype=torch.float32, device=device)

        self.gamma = gamma
        self.lam = lam
        self.max_size = size
        self.device = device

        self.ptr = 0
        self.path_start = 0
        self._is_ready = False

    def add(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size, "RolloutBuffer full; call finish_path() then get()."
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.acts[self.ptr] = int(act)
        self.rews[self.ptr] = float(rew)
        self.dones[self.ptr] = float(done)  # treat truncation as done for advantage cutoff
        self.vals[self.ptr] = float(val)
        self.logps[self.ptr] = float(logp)
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0):
        """
        Compute GAE advantages and returns for the trajectory slice
        [path_start:ptr). If the trajectory reached a true terminal,
        pass last_value=0. If it ended due to time-limit (not done),
        pass critic bootstrap value as last_value.
        """
        assert self.ptr > self.path_start, "No steps to finish."
        idx = slice(self.path_start, self.ptr)

        vals = self.vals[idx]
        rews = self.rews[idx]
        dones = self.dones[idx]

        # v_next: values shifted left, with bootstrap at the end
        v_next = torch.cat([vals[1:], torch.as_tensor([last_value], device=self.device)])

        # TD residuals (stop-advantage at terminal steps)
        deltas = rews + self.gamma * v_next * (1.0 - dones) - vals

        # discounted cumulative sum with factor gamma*lambda, also stop at dones
        # Implement "stop at done" by zeroing running sum when done==1.
        adv = torch.zeros_like(deltas)
        running = 0.0
        for t in reversed(range(deltas.shape[0])):
            running = deltas[t] + self.gamma * self.lam * running * (1.0 - dones[t])
            adv[t] = running

        self.adv[idx] = adv
        self.ret[idx] = adv + vals

        self.path_start = self.ptr  # next path starts here

        # If we've filled the buffer, mark ready
        if self.ptr == self.max_size:
            self._is_ready = True

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Returns all data and resets the buffer pointer.
        Normalizes advantages (mean 0, std 1).
        """
        assert self._is_ready, "Buffer not ready; must fill and finish_path first."
        # Advantage normalization (helps PPO)
        adv = self.adv
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        batch = dict(
            obs=self.obs,
            acts=self.acts,
            logps=self.logps,
            adv=adv,
            ret=self.ret,
            vals=self.vals,
        )
        # Reset for next rollout
        self.ptr = 0
        self.path_start = 0
        self._is_ready = False
        return batch
