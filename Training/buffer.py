"""
Step 2: rollout storage and advantages.

PPO collects T steps from all N cars with the current policy, then learns from that batch and
throws it away. Everything is stored as (T, N, ...) arrays: time first, car second.
"""
import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, steps: int, num_agents: int, obs_dim: int, act_dim: int):
        T, N = steps, num_agents
        self.T, self.N = T, N
        self.obs = np.zeros((T, N, obs_dim), np.float32)
        self.actions = np.zeros((T, N, act_dim), np.float32)
        self.log_probs = np.zeros((T, N), np.float32)
        self.rewards = np.zeros((T, N), np.float32)
        self.dones = np.zeros((T, N), np.float32)    # 1 if the episode ended after this step
        self.values = np.zeros((T, N), np.float32)
        self.advantages = np.zeros((T, N), np.float32)
        self.returns = np.zeros((T, N), np.float32)
        self.t = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        """Store one step for all cars (each argument has a leading N dimension)."""
        self.obs[self.t] = obs
        self.actions[self.t] = actions
        self.log_probs[self.t] = log_probs
        self.rewards[self.t] = rewards
        self.dones[self.t] = dones
        self.values[self.t] = values
        self.t += 1

    def full(self) -> bool:
        return self.t >= self.T

    def compute_advantages(self, last_values: np.ndarray, gamma: float, lam: float):
        """
        Fill self.advantages and self.returns, both (T, N).

        last_values (N,): the critic's value of the observation AFTER the final stored step,
        used to "bootstrap" the rollouts that were cut off mid-episode.

        Generalised Advantage Estimation (Schulman et al. 2015, eq. 16), worked backwards in time:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
        Truncated episodes are handled in ppo.py: gamma * V(final_obs) is already folded into their
        last reward, so here every done simply cuts the future off.
        """
        gae = np.zeros(self.N, np.float32)
        for t in reversed(range(self.T)):
            next_values = last_values if t == self.T - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
        self.returns[:] = self.advantages + self.values

    def minibatches(self, batch_size: int, device):
        """Yields shuffled minibatches of the flattened (T*N) data as torch tensors.  DONE."""
        flat = {
            "obs": self.obs.reshape(self.T * self.N, -1),
            "actions": self.actions.reshape(self.T * self.N, -1),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }
        idx = np.random.permutation(self.T * self.N)
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            yield {k: torch.as_tensor(v[b], device=device) for k, v in flat.items()}

    def reset(self):
        self.t = 0
