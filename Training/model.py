"""
Step 1: the actor-critic network.

The policy (actor) maps an observation to a probability distribution over actions; the critic maps
an observation to a single number, the estimated return (value). Both see the same 44 inputs.

Shapes used everywhere:  B = batch size,  obs (B, 44),  actions (B, 2).
"""
import torch
import torch.nn as nn


def _layer(in_dim: int, out_dim: int, gain: float = 2 ** 0.5) -> nn.Linear:
    """Linear layer with orthogonal weights and zero bias."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 44, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        # separate actor and critic MLPs; the tiny final gain makes initial means ~0 (near-random driving)
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, act_dim, gain=0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, gain=1.0),
        )
        # state-independent log std: any real number gives a valid std = exp(log_std) > 0
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, obs_dim) -> value (B,)."""
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """
        obs (B, obs_dim). If `action` is None, sample a new one from the policy; otherwise evaluate
        the given one (you need that during the PPO update).
        Returns:
            action    (B, act_dim)   NOT clipped: clip only when sending to the game
            log_prob  (B,)           log pi(action | obs), summed over the action dimensions
            entropy   (B,)           entropy of the action distribution
            value     (B,)           critic's estimate for obs
        """
        mean = self.actor(obs)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.get_value(obs)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, obs_dim) -> the action you'd take without exploration noise, (B, act_dim) in [-1, 1].
        Used by play.py and when exporting the model to Unity."""
        return self.actor(obs).clamp(-1.0, 1.0)
