# Learning PPO with Track Rush

A step-by-step path from "what is RL" to a PPO agent that drives your stunt tracks. Each step
teaches one idea, then has you implement it in the skeleton, then gives you a check so you know
it works before moving on.

**How to use this file**
- Do the steps in order. Each builds on the previous one.
- Every step ends with a **check**. Don't move on until it passes.
- Hints are folded inside `▶ Hint` blocks. Try without them first, then open one hint at a time.
- Write your answers to the **Think** questions somewhere. They are the actual learning.

Reading to keep open:
- PPO paper: Schulman et al. 2017, *Proximal Policy Optimization Algorithms* (arXiv 1707.06347)
- GAE paper: Schulman et al. 2015, *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (arXiv 1506.02438)
- *The 37 Implementation Details of PPO* (Huang et al., ICLR blog track 2022), for when things work but not well
- OpenAI *Spinning Up*, "Part 3: Intro to Policy Optimization", the clearest derivation of the policy gradient

---

## Step 0: Set up and meet the environment

```bash
cd ~/Desktop/unity_projects/AI_Playground/Training
source .venv/bin/activate
python smoke_test.py
```

You should see 40 cars, the 44 labelled observations of car 0, and some episodes ending.

### The RL framing

Reinforcement learning is a loop:

```
          observation s_t, reward r_t
   Game  ───────────────────────────►  Agent (your policy)
         ◄───────────────────────────
                  action a_t
```

| RL word | In Track Rush |
|---|---|
| **State / observation** $s_t$ | the 44 numbers (`trackrush_env.OBS_NAMES`) |
| **Action** $a_t$ | 2 numbers: steer, pedal, each in $[-1, 1]$ |
| **Reward** $r_t$ | progress along the track, checkpoints, finish, minus penalties |
| **Episode** | one attempt, from spawn until finish, fail or time limit |
| **Policy** $\pi(a \mid s)$ | your neural network: given $s$, a probability distribution over actions |
| **Step** | one decision; the game runs 5 physics steps (1/12 s) between decisions |

The goal is a policy that maximises the **return**, the discounted sum of future rewards:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

The **discount** $\gamma \in [0, 1)$ (we use 0.99) makes near rewards count more than far ones and
keeps the sum finite.

### Exercise 0.1: feel the rewards with your own hands
1. In Unity, open `Assets/Scenes/Training.unity` and select the **Training Manager** object.
2. Tick **Keyboard Test** and press Play. You now drive agent 0 with WASD.
3. Watch the reward in the top-left overlay as you drive well, scrape a wall, sit on the grass,
   drive backwards and finish.
4. Untick Keyboard Test when you're done.

**Think**
1. Driving 100 m forward gives +2. A checkpoint gives +1. Hitting a wall costs 0.3. Which of these
   will dominate what the agent learns early on? Why?
2. With $\gamma = 0.99$, a reward 100 steps (about 8 s) in the future is worth $0.99^{100} \approx 0.37$
   of a reward now. Roughly how far ahead does the agent "care"? Is that enough to plan for a jump?
3. Why does the agent see the road *ahead* instead of just its current position?

---

## Step 1: Policy gradients, the idea PPO is built on

We want to change the network's weights $\theta$ to make high-return actions more likely. The
**policy gradient theorem** says the direction to move is:

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t\Big]$$

Read it as: "for every action taken, push its log-probability up, weighted by how good things
turned out afterwards." Actions followed by high returns become more likely; actions followed by
low returns become less likely.

This raw version (REINFORCE) works in principle but is extremely noisy, because $G_t$ swings wildly
from episode to episode. Two fixes give us modern methods.

### Fix 1: the advantage
Subtract a **baseline**, the value of the state, which is how good things usually are from here:

$$V(s_t) \approx \mathbb{E}[G_t \mid s_t] \qquad A_t = G_t - V(s_t)$$

The **advantage** $A_t$ answers "was this action *better or worse than usual*?" rather than "was
the outcome good?". Being on a long straight is always good; what matters is whether *this action*
beat what we normally do there. The gradient stays correct on average but has far less variance.

A second network head, the **critic**, learns $V(s)$. The policy is the **actor**. Together they
make an **actor-critic**.

### Fix 2: don't take steps that are too big (this is PPO)
The gradient is only accurate near the current policy. One big update can wreck a policy that took
hours to learn, and then the data it collects is garbage, so it can't recover. PPO limits how far
each update can move the policy. That's Step 4.

### Continuous actions: a Gaussian policy
Our actions are real numbers, so the policy outputs a **Normal distribution** per action dimension:
- a **mean** $\mu_\theta(s)$, produced by the network
- a **standard deviation** $\sigma$, the amount of exploration noise

To act, sample $a \sim \mathcal{N}(\mu_\theta(s), \sigma^2)$. While training, the noise is how the
agent tries new things. In the game you use the mean (`act_deterministic`).

For 2 independent dimensions, the log-probability is the **sum** of the per-dimension
log-probabilities. The **entropy** (how spread out the distribution is) is also a sum, and for a
Gaussian it depends only on $\sigma$.

**Think**
1. Why do we use $\log \pi$ in the gradient instead of $\pi$? (Hint: what is $\nabla \log x$?)
2. If $\sigma$ shrinks to almost 0 early on, what happens to exploration?
3. Why can't the advantage simply be $r_t$, the reward of one step?

---

## Step 2: Build the actor-critic (`model.py`)

Implement the four methods of `ActorCritic`. Shapes are in the docstrings: `B` is batch size,
obs `(B, 44)`, actions `(B, 2)`.

1. **`__init__`**: an actor MLP (44 → hidden → hidden → 2, outputting the mean), a critic MLP
   (44 → hidden → hidden → 1), and a learnable **log** standard deviation of shape `(2,)`.
2. **`get_value(obs)`**: the critic's output, squeezed to `(B,)`.
3. **`get_action_and_value(obs, action=None)`**: build the Normal distribution; sample if no
   action is given; return action, log-prob (summed over the 2 dimensions), entropy (summed), value.
4. **`act_deterministic(obs)`**: the mean, clamped to $[-1, 1]$.

<details><summary>▶ Hint 1: why log std?</summary>

$\sigma$ must be positive. If you learn $\log\sigma$ and use $\sigma = e^{\log\sigma}$, any real
number is valid, so the optimiser can't produce an invalid negative $\sigma$. Starting $\log\sigma$
around $-0.5$ (so $\sigma \approx 0.6$) is a reasonable amount of initial exploration for actions in $[-1, 1]$.
</details>

<details><summary>▶ Hint 2: the distribution</summary>

`torch.distributions.Normal(mean, std)` broadcasts: `mean` is `(B, 2)` and `std` can be `(2,)`.
`.sample()`, `.log_prob(a)` (shape `(B, 2)`, so sum over the last dimension), and `.entropy()`.
</details>

<details><summary>▶ Hint 3: initialisation that tends to work</summary>

Tanh activations, orthogonal weight init with gain $\sqrt 2$ for hidden layers, a **tiny** gain
(0.01) on the actor's last layer (so initial means are near 0, which is near-random driving), and
gain 1 on the critic's last layer.
</details>

### Check 2
```python
import torch
from model import ActorCritic

m = ActorCritic(44, 2)
obs = torch.randn(5, 44)
a, lp, ent, v = m.get_action_and_value(obs)
assert a.shape == (5, 2) and lp.shape == (5,) and ent.shape == (5,) and v.shape == (5,)
_, lp2, _, _ = m.get_action_and_value(obs, a)          # evaluating the same action...
assert torch.allclose(lp, lp2)                          # ...gives the same log-prob
assert m.act_deterministic(obs).abs().max() <= 1
assert m.get_value(obs).shape == (5,)
print("model OK, entropy per sample:", ent[0].item())
```
For 2 dimensions with $\sigma \approx 0.6$, entropy should be about 1.8. With $\sigma = 1$ it's about 2.84.

---

## Step 3: Advantages with GAE (`buffer.py`)

PPO collects a **rollout**: $T$ steps from all $N$ cars (here $T = 256$, $N = 40$), stored as
`(T, N)` arrays. Now you turn rewards and values into advantages.

### TD error: a one-step advantage estimate
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
"What I got this step, plus what I expect afterwards, minus what I expected before." Low variance,
but biased, because $V$ is only an estimate.

### Monte Carlo: the full return
$A_t = G_t - V(s_t)$ uses real rewards to the end of the episode. Unbiased, but high variance.

### GAE blends the two
$$A_t = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \dots$$
With $\lambda = 0$ this is the TD error; with $\lambda = 1$ it is Monte Carlo. $\lambda = 0.95$ is
the usual sweet spot. It is cheap to compute **backwards** in time:
$$A_t = \delta_t + \gamma\lambda\,(1 - d_t)\,A_{t+1}$$
where $d_t = 1$ if the episode ended after step $t$. That $(1 - d_t)$ stops the next episode's
future leaking into this one. Similarly, $V(s_{t+1})$ inside $\delta_t$ must be zeroed when $d_t = 1$.

Then the critic's training target is
$$R_t = A_t + V(s_t).$$

**What's $V(s_{t+1})$ at the last step $T-1$?** The rollout stops mid-episode for most cars. You
pass `last_values`, the critic's value of the observation after the last step, to **bootstrap**.

Implement `compute_advantages(last_values, gamma, lam)`, filling `self.advantages` and `self.returns`.

<details><summary>▶ Hint: the loop structure</summary>

Keep a running `last_adv` of shape `(N,)`, starting at zeros. Loop `t` from `T-1` down to `0`.
The "next value" is `last_values` at `t = T-1`, otherwise `self.values[t+1]`. Build a mask
`1 - self.dones[t]`, compute `delta`, update `last_adv`, store it. Everything is vectorised over
the N cars, so there's no loop over cars.
</details>

### Check 3: compute these by hand first, then test your code
```python
import numpy as np
from buffer import RolloutBuffer

b = RolloutBuffer(steps=3, num_agents=1, obs_dim=1, act_dim=1)
b.rewards[:, 0] = [1, 2, 3]
b.values[:, 0] = [0.5, 0.5, 0.5]

b.dones[:, 0] = [0, 0, 0]
b.compute_advantages(np.array([1.0]), gamma=1.0, lam=1.0)
print(b.advantages[:, 0])   # expect [6.5, 5.5, 3.5]
print(b.returns[:, 0])      # expect [7.0, 6.0, 4.0]

b.dones[:, 0] = [0, 1, 0]   # an episode ended after step 1
b.compute_advantages(np.array([1.0]), gamma=1.0, lam=1.0)
print(b.advantages[:, 0])   # expect [2.5, 1.5, 3.5]

b2 = RolloutBuffer(steps=2, num_agents=1, obs_dim=1, act_dim=1)
b2.rewards[:, 0] = [1, 1]
b2.compute_advantages(np.array([0.0]), gamma=0.5, lam=0.5)
print(b2.advantages[:, 0])  # expect [1.25, 1.0]
```

**Think**
1. In the second test, why did step 2's advantage stay at 3.5 while steps 0 and 1 changed?
2. What goes wrong if you forget the $(1 - d_t)$ masks?
3. Why is $V(s)$ subtracted inside $\delta$ but added back for the returns?

---

## Step 4: The PPO update (`ppo.py`, function `ppo_update`)

### The problem PPO solves
After collecting data with the old policy $\pi_{\text{old}}$, we want to improve the policy using
that data several times over (it was expensive to collect). But the further the new policy drifts
from the one that collected the data, the less that data tells us.

### The probability ratio
$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} = \exp\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{old}}(a_t|s_t)\big)$$
This is 1 before any update. Values above 1 mean the new policy likes this action more than the
old one did.

### The clipped surrogate objective (PPO paper, eq. 7)
$$L^{\text{CLIP}} = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\; \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big]$$
with $\epsilon = 0.2$. We **maximise** this. What the min and clip do:

| Advantage | Ratio | What happens |
|---|---|---|
| $A > 0$ (good action) | $\rho < 1 + \epsilon$ | normal gradient: make it more likely |
| $A > 0$ | $\rho > 1 + \epsilon$ | clipped: no further push, it's already much more likely |
| $A < 0$ (bad action) | $\rho > 1 - \epsilon$ | normal gradient: make it less likely |
| $A < 0$ | $\rho < 1 - \epsilon$ | clipped: no further push |

So each update can only move each action's probability by about $\pm 20\%$ before the incentive
disappears. That's the "proximal" in PPO.

### The full loss (minimised)
$$\mathcal{L} = -L^{\text{CLIP}} \;+\; c_v \cdot \tfrac12\big(V_\theta(s_t) - R_t\big)^2 \;-\; c_e \cdot \text{entropy}$$
- Negative $L^{\text{CLIP}}$, because optimisers minimise.
- **Value loss**, so the critic tracks the returns. $c_v = 0.5$.
- **Entropy bonus**, so exploration doesn't collapse too early. $c_e = 0.005$.

### The procedure
For `cfg.epochs` passes over the batch, for each shuffled minibatch (`buffer.minibatches`):
1. Re-evaluate the stored actions under the current network to get new log-probs, entropy and values.
2. Compute the ratio, the clipped objective, the value loss and the entropy, and combine them.
3. `optimizer.zero_grad()`, `loss.backward()`, clip the gradient norm to 0.5, `optimizer.step()`.

Also standard: **normalise the advantages** in each minibatch (subtract the mean, divide by the std).

### Diagnostics to return and log
- **approx_kl**, about how far the policy moved: `((ratio - 1) - log_ratio).mean()`. Healthy: 0.005–0.03.
- **clip_frac**, the fraction of samples where clipping was active. Healthy: 0.05–0.3.
- the policy loss, value loss and entropy.

<details><summary>▶ Hint: the loss lines in words</summary>

`log_ratio = new_log_prob - old_log_prob`, `ratio = exp(log_ratio)`.
Two surrogates: `ratio * adv` and `clamp(ratio, 1-eps, 1+eps) * adv`.
Policy loss: `-min(surr1, surr2).mean()`.
Value loss: `0.5 * ((new_value - returns) ** 2).mean()`.
Loss: `policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()`.
</details>

### Check 4
Run `python ppo.py --run-id check` for a few updates (Ctrl+C stops it and saves). It's working when:
- approx_kl stays well under 0.05, and clip_frac is not 0 or 1;
- the value loss falls over the first updates;
- entropy decreases slowly, not suddenly.

**Think**
1. Why is clipping the ratio more stable than just using a small learning rate?
2. Why re-use each batch for several epochs, and what would happen with 50 epochs?
3. Why does the old log-prob have to come from the buffer and not be recomputed?

---

## Step 5: Handle time limits properly (the TODO in `ppo.py`)

Episodes are **truncated** after 1200 decisions. The car didn't fail; it ran out of time. If you
treat that like a crash (`done = True` with no future), you teach the critic that "the world ends
here", which is false and makes value estimates noisy.

The standard fix: for truncated cars, add the value of where they *would have* continued to the
last reward, then mark the step as done:

$$r_t \leftarrow r_t + \gamma\, V(s_{\text{final}})$$

where $s_{\text{final}}$ is `info["final_obs"]` for those cars. Terminated cars (finish, crash,
stuck) keep $V = 0$ after the end.

<details><summary>▶ Hint</summary>

Inside the collect loop, after `env.step`: if `truncated.any()`, run `model.get_value` (under
`torch.no_grad()`) on `info["final_obs"][truncated]` and add `gamma *` that to `reward[truncated]`.
Keep `done = terminated | truncated` for the buffer.
</details>

**Think**: why is it fine for "finish" to have no future value, even though finishing is great?

---

## Step 6: Train and read the curves

```bash
python ppo.py --run-id first_try
tensorboard --logdir runs        # in a second terminal; open the URL it prints
```

Watch `perf/steps_per_second` to see your real speed. 5M steps on a laptop takes hours, not
minutes; let it run.

### What learning looks like here
1. **Random phase.** `episode/return` around −1 to −5: cars wander, stall and get stuck.
2. **"Forward is good."** Returns turn positive as cars learn to hold throttle and stay on the road.
3. **Corners.** `episode/length` rises (fewer crashes), and checkpoints start appearing.
4. **Finishes.** Returns jump by the +5 finish bonus. A clean run of track 1 is worth about +40.
5. **Stunts.** Jumps and loops come last: they need speed held over many seconds, which is the
   long-term credit assignment that $\gamma$ and GAE are for.

### When it doesn't learn

| Symptom | Likely cause |
|---|---|
| Return flat from the start | Sign of the policy loss; log-prob not summed over action dims; advantages all ~0 (check GAE) |
| Entropy drops to almost nothing quickly | Learning rate too high or `ent_coef` too low; the policy stops exploring |
| approx_kl above 0.1, clip_frac above 0.5 | Steps too big: lower `lr`, fewer epochs, bigger minibatches |
| Value loss huge or growing | Returns large compared with network outputs; try normalising returns or rewards |
| NaN | log std ran away (clamp it), or division by a zero std in advantage normalisation (add 1e-8) |
| Learns, then collapses | Too many epochs or too high a learning rate; anneal the LR |

---

## Step 7: See it drive, and race it

**Watch a checkpoint** (no learning):
```bash
python play.py runs/first_try/checkpoint.pt
```
then press Play on the Training scene in Unity, or open the game and pick **WATCH AI TRAIN**.

**Train while watching 1000 cars**: run `python ppo.py --run-id live --editor`, then pick
WATCH AI TRAIN in the game with 1000 cars. With $N = 1000$, a rollout of 256 steps is a batch of
256,000; set `rollout_steps` to about 32–64 for this mode. You get the follow-the-leader camera and
the in-game learning curve, but it runs at about real time, so for real training use the headless build.

**Race it** in the time attack:
```bash
python export_onnx.py runs/first_try/checkpoint.pt
cp runs/first_try/TMRacer.onnx ../Assets/Resources/ML/TMRacer.onnx
```
In the game menu, turn on **AI RIVAL**. Your policy races beside you, passing through you like a ghost.

---

## Step 8: Experiments (change one thing at a time)

Each one teaches something. Make a new `--run-id` per experiment and compare them in TensorBoard.

1. **$\lambda$ = 0, 0.95, 1.0.** Feel the bias-variance trade-off in GAE.
2. **$\gamma$ = 0.9 vs 0.99 vs 0.995.** Does a short horizon still clear the jumps?
3. **`ent_coef` = 0 vs 0.005 vs 0.02.** Exploration vs settling down.
4. **`clip_coef` = 0.1 vs 0.2 vs 0.4, and with epochs = 10.** Why the clip matters.
5. **Observation normalisation.** Keep a running mean and std of observations and feed the network
   `(obs - mean) / std`. Some inputs reach ±20 when a car is far off the road.
6. **Learning-rate annealing.** Decay `lr` linearly to 0 over training.
7. **Network size.** Hidden layers of 64 vs 256 vs 512.
8. **Rewards (Unity side).** Change the constants at the top of `Assets/Scripts/TrackRush/ML/RacerAgent.cs`,
   then rebuild (**Track Rush > Build Training Environment**). What happens if the progress reward
   is 10× larger? If there is no wall penalty?
9. **Curriculum.** On the Training Manager, train on track 1 only (`Tracks = [0]`), then all five.
   Rebuild after changing it.
10. **Compare with Unity's trainer.** `mlagents-learn` with an equivalent config. Is your PPO as good?

---

## Step 9: Go faster with parallel games (`vec_env.py`)

Collecting experience is the slow part. Fill in `vec_env.py` so that K headless games (ports
5005, 5006, ...) run in separate processes, and the main process sees one environment with the
sum of their cars. The key trick: **send actions to every worker first, then receive from every
worker**, so the games simulate at the same time.

Then in `ppo.py` swap `TrackRushEnv(...)` for `ParallelTrackRushEnv(num_instances=4)`. Nothing else
changes, because the interface is identical. That is the point of a vectorised environment.

On your machine (12 threads), try 2, 3 and 4 instances and watch `perf/steps_per_second`. Stop
adding instances when it stops rising.

---

## Where to go next
- **Recurrent policies (LSTM).** Memory helps when the observation doesn't show everything.
- **SAC.** Off-policy and sample-efficient for continuous control. Compare it against your PPO.
- **Self-play / racing.** Turn car collisions back on and train against each other.
- **Imitation.** Record yourself driving (Keyboard Test) and pre-train the policy on it (behaviour cloning).
