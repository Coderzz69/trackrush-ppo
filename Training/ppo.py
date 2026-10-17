"""
Step 3: the PPO training loop.

    python ppo.py --run-id first_try               # headless build, fast
    python ppo.py --run-id watch --editor          # train while watching in the Editor
    python ppo.py --run-id watch --game 100        # train while watching in the standalone game (no Editor)
    python ppo.py --run-id fast --instances 4      # 4 headless games in parallel
    python ppo.py --run-id big --cars-per-track 20 # 100 cars (20 on each of the 5 tracks) in one game
    tensorboard --logdir runs                      # learning curves

    # until the reward curve levels off, with a video every 1000 episodes and a report for git:
    python ppo.py --run-id v1 --instances 2 --cars-per-track 10 --video-every 1000 --until-stable

Reading list: Schulman et al. 2017 "Proximal Policy Optimization Algorithms" (eq. 7 and 9), and
"The 37 Implementation Details of PPO" (ICLR blog track, 2022).
"""
import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import report
from buffer import RolloutBuffer
from milestones import Milestones
from model import ActorCritic
from trackrush_env import TrackRushEnv
from vec_env import ParallelTrackRushEnv

RUNS = Path(__file__).resolve().parent / "runs"


@dataclass
class Config:
    # starting points - tuning them is part of the learning
    total_steps: int = 5_000_000    # decisions summed over all cars
    rollout_steps: int = 256       # T: steps per car between updates (batch = T * num_cars)
    epochs: int = 4                # passes over each batch
    minibatch_size: int = 4096
    lr: float = 3e-4
    gamma: float = 0.99            # discount
    gae_lambda: float = 0.95
    clip_coef: float = 0.2         # PPO's epsilon
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5
    time_scale: float = 20.0
    seed: int = 1


def ppo_update(model, optimizer, buffer: RolloutBuffer, cfg: Config, device) -> dict:
    """
    One PPO update over the collected rollout: `epochs` passes of shuffled minibatches, each minimising
        -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv) + vf_coef * value_loss - ent_coef * entropy
    (PPO paper eq. 7 and 9). Returns averaged diagnostics for TensorBoard.
    """
    totals = {"loss/policy": 0.0, "loss/value": 0.0, "policy/entropy": 0.0,
              "policy/approx_kl": 0.0, "policy/clip_frac": 0.0}
    n = 0
    for _ in range(cfg.epochs):
        for mb in buffer.minibatches(cfg.minibatch_size, device):
            _, new_log_prob, entropy, new_value = model.get_action_and_value(mb["obs"], mb["actions"])

            log_ratio = new_log_prob - mb["log_probs"]
            ratio = log_ratio.exp()

            adv = mb["advantages"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # unbiased, low-variance KL estimator: E[(r - 1) - log r]
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_frac = ((ratio - 1).abs() > cfg.clip_coef).float().mean()
            totals["loss/policy"] += policy_loss.item()
            totals["loss/value"] += value_loss.item()
            totals["policy/entropy"] += entropy_mean.item()
            totals["policy/approx_kl"] += approx_kl.item()
            totals["policy/clip_frac"] += clip_frac.item()
            n += 1

    stats = {k: v / max(n, 1) for k, v in totals.items()}
    stats["policy/std"] = model.log_std.exp().mean().item()
    stats["lr"] = optimizer.param_groups[0]["lr"]
    return stats


def save_checkpoint(model, cfg: Config, run_dir: Path, step: int, obs_dim: int, act_dim: int, path: Path | None = None):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step,
                "obs_dim": obs_dim, "act_dim": act_dim}, path or run_dir / "checkpoint.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="ppo")
    p.add_argument("--editor", action="store_true", help="connect to the open Editor instead of the build")
    p.add_argument("--game", type=int, default=None, metavar="CARS",
                   help="train while watching: launches the standalone game in Watch AI Train with CARS cars")
    p.add_argument("--track", type=int, default=0, help="track for --game")
    p.add_argument("--instances", type=int, default=1, help="parallel headless games (see vec_env.py)")
    p.add_argument("--cars-per-track", type=int, default=None,
                   help="cars on each of the 5 tracks in every headless game (build default: 8)")
    p.add_argument("--total-steps", type=int, default=None, help=f"default {Config.total_steps:,} "
                   "(30M with --until-stable, which usually stops long before)")
    p.add_argument("--video-every", type=int, default=0, metavar="EPISODES",
                   help="every EPISODES episodes: snapshot, film the best of --video-cars cars, add to the report")
    p.add_argument("--video-cars", type=int, default=10)
    p.add_argument("--video-track", type=int, default=0)
    p.add_argument("--no-videos", action="store_true", help="milestones and report without filming")
    p.add_argument("--until-stable", action="store_true",
                   help="stop once the reward curve levels off (checked at every milestone; needs --video-every)")
    p.add_argument("--min-steps", type=int, default=1_500_000, help="--until-stable never stops before this")
    args = p.parse_args()
    if args.until_stable and not args.video_every:
        p.error("--until-stable needs --video-every (the curve is checked per block of episodes)")

    total = args.total_steps or (30_000_000 if args.until_stable else Config.total_steps)
    cfg = Config(time_scale=1.0 if args.editor or args.game else Config.time_scale, total_steps=total)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.run_id
    writer = SummaryWriter(str(run_dir))

    if args.game:
        env = TrackRushEnv(watch=args.game, watch_track=args.track, time_scale=cfg.time_scale, seed=cfg.seed)
    elif args.instances > 1 and not args.editor:
        env = ParallelTrackRushEnv(num_instances=args.instances, time_scale=cfg.time_scale, seed=cfg.seed,
                                   agents_per_track=args.cars_per_track)
    else:
        env = TrackRushEnv(editor=args.editor, time_scale=cfg.time_scale, seed=cfg.seed,
                           agents_per_track=args.cars_per_track)
    milestones = None
    if args.video_every:
        milestones = Milestones(args.run_id, run_dir, every=args.video_every, video_cars=args.video_cars,
                                video_track=args.video_track, videos=not args.no_videos,
                                stable_window=5, min_steps=args.min_steps if args.until_stable else 10**18,
                                meta={"num_cars": env.num_agents, "instances": args.instances,
                                      "cars_per_track": args.cars_per_track, "device": str(device),
                                      "config": asdict(cfg)})
    try:
        model, step, reason = train(env, cfg, device, run_dir, writer, milestones)
    finally:
        writer.close()
        env.close()   # always stop the Unity process, even on errors

    if milestones is not None:
        milestones.finish(step, lambda path: save_checkpoint(model, cfg, run_dir, step, env.obs_dim, env.act_dim, path),
                          reason)
        print(f"report: {report.write(run_dir)}")


def train(env: TrackRushEnv, cfg: Config, device, run_dir: Path, writer: SummaryWriter,
          milestones: Milestones | None = None):
    """Returns (model, steps, why it stopped: "stable" | "total_steps" | "interrupted")."""
    obs = env.reset()
    N = env.num_agents
    print(f"{N} cars, obs {env.obs_dim}, actions {env.act_dim}, device {device}")
    if milestones is not None:
        milestones.data["num_cars"] = N

    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    buffer = RolloutBuffer(cfg.rollout_steps, N, env.obs_dim, env.act_dim)

    step, update, t0 = 0, 0, time.time()
    reason, stable = "total_steps", False
    snapshot = lambda path: save_checkpoint(model, cfg, run_dir, step, env.obs_dim, env.act_dim, path)
    try:
        while step < cfg.total_steps:
            # ---------------------------------------------------------- 1. collect a rollout
            buffer.reset()
            while not buffer.full():
                with torch.no_grad():
                    action, log_prob, _, value = model.get_action_and_value(torch.as_tensor(obs, device=device))
                action, log_prob, value = action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

                next_obs, reward, terminated, truncated, info = env.step(action)   # env clips to [-1, 1]

                # truncated cars didn't fail, they ran out of time: bootstrap with the value of where
                # they would have continued, r += gamma * V(final_obs). Terminated cars keep V = 0.
                if truncated.any():
                    with torch.no_grad():
                        final_v = model.get_value(torch.as_tensor(info["final_obs"][truncated], device=device))
                    reward = reward + 0  # copy, don't modify the env's array in place
                    reward[truncated] += cfg.gamma * final_v.cpu().numpy()
                done = terminated | truncated

                buffer.add(obs, action, log_prob, reward, done, value)
                obs = next_obs
                step += N
                for ret, length in zip(info["episode_returns"], info["episode_lengths"]):
                    writer.add_scalar("episode/return", ret, step)
                    writer.add_scalar("episode/length", length, step)
                if milestones is not None and milestones.add(step, info, snapshot):
                    stable = True

            # ---------------------------------------------------------- 2. advantages
            with torch.no_grad():
                last_values = model.get_value(torch.as_tensor(obs, device=device)).cpu().numpy()
            buffer.compute_advantages(last_values, cfg.gamma, cfg.gae_lambda)

            # ---------------------------------------------------------- 3. learn
            # linearly anneal the learning rate to 0 over training
            optimizer.param_groups[0]["lr"] = cfg.lr * max(1.0 - step / cfg.total_steps, 0.0)
            stats = ppo_update(model, optimizer, buffer, cfg, device)
            update += 1

            # ---------------------------------------------------------- 4. log and save
            for k, v in (stats or {}).items():
                writer.add_scalar(k, v, step)
            sps = step / (time.time() - t0)
            writer.add_scalar("perf/steps_per_second", sps, step)
            print(f"update {update:4d}  step {step:>10,}  {sps:7.0f} steps/s  "
                  f"mean reward/step {buffer.rewards.mean():+.4f}")
            if update % 10 == 0:
                save_checkpoint(model, cfg, run_dir, step, env.obs_dim, env.act_dim)
            if stable:
                print("the reward curve has levelled off: stopping")
                reason = "stable"
                break
    except KeyboardInterrupt:
        print("stopping ...")
        reason = "interrupted"
    save_checkpoint(model, cfg, run_dir, step, env.obs_dim, env.act_dim)
    print(f"saved {run_dir / 'checkpoint.pt'}")
    return model, step, reason


if __name__ == "__main__":
    main()
