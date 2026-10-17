"""
Step 0: check the connection before writing any RL.  COMPLETE.

Drives every car with random actions and prints what comes back from the game.

    python smoke_test.py            # headless build
    python smoke_test.py --editor   # the open Editor (run this, then press Play)
"""
import argparse

import numpy as np

from trackrush_env import TrackRushEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--editor", action="store_true")
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()

    with TrackRushEnv(editor=args.editor, time_scale=1.0 if args.editor else 20.0) as env:
        obs = env.reset()
        print(f"behavior {env.behavior_name}: {env.num_agents} cars, obs {obs.shape}, actions ({env.num_agents}, {env.act_dim})")
        print("car 0's first observation:")
        for name, value in zip(env.obs_names, obs[0]):
            print(f"  {name:28s} {value:+.3f}")

        rng = np.random.default_rng(0)
        returns, rewards_seen, terms, truncs = [], [], 0, 0
        for _ in range(args.steps):
            actions = np.column_stack([rng.uniform(-0.3, 0.3, env.num_agents),   # steer
                                       rng.uniform(0.3, 1.0, env.num_agents)])   # mostly throttle
            obs, rew, term, trunc, info = env.step(actions)
            rewards_seen.append(rew)
            terms += term.sum()
            truncs += trunc.sum()
            returns += info["episode_returns"]

        r = np.concatenate(rewards_seen)
        print(f"\n{args.steps} steps: reward per step mean {r.mean():+.4f} (min {r.min():+.2f}, max {r.max():+.2f})")
        print(f"episodes ended: {terms} terminated, {truncs} truncated; "
              f"mean return {np.mean(returns) if returns else float('nan'):+.2f}")


if __name__ == "__main__":
    main()
