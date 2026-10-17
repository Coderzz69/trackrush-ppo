"""
Watch a trained checkpoint drive, in real time, without learning.  COMPLETE (uses your model.py).

    python play.py runs/first_try/checkpoint.pt --game 100     # opens the standalone game by itself
    python play.py runs/first_try/checkpoint.pt               # or the Editor:
then in Unity press Play on the Training scene, or start "Watch AI Train" in the game.
"""
import argparse

import torch

from model import ActorCritic
from trackrush_env import TrackRushEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--build", action="store_true", help="use the headless build instead of the Editor")
    p.add_argument("--game", type=int, default=None, metavar="CARS",
                   help="launch the standalone game in Watch AI Train with CARS cars (no Editor needed)")
    p.add_argument("--track", type=int, default=0, help="track for --game")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = (TrackRushEnv(watch=args.game, watch_track=args.track, time_scale=1.0) if args.game else
           TrackRushEnv(editor=not args.build, time_scale=1.0, no_graphics=False))
    with env:
        obs = env.reset()
        try:
            while True:
                with torch.no_grad():
                    action = model.act_deterministic(torch.as_tensor(obs)).numpy()
                obs, *_ = env.step(action)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
