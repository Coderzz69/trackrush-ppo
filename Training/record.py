"""
Film a checkpoint driving: the best of several cars, as an MP4 plus a short timelapse GIF.

    python record.py runs/live/checkpoint.pt --out videos/live            # -> videos/live.mp4, .gif, .json

Launches the standalone game (Track Rush > Build Game (Linux)) in Watch AI Train record mode with
`--cars` cars from a mass start. Car 0 drives the deterministic policy (the mean action), the others
sample from it like during training, so the run differs per car; the camera follows whichever car is
furthest ahead, so the video shows the best run. Recording stops when the first car crosses the line
(with a mass start that is also the fastest lap), or once every car's first attempt has ended, or after
--max-seconds of game time. Needs ffmpeg.

ppo.py --video-every calls this for snapshots during training; see README.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from model import ActorCritic
from trackrush_env import TrackRushEnv

FPS = 30
DECISIONS_PER_SECOND = 12   # 60 Hz physics, a decision every 5 steps
GIF_SECONDS = 12            # the GIF is a timelapse of the whole run squeezed to about this long


def _encoder() -> list[str]:
    """Best available H.264 encoder (Fedora's ffmpeg has no libx264), as ffmpeg arguments."""
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if " libx264 " in encoders:
        return ["-c:v", "libx264", "-crf", "30", "-preset", "veryfast"]
    if " libopenh264 " in encoders:
        return ["-c:v", "libopenh264", "-b:v", "600k"]
    return ["-c:v", "mpeg4", "-q:v", "5"]


def encode(frames: Path, out: Path) -> tuple[Path, Path, float]:
    """frames/frame_00000.jpg ... -> out.mp4 and out.gif. Returns the paths and the duration in seconds."""
    n = len(list(frames.glob("frame_*.jpg")))
    if n == 0:
        raise RuntimeError(f"no frames in {frames}; see logs/record.log")
    mp4, gif = out.with_suffix(".mp4"), out.with_suffix(".gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "frame_%05d.jpg"),
                    "-vf", "scale=854:-2", *_encoder(), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                   check=True)
    duration = n / FPS
    speed = max(1.0, duration / GIF_SECONDS)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    f"setpts=PTS/{speed:.3f},fps=8,scale=400:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    return mp4, gif, duration


def record(checkpoint: str | Path, out: str | Path, cars: int = 10, track: int = 0, caption: str = "",
           max_seconds: float = 75.0, seed: int = 0) -> dict:
    """Films `checkpoint` and writes out.mp4, out.gif and out.json (the result, also returned)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch.manual_seed(seed)

    frames = Path(tempfile.mkdtemp(prefix="trackrush_frames_"))
    returns, finished, lengths = np.full(cars, np.nan), np.zeros(cars, bool), np.zeros(cars, int)
    try:
        with TrackRushEnv(watch=cars, watch_track=track, time_scale=1.0, record_dir=frames, caption=caption,
                          seed=seed) as env:
            obs = env.reset()
            first_done = np.zeros(env.num_agents, bool)
            max_steps = int(max_seconds * DECISIONS_PER_SECOND)
            for step in range(max_steps):
                with torch.no_grad():
                    o = torch.as_tensor(obs)
                    action, *_ = model.get_action_and_value(o)
                    action[0] = model.act_deterministic(o[:1])[0]
                obs, _, term, trunc, info = env.step(action.numpy())
                ended = np.flatnonzero(term | trunc)   # same order as the info lists
                for i, ret, length, fin in zip(ended, info["episode_returns"], info["episode_lengths"],
                                               info["episode_finished"]):
                    if not first_done[i]:
                        first_done[i] = True
                        returns[i], lengths[i], finished[i] = ret, length, fin
                if finished.any() or first_done.all():
                    break
            for _ in range(DECISIONS_PER_SECOND):   # one more second so the video doesn't cut off hard
                with torch.no_grad():
                    obs, *_ = env.step(model.act_deterministic(torch.as_tensor(obs)).numpy())
        mp4, gif, duration = encode(frames, out)
    finally:
        shutil.rmtree(frames, ignore_errors=True)

    best = int(np.nanargmax(returns)) if not np.isnan(returns).all() else 0
    result = {
        "checkpoint": str(checkpoint), "cars": cars, "track": track, "caption": caption,
        "mp4": mp4.name, "gif": gif.name, "seconds": round(duration, 1),
        "best_car": best, "best_return": None if np.isnan(returns[best]) else round(float(returns[best]), 2),
        "best_finished": bool(finished[best]),
        "best_time_s": round(lengths[best] / DECISIONS_PER_SECOND, 2) if finished[best] else None,
        "finished": int(finished.sum()),
        "mean_return": None if np.isnan(returns).all() else round(float(np.nanmean(returns)), 2),
        "deterministic_return": None if np.isnan(returns[0]) else round(float(returns[0]), 2),
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--cars", type=int, default=10)
    p.add_argument("--track", type=int, default=0)
    p.add_argument("--caption", default="", help='text on the video, "|" for a new line')
    p.add_argument("--max-seconds", type=float, default=75.0)
    args = p.parse_args()
    result = record(args.checkpoint, args.out, args.cars, args.track, args.caption, args.max_seconds)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
