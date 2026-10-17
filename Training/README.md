# Track Rush: PPO from scratch

Everything Python lives in this folder:
`~/Desktop/unity_projects/AI_Playground/Training/`

```bash
cd ~/Desktop/unity_projects/AI_Playground/Training
source .venv/bin/activate        # Python 3.10: mlagents-envs 1.1.0, torch 2.8 (CUDA), tensorboard
```

## Files

| File | Status | What it is |
|---|---|---|
| `trackrush_env.py` | done | The game as a vectorised env: `reset()`, `step(actions)`, observation names, rewards |
| `smoke_test.py` | done | Random driving to check the connection. Run this first. |
| `model.py` | done | Actor-critic network (tanh MLPs, orthogonal init, learnable log std) |
| `buffer.py` | done | Rollout storage `(T, N, ...)` and GAE |
| `ppo.py` | done | Training loop, clipped PPO update, time-limit bootstrapping, LR annealing |
| `play.py` | done | Watch a checkpoint drive (uses your `model.py`) |
| `export_onnx.py` | done | Checkpoint to `TMRacer.onnx` for the game's AI RIVAL option |
| `vec_env.py` | done | Several game instances in parallel: `python ppo.py --instances 4` |
| `milestones.py` | done | Every N episodes: snapshot, stats, background video, "has the curve levelled off?" |
| `record.py` | done | Film a checkpoint (best of 10 cars) to MP4 + GIF with the standalone game |
| `report.py` | done | `reports/<run-id>/REPORT.md` with the reward curve and the videos, for git |

## Suggested order

1. `python smoke_test.py`: see the 44 observations and the rewards.
2. `model.py`: `__init__`, `get_value`, `get_action_and_value`, `act_deterministic`.
3. `buffer.py`: `compute_advantages` (GAE).
4. `ppo.py`: `ppo_update` (clipped objective), then the truncation TODO.
5. Train and watch: `python ppo.py --run-id first_try`, `tensorboard --logdir runs`.
6. See it drive: `python play.py runs/first_try/checkpoint.pt`, then press Play in Unity
   (Training scene, or Watch AI Train in the game).
7. Race it: `python export_onnx.py runs/first_try/checkpoint.pt --install --build` (Editor closed),
   then start `../Builds/TrackRush/TrackRush.sh` and turn on AI RIVAL in the menu.
8. Faster: `python ppo.py --run-id fast --instances 4` (one headless game per process).

## Ways to connect

| | How | Speed |
|---|---|---|
| Headless build (default) | `TrackRushEnv()` launches `Build/TrackRushTrain.x86_64` | fastest; 40 cars on 5 tracks |
| Editor, Training scene | `TrackRushEnv(editor=True)`, then press Play | slower, you can watch |
| Game, "Watch AI Train" | `TrackRushEnv(editor=True)`, then pick the mode in the menu | up to 1000 cars, live stats |
| Standalone game, no Editor | `python ppo.py --game 100` (or `TrackRushEnv(watch=100)`) launches it in Watch AI Train | same view, without the Editor's overhead |

After changing anything on the Unity side, rebuild with **Track Rush > Build Training Environment (Linux)**,
and for the standalone game **Track Rush > Build Game (Linux)** (goes to `../Builds/TrackRush/`).

**Watching without the Editor.** `python ppo.py --run-id live --game 100 [--track 2]` starts the built game
straight into Watch AI Train with 100 cars and closes it when training stops; `python play.py
runs/live/checkpoint.pt --game 100` does the same for a trained policy. To start the game yourself, use
`../Builds/TrackRush/TrackRush.sh`, not the `.x86_64` directly: a standalone ML-Agents game only connects
to Python when launched with `--mlagents-port 5004`, which the script adds.

**How many cars.** A headless game runs 8 cars on each of the 5 tracks (40). Change it at launch,
no rebuild needed: `--cars-per-track N` (5×N cars per game) and `--instances K` (K games in parallel).

```bash
python ppo.py --run-id big --instances 2 --cars-per-track 10   # 100 cars, ~8k steps/s
python ppo.py --run-id big --cars-per-track 20                 # 100 cars in one game, ~5.5k steps/s
```
In Python: `TrackRushEnv(agents_per_track=20, tracks=[0, 1, 2, 3, 4])`.

## The interface (from `trackrush_env.py`)

- **Observations** `(N, 44)`: see `OBS_NAMES`. Own velocity, pose relative to the road, contact
  flags, the road 10–140 m ahead in the car's frame (with gap flags), progress, previous action.
- **Actions** `(N, 2)` in `[-1, 1]`: `steer`, `pedal` (+ throttle, - brake).
- **Rewards**: +0.02 per metre of progress, +1 per checkpoint, +5 for finishing; -0.01 per step on
  grass, -0.3 for hitting a wall, -0.02 per step scraping one; -1 for a fail (off road 3 s, stuck
  4 s, wrong way 1.5 s).
- **Episodes** end on finish or fail (`terminated`), or are cut off after 1200 decisions (`truncated`).
  Cars restart automatically; `info["final_obs"]` holds the last observation of the episode that ended.

## Progress videos and a report for git

```bash
python ppo.py --run-id v1 --instances 2 --cars-per-track 10 --video-every 1000 --until-stable
```
Every 1000 episodes (all cars together) the policy is saved and filmed in the standalone game: 10 cars
from a mass start, camera on the leader, stopping when the first car crosses the line. Training keeps
going while a video records (a small game window pops up each time). `--until-stable` stops once the
last 5 block averages stay in a narrow band without beating the earlier best (never before
`--min-steps`, default 1.5M). Then `reports/v1/` holds `REPORT.md`, `reward_curve.png` and `videos/`
(about 4-5 MB per milestone): that folder is what you commit. `python report.py runs/v1` rebuilds the
report at any time, also mid-training. Options: `--video-cars`, `--video-track`, `--no-videos`, `--total-steps`.
