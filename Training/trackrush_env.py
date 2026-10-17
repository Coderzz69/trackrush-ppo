"""
Track Rush <-> Python bridge.  COMPLETE - this is the engine I/O, not RL; you shouldn't need to edit it.

Wraps ML-Agents' low-level API into a vectorised, gym-like interface: every car in the game is one
row of the arrays you get back, in a fixed order, and episodes restart automatically.

    env = TrackRushEnv()                       # headless build (fast)
    env = TrackRushEnv(editor=True)            # the open Unity Editor: run this, then press Play
    env = TrackRushEnv(agents_per_track=20)    # 20 cars on each of the 5 tracks = 100 cars (build only)
    env = TrackRushEnv(watch=100)              # the standalone game in Watch AI Train with 100 cars, no Editor
    obs = env.reset()                          # (num_agents, 44) float32
    obs, rew, term, trunc, info = env.step(actions)   # actions: (num_agents, 2) in [-1, 1]

One step = one decision. The game runs physics at 60 Hz and each car decides every 5 physics
steps, so a step is 1/12 s of game time; rewards are summed over those 5 physics steps.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

EDITOR_PORT = 5004       # the Editor (and the game's "Watch AI Train" mode) connect here
BUILD_BASE_PORT = 5005   # builds use 5005 + worker_id, so several can run at once
DEFAULT_BUILD = Path(__file__).resolve().parent / "Build" / "TrackRushTrain.x86_64"
GAME_BUILD = Path(__file__).resolve().parent.parent / "Builds" / "TrackRush" / "TrackRush.x86_64"
LOG_DIR = Path(__file__).resolve().parent / "logs"   # Unity's own log goes here instead of your console

LOOK_AHEAD_M = (10, 20, 35, 50, 75, 100, 140)

# What each of the 44 observation numbers means (same order Unity sends them; see RacerAgent.cs).
OBS_NAMES = [
    # own motion, in the car's frame
    "vel_sideways/20", "vel_up/20", "vel_forward/105", "yaw_rate/3",
    # pose relative to the road
    "heading_sin", "heading_cos", "roll_vs_road", "pitch_vs_road",
    "lateral_offset/half_width",          # -1 / +1 = left / right road edge
    "height_above_road/5",
    # contact
    "grounded", "on_grass", "boost_left",
    # the road ahead in the car's frame: (sideways/30, up/30, forward/distance, is_gap)
    *[f"ahead_{d}m_{k}" for d in LOOK_AHEAD_M for k in ("x/30", "y/30", "z/d", "gap")],
    # misc
    "progress_fraction", "prev_steer", "prev_pedal",
]
assert len(OBS_NAMES) == 44

# Actions, both continuous in [-1, 1] (Unity clips anything outside):
ACTION_NAMES = ["steer (-1 left, +1 right)", "pedal (+ throttle, - brake/reverse)"]

# Rewards Unity hands out (constants at the top of RacerAgent.cs):
#   +0.02 per metre of progress along the track     +1 per checkpoint     +5 finish (episode ends)
#   -0.01 per physics step on grass                 -0.3 wall hit         -0.02 per step scraping a wall
#   -1 and episode ends: off the road 3 s, stuck 4 s, or facing the wrong way 1.5 s
# Episodes are also cut off ("truncated") after 6000 physics steps = 1200 decisions.


class TrackRushEnv:
    """All cars of one Unity instance as a vectorised environment with auto-reset."""

    def __init__(self, editor: bool = False, build_path: str | Path | None = None, worker_id: int = 0,
                 time_scale: float = 20.0, no_graphics: bool = True, seed: int = 0, timeout_wait: int = 600,
                 agents_per_track: int | None = None, tracks: list[int] | None = None,
                 watch: int | None = None, watch_track: int = 0, mass_start: bool = True,
                 record_dir: str | Path | None = None, caption: str = "", window: tuple[int, int] = (1280, 720)):
        """agents_per_track / tracks: override the build's 8 cars on tracks 0-4 (ignored with editor=True,
        where the Training scene's TrainingManager settings apply).
        watch: launch the standalone game (Track Rush > Build Game (Linux)) straight into Watch AI Train
        with this many cars on track `watch_track`; it is closed again with the env.
        record_dir (with watch): the game saves every frame there as a JPG at 30 fps of game time, in a
        `window`-sized window with `caption` ("|" = new line) instead of the interactive HUD. See record.py."""
        self._channel = EngineConfigurationChannel()
        self._game = None
        if watch:
            if not GAME_BUILD.exists():
                raise FileNotFoundError(f"{GAME_BUILD} not found. In Unity: Track Rush > Build Game (Linux).")
            LOG_DIR.mkdir(exist_ok=True)
            # the game waits for our port to open, so it is fine to start it before UnityEnvironment
            # --mlagents-port: a standalone player only connects to a trainer when given a port
            cmd = [str(GAME_BUILD), "--mlagents-port", str(EDITOR_PORT),
                   "-watch", str(watch), "-track", str(watch_track), "-massStart", "1" if mass_start else "0"]
            if record_dir:
                cmd += ["-record", str(record_dir), "-caption", caption, "-screen-fullscreen", "0",
                        "-screen-width", str(window[0]), "-screen-height", str(window[1])]
            log = LOG_DIR / ("record.log" if record_dir else "game.log")
            self._game = subprocess.Popen(cmd + ["-logFile", str(log)])
            print(f"Started the game in Watch AI Train with {watch} cars ...")
            try:
                self._env = UnityEnvironment(file_name=None, base_port=EDITOR_PORT, seed=seed,
                                             side_channels=[self._channel], timeout_wait=timeout_wait)
            except BaseException:
                self._game.kill()
                raise
        elif editor:
            print(f"Waiting for Unity on port {EDITOR_PORT}: press Play in the Editor "
                  "(Training scene, or Watch AI Train in the game) ...")
            self._env = UnityEnvironment(file_name=None, base_port=EDITOR_PORT, seed=seed,
                                         side_channels=[self._channel], timeout_wait=timeout_wait)
        else:
            path = Path(build_path) if build_path else DEFAULT_BUILD
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. In Unity: Track Rush > Build Training Environment (Linux).")
            LOG_DIR.mkdir(exist_ok=True)
            extra = []
            if agents_per_track:
                extra += ["-agentsPerTrack", str(agents_per_track)]
            if tracks:
                extra += ["-tracks", ",".join(map(str, tracks))]
            self._env = UnityEnvironment(file_name=str(path), worker_id=worker_id, base_port=BUILD_BASE_PORT,
                                         seed=seed, no_graphics=no_graphics, log_folder=str(LOG_DIR),
                                         side_channels=[self._channel], timeout_wait=timeout_wait,
                                         additional_args=extra)
        self._channel.set_configuration_parameters(time_scale=time_scale)
        self._env.reset()

        self.behavior_name = list(self._env.behavior_specs)[0]
        spec = self._env.behavior_specs[self.behavior_name]
        self.obs_dim = int(np.prod(spec.observation_specs[0].shape))
        self.act_dim = spec.action_spec.continuous_size
        self.obs_names, self.action_names = OBS_NAMES, ACTION_NAMES

        self._row: dict[int, int] = {}   # ML-Agents agent id -> row in our arrays
        self._decision_ids = None        # agent ids of the last decision batch, in ML-Agents' order
        self.num_agents = 0
        self._ep_return = self._ep_len = None

    # ------------------------------------------------------------------ public API

    def reset(self) -> np.ndarray:
        """Starts fresh episodes for every car. Returns obs of shape (num_agents, obs_dim)."""
        self._env.reset()
        ds = self._wait_for_decisions()[0]
        self._row = {int(a): i for i, a in enumerate(sorted(ds.agent_id))}
        self.num_agents = len(self._row)
        self._ep_return = np.zeros(self.num_agents, np.float64)
        self._ep_len = np.zeros(self.num_agents, np.int64)
        obs = np.zeros((self.num_agents, self.obs_dim), np.float32)
        obs[self._rows(ds.agent_id)] = self._obs(ds)
        self._decision_ids = ds.agent_id
        return obs

    def step(self, actions: np.ndarray):
        """
        actions: (num_agents, act_dim), row i is car i.
        Returns
          obs         (num_agents, obs_dim)  next observation. For a car whose episode just ended
                                             this is already the FIRST obs of its new episode.
          rewards     (num_agents,)          reward earned since the previous step
          terminated  (num_agents,) bool     episode ended for real (finish, crash, stuck, wrong way)
          truncated   (num_agents,) bool     episode was cut off by the time limit (not a failure!)
          info        dict:
             "final_obs"        (num_agents, obs_dim)  last obs of the episode that ended
                                                       (valid where terminated | truncated)
             "episode_returns"  list of total rewards of episodes that ended this step
             "episode_lengths"  list of their lengths in steps
             "episode_finished" list of bools: did that episode end by crossing the finish line
        """
        actions = np.asarray(actions, np.float32).reshape(self.num_agents, self.act_dim)
        self._env.set_actions(self.behavior_name,
                              ActionTuple(continuous=np.clip(actions[self._rows(self._decision_ids)], -1, 1)))

        n = self.num_agents
        rewards = np.zeros(n, np.float32)
        terminated = np.zeros(n, bool)
        truncated = np.zeros(n, bool)
        final_obs = np.zeros((n, self.obs_dim), np.float32)

        ds, ts = self._wait_for_decisions(step_first=True, on_terminal=lambda ts: self._collect_terminal(
            ts, rewards, terminated, truncated, final_obs))

        obs = np.zeros((n, self.obs_dim), np.float32)
        rows = self._rows(ds.agent_id)
        if len(rows) != n:
            raise RuntimeError(f"Expected decisions from {n} cars, got {len(rows)}. Did the scene change?")
        obs[rows] = self._obs(ds)
        done = terminated | truncated
        # cars that just restarted report their new episode's (≈0) reward here; it is not part of
        # the transition that ended, so only add decision rewards for cars still in their episode
        alive = ~done[rows]
        rewards[rows[alive]] += ds.reward[alive]
        self._decision_ids = ds.agent_id

        self._ep_return += rewards
        self._ep_len += 1
        info = {"final_obs": final_obs,
                "episode_returns": self._ep_return[done].tolist(),
                "episode_lengths": self._ep_len[done].tolist(),
                # the finish bonus (+5) is the only way a final step earns more than +3; a fail costs -1
                "episode_finished": (terminated[done] & (rewards[done] > 3.0)).tolist()}
        self._ep_return[done] = 0
        self._ep_len[done] = 0
        return obs, rewards, terminated, truncated, info

    def close(self):
        try:
            self._env.close()
        finally:
            if self._game is not None:
                self._game.terminate()
                try:
                    self._game.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._game.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ internals

    def _wait_for_decisions(self, step_first=False, on_terminal=None):
        """Advances Unity until the cars ask for actions; episode ends on the way go to on_terminal."""
        if step_first:
            self._env.step()
        while True:
            ds, ts = self._env.get_steps(self.behavior_name)
            if on_terminal is not None and len(ts) > 0:
                on_terminal(ts)
            if len(ds) > 0:
                return ds, ts
            self._env.step()

    def _collect_terminal(self, ts, rewards, terminated, truncated, final_obs):
        rows = self._rows(ts.agent_id)
        rewards[rows] += ts.reward
        truncated[rows] |= ts.interrupted
        terminated[rows] |= ~ts.interrupted
        final_obs[rows] = self._obs(ts)

    def _rows(self, agent_ids) -> np.ndarray:
        return np.array([self._row[int(a)] for a in agent_ids], dtype=np.int64)

    @staticmethod
    def _obs(steps) -> np.ndarray:
        return steps.obs[0].reshape(len(steps), -1).astype(np.float32)
