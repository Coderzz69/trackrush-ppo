"""
Step 4 (optional, once PPO works): run several game instances in parallel.

One TrackRushEnv already gives you every car in one game. To go faster, launch K headless builds
(worker_id 0..K-1 -> ports 5005..5005+K-1), each in its own process, and present them to ppo.py as
one big environment whose num_agents is the sum over all instances.

Why processes: env.step() blocks while Unity simulates. With one process per instance, all K games
simulate at the same time while the main process waits for all of them.
Uses the "spawn" start method; forking a process that holds gRPC state is unsafe.
"""
import multiprocessing as mp

import numpy as np

from trackrush_env import TrackRushEnv


def _worker(remote, worker_id: int, time_scale: float, seed: int, env_kwargs: dict):
    """Runs in a child process and owns one TrackRushEnv; answers commands sent over `remote`."""
    env = TrackRushEnv(worker_id=worker_id, time_scale=time_scale, seed=seed + worker_id, **env_kwargs)
    try:
        obs = env.reset()
        remote.send((env.num_agents, env.obs_dim, env.act_dim, obs))
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset())
            elif cmd == "step":
                remote.send(env.step(data))
            elif cmd == "close":
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()
        remote.close()


class ParallelTrackRushEnv:
    """Same interface as TrackRushEnv (reset / step / close / num_agents / obs_dim / act_dim)."""

    def __init__(self, num_instances: int = 4, time_scale: float = 20.0, seed: int = 0, **env_kwargs):
        """env_kwargs go to every TrackRushEnv, e.g. agents_per_track=20."""
        ctx = mp.get_context("spawn")
        self._remotes, self._procs = [], []
        for i in range(num_instances):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, i, time_scale, seed, env_kwargs), daemon=True)
            proc.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(proc)

        # every worker reports its shape once it's up (each env has already been reset)
        counts = []
        for remote in self._remotes:
            n, self.obs_dim, self.act_dim, _ = remote.recv()
            counts.append(n)
        bounds = np.cumsum([0] + counts)
        self._slices = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        self.num_agents = int(bounds[-1])
        self._closed = False

    def reset(self) -> np.ndarray:
        for remote in self._remotes:
            remote.send(("reset", None))
        return np.concatenate([remote.recv() for remote in self._remotes])

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, np.float32).reshape(self.num_agents, self.act_dim)
        # send to all first, then receive from all: that's what makes the games simulate in parallel
        for remote, sl in zip(self._remotes, self._slices):
            remote.send(("step", actions[sl]))
        results = [remote.recv() for remote in self._remotes]

        obs, rew, term, trunc, infos = zip(*results)
        info = {"final_obs": np.concatenate([i["final_obs"] for i in infos]),
                "episode_returns": [r for i in infos for r in i["episode_returns"]],
                "episode_lengths": [l for i in infos for l in i["episode_lengths"]],
                "episode_finished": [f for i in infos for f in i["episode_finished"]]}
        return np.concatenate(obs), np.concatenate(rew), np.concatenate(term), np.concatenate(trunc), info

    def close(self):
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
