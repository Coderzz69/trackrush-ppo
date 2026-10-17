"""
Progress milestones during training: every `every` episodes (summed over all cars), snapshot the policy,
log the stats of that block of episodes, film the snapshot in the background (record.py), and decide
whether the reward curve has levelled off. Used by ppo.py --video-every; report.py turns the result
into reports/<run-id>/REPORT.md.

Files:
    runs/<run-id>/snapshots/ep_0001000.pt        policy at each milestone (not for git)
    runs/<run-id>/milestones.json                run info + one entry per milestone
    runs/<run-id>/episodes.npz                   every episode's return / finish flag / step
    reports/<run-id>/videos/ep_0001000.mp4|.gif   the videos (for git)
"""
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"


class Milestones:
    def __init__(self, run_id: str, run_dir: Path, every: int = 1000, video_cars: int = 10, video_track: int = 0,
                 videos: bool = True, stable_window: int = 5, min_steps: int = 1_500_000, meta: dict | None = None):
        self.run_id, self.run_dir, self.every = run_id, run_dir, every
        self.video_cars, self.video_track, self.videos = video_cars, video_track, videos
        self.stable_window, self.min_steps = stable_window, min_steps
        self.video_dir = REPORTS / run_id / "videos"
        (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.returns: list[float] = []
        self.finished: list[bool] = []
        self.ep_steps: list[int] = []
        self.next_at = every
        self.t0 = time.time()
        self.data = {"run_id": run_id, "started": time.strftime("%Y-%m-%d %H:%M"), "every": every,
                     "video_cars": video_cars, "video_track": video_track, "stable_window": stable_window,
                     "min_steps": min_steps, **(meta or {}), "milestones": [], "stopped": None}
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._record_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ called from the training loop

    def add(self, step: int, info: dict, save_snapshot) -> bool:
        """Feed one env.step()'s info. save_snapshot(path) writes the current policy. True = curve is stable."""
        self.returns += info["episode_returns"]
        self.finished += info["episode_finished"]
        self.ep_steps += [step] * len(info["episode_returns"])
        if len(self.returns) < self.next_at:
            return False
        self.next_at += self.every
        self._milestone(step, save_snapshot)
        return self.stable(step)

    def finish(self, step: int, save_snapshot, reason: str):
        """Final milestone (unless one was just taken), then wait for the videos still being recorded."""
        if len(self.returns) - (self.next_at - self.every) >= self.every // 4:
            self._milestone(step, save_snapshot, final=True)
        with self._lock:
            self.data["stopped"] = reason
            self.data["wall_time_s"] = round(time.time() - self.t0)
            self._save()
        if self.videos and self._jobs.unfinished_tasks:
            print(f"waiting for {self._jobs.unfinished_tasks} video(s) to finish recording (Ctrl+C again to skip) ...")
            try:
                self._jobs.join()
            except KeyboardInterrupt:
                print("skipped the remaining videos")

    def stable(self, step: int) -> bool:
        """
        The curve counts as levelled off when, after min_steps, the last `stable_window` block averages
        stay inside a narrow band (2 x max(1, 5% of their mean)) and none of them beats the best earlier
        block by more than that band's half-width.
        """
        means = [m["mean_return"] for m in self.data["milestones"] if not m.get("final")]
        w = self.stable_window
        if step < self.min_steps or len(means) < w + 1:
            return False
        recent = means[-w:]
        band = max(1.0, 0.05 * abs(float(np.mean(recent))))
        return max(recent) - min(recent) <= 2 * band and max(recent) <= max(means[:-w]) + band

    # ------------------------------------------------------------------ internals

    def _milestone(self, step: int, save_snapshot, final: bool = False):
        n = len(self.returns)
        block = np.array(self.returns[-self.every:])
        fin = np.array(self.finished[-self.every:])
        name = f"ep_{n:07d}"
        snapshot = self.run_dir / "snapshots" / f"{name}.pt"
        save_snapshot(snapshot)
        entry = {
            "name": name, "episodes": n, "step": step, "minutes": round((time.time() - self.t0) / 60, 1),
            "mean_return": round(float(block.mean()), 3), "std_return": round(float(block.std()), 3),
            "max_return": round(float(block.max()), 3), "finish_rate": round(float(fin.mean()), 4),
            "best_return_so_far": round(float(max(self.returns)), 3), "final": final,
        }
        with self._lock:
            self.data["milestones"].append(entry)
            self._save()
            np.savez_compressed(self.run_dir / "episodes.npz", returns=np.array(self.returns, np.float32),
                                finished=np.array(self.finished, bool), step=np.array(self.ep_steps, np.int64))
        print(f"milestone {n:,} episodes: avg return {entry['mean_return']:+.2f}, "
              f"finish rate {entry['finish_rate']:.0%}" + (" (recording video)" if self.videos else ""))
        if self.videos:
            caption = (f"Run {self.run_id}  ·  {'final' if final else 'after'} {n:,} episodes|"
                       f"{step / 1e6:.2f}M steps  ·  avg return {entry['mean_return']:+.1f} (last {len(block):,})")
            self._jobs.put((entry, snapshot, caption))

    def _record_loop(self):
        while True:
            entry, snapshot, caption = self._jobs.get()
            try:
                out = self.video_dir / entry["name"]
                proc = subprocess.run([sys.executable, str(HERE / "record.py"), str(snapshot), "--out", str(out),
                                       "--cars", str(self.video_cars), "--track", str(self.video_track),
                                       "--caption", caption], cwd=HERE, capture_output=True, text=True,
                                      start_new_session=True)  # Ctrl+C on training lets the current video finish
                result_file = out.with_suffix(".json")
                if proc.returncode == 0 and result_file.exists():
                    video = json.loads(result_file.read_text())
                    result_file.unlink()
                else:
                    video = {"error": (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown"]}
                    print(f"video for {entry['name']} failed: {video['error']}")
                with self._lock:
                    entry["video"] = video
                    self._save()
            finally:
                self._jobs.task_done()

    def _save(self):
        (self.run_dir / "milestones.json").write_text(json.dumps(self.data, indent=2))
