"""
Training report for git: reports/<run-id>/REPORT.md with the reward curve and the milestone videos.

    python report.py runs/v1          # (re)build it from runs/v1/milestones.json + episodes.npz

ppo.py --video-every writes those files and calls this at the end; you can also run it mid-training.
Only reports/ is meant for git (runs/ is ignored): REPORT.md, reward_curve.png and videos/.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from milestones import REPORTS  # noqa: E402

INLINE_GIFS = 8   # GIFs shown inline (evenly spaced, always incl. the last); the table links every video

STOPPED = {"stable": "the reward curve levelled off", "total_steps": "the step budget ran out",
           "interrupted": "stopped by hand (Ctrl+C)", None: "still running"}


def _fmt_time(seconds: float | None) -> str:
    return "-" if seconds is None else f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} s"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def plot(data: dict, episodes: dict, out: Path):
    returns = episodes["returns"]
    ms = data["milestones"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(1, len(returns) + 1)
    ax.scatter(x, returns, s=1, alpha=0.08, color="#888", rasterized=True, label="episode return")
    w = min(data["every"], max(1, len(returns) // 10))
    if len(returns) >= w:
        avg = np.convolve(returns, np.ones(w) / w, mode="valid")
        ax.plot(x[w - 1:], avg, color="#1f77b4", lw=2, label=f"average of last {w:,}")
    vids = [m for m in ms if m.get("video", {}).get("mp4")]
    ax.scatter([m["episodes"] for m in vids], [m["mean_return"] for m in vids], marker="v", s=40,
               color="#d62728", zorder=3, label="video taken")
    if data.get("stopped") == "stable" and ms:
        ax.axvline(ms[-1]["episodes"], color="#2ca02c", ls="--", lw=1.2, label="levelled off")
    lo, hi = np.percentile(returns, [1, 99.8]) if len(returns) else (0, 1)
    ax.set_ylim(lo - 2, hi + 2)
    ax.set_ylabel("return per episode")
    ax.set_title(f"Track Rush PPO, run {data['run_id']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax2.plot([m["episodes"] for m in ms], [100 * m["finish_rate"] for m in ms], color="#2ca02c", marker="o", ms=3)
    ax2.set_ylabel("finished %")
    ax2.set_xlabel("episodes (all cars)")
    ax2.set_ylim(-3, 103)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def write(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "milestones.json").read_text())
    episodes = dict(np.load(run_dir / "episodes.npz"))
    out_dir = REPORTS / data["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(data, episodes, out_dir / "reward_curve.png")

    ms = data["milestones"]
    last = ms[-1] if ms else {}
    cfg = data.get("config", {})
    vids = [m for m in ms if m.get("video", {}).get("mp4")]
    best_lap = min((m["video"]["best_time_s"] for m in vids if m["video"].get("best_time_s")), default=None)
    first_finish = next((m for m in vids if m["video"].get("best_finished")), None)
    first_finish_text = f"after {first_finish['episodes']:,} episodes" if first_finish else "-"
    best_block = max(ms, key=lambda m: m["mean_return"]) if ms else {}

    L = [f"# Track Rush PPO: run `{data['run_id']}`", ""]
    L += [f"PPO written from scratch (PyTorch) driving the cars of a Trackmania-style Unity game. Started "
          f"{data['started']}; ran {last.get('step', 0) / 1e6:.2f}M decisions over {last.get('episodes', 0):,} "
          f"episodes in {_duration(data.get('wall_time_s', last.get('minutes', 0) * 60))}, and stopped because "
          f"{STOPPED.get(data.get('stopped'), data.get('stopped'))}.", ""]
    L += ["| | |", "|---|---|",
          f"| Final average return (last {data['every']:,} episodes) | **{last.get('mean_return', 0):+.2f}** |",
          f"| Best block average | {best_block.get('mean_return', 0):+.2f} (at {best_block.get('episodes', 0):,} episodes) |",
          f"| Best single episode | {last.get('best_return_so_far', 0):+.2f} |",
          f"| Finish rate, last block | {last.get('finish_rate', 0):.0%} |",
          f"| Best lap in the videos | {_fmt_time(best_lap)} |",
          f"| First filmed finish | {first_finish_text} |",
          f"| Cars training at once | {data.get('num_cars', '?')} ({data.get('instances', 1)} game instance(s)) |",
          ""]
    L += ["![reward curve](reward_curve.png)", ""]

    if vids:
        L += ["## The best run at each milestone", "",
              f"Every {data['every']:,} episodes the current policy was saved and filmed: {data['video_cars']} cars "
              "from a mass start (car A000 drives the deterministic policy, the others sample from it as in "
              "training), with the camera on whichever car is furthest ahead. Filming stops when the first car "
              "crosses the line, which with a mass start is the fastest lap. Click a GIF for the full video.", ""]
        picks = sorted({round(i) for i in np.linspace(0, len(vids) - 1, min(INLINE_GIFS, len(vids)))})
        for i in picks:
            m, v = vids[i], vids[i]["video"]
            result = (f"finished in **{_fmt_time(v['best_time_s'])}**" if v.get("best_finished")
                      else f"best return {v['best_return']:+.1f}, no car finished within {v['seconds']:.0f} s")
            L += [f"### {'Final, ' if m.get('final') else ''}{m['episodes']:,} episodes  ·  {m['step'] / 1e6:.2f}M steps",
                  "", f"Training average {m['mean_return']:+.1f}, finish rate {m['finish_rate']:.0%}. "
                  f"Filmed best run: {result}.", "",
                  f"[![{m['name']}](videos/{v['gif']})](videos/{v['mp4']})", ""]

    L += ["## All milestones", "",
          "| Episodes | Steps | Minutes | Avg return | Finish rate | Best episode so far | Filmed best run | Video |",
          "|---:|---:|---:|---:|---:|---:|---|---|"]
    for m in ms:
        v = m.get("video", {})
        if v.get("mp4"):
            run = _fmt_time(v["best_time_s"]) if v.get("best_finished") else f"{v['best_return']:+.1f}, no finish"
            link = f"[mp4](videos/{v['mp4']})"
        else:
            run, link = ("failed" if v.get("error") else "-"), "-"
        L.append(f"| {m['episodes']:,}{' (final)' if m.get('final') else ''} | {m['step'] / 1e6:.2f}M | {m['minutes']} | "
                 f"{m['mean_return']:+.2f} | {m['finish_rate']:.0%} | {m['best_return_so_far']:+.2f} | {run} | {link} |")
    L.append("")

    if cfg:
        L += ["## Setup", "", "| Hyperparameter | Value |", "|---|---|"]
        L += [f"| `{k}` | {v} |" for k, v in cfg.items()]
        L += ["", "Observations: 44 numbers (own motion, pose relative to the road, the road 10 to 140 m ahead, "
              "progress, previous action). Actions: steer and pedal in [-1, 1]. Rewards: +0.02 per metre of "
              "progress, +1 per checkpoint, +5 for finishing, small penalties for grass and walls, -1 for "
              "crashing, getting stuck or driving the wrong way.", ""]
        L += [f"\"Levelled off\" means: after {data['min_steps'] / 1e6:.1f}M steps, the last "
              f"{data['stable_window']} block averages stay within a narrow band and none beats the earlier best.", ""]

    (out_dir / "REPORT.md").write_text("\n".join(L))
    return out_dir / "REPORT.md"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="runs/<run-id>")
    print(write(p.parse_args().run_dir))


if __name__ == "__main__":
    main()
