"""
Export a checkpoint so the game can run it (the "AI RIVAL" option).  COMPLETE (uses your model.py).

    python export_onnx.py runs/first_try/checkpoint.pt             # writes runs/first_try/TMRacer.onnx
    python export_onnx.py runs/first_try/checkpoint.pt --install   # ...and copies it into the game
    python export_onnx.py runs/first_try/checkpoint.pt --install --build   # ...and rebuilds the game too
Without --build, rebuild yourself (Track Rush > Build Game (Linux)). Then turn on AI RIVAL in the menu.

Unity's ML-Agents reads models by tensor name, so the network is wrapped to match what it expects:
input "obs_0" (batch, 44) and outputs version_number / memory_size / continuous_actions /
continuous_action_output_shape / deterministic_continuous_actions.
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from model import ActorCritic

PROJECT = Path(__file__).resolve().parent.parent
GAME_MODEL = PROJECT / "Assets" / "Resources" / "ML" / "TMRacer.onnx"


def build_game():
    """Rebuilds the standalone game with Unity in batch mode (the Editor must be closed: it locks the project)."""
    version = re.search(r"m_EditorVersion: (\S+)", (PROJECT / "ProjectSettings" / "ProjectVersion.txt").read_text())[1]
    unity = Path.home() / "Unity" / "Hub" / "Editor" / version / "Editor" / "Unity"
    log = Path(__file__).resolve().parent / "logs" / "game_build.log"
    log.parent.mkdir(exist_ok=True)
    print(f"building the game with Unity {version} (a few minutes) ...")
    subprocess.run([str(unity), "-batchmode", "-quit", "-projectPath", str(PROJECT),
                    "-executeMethod", "TrackRush.EditorTools.GameBuild.Build", "-logFile", str(log)])
    if "Game built:" in log.read_text(errors="ignore"):
        print(f"done: start the game with {PROJECT / 'Builds' / 'TrackRush' / 'TrackRush.sh'}")
    else:
        print(f"build failed, see {log} (is the Editor open? close it, or use Track Rush > Build Game (Linux) there)")


class UnityPolicy(nn.Module):
    def __init__(self, model: ActorCritic, act_dim: int):
        super().__init__()
        self.model = model
        self.version_number = nn.Parameter(torch.tensor([3.0]), requires_grad=False)  # ML-Agents model API 2.0
        self.memory_size = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.action_shape = nn.Parameter(torch.tensor([float(act_dim)]), requires_grad=False)

    def forward(self, obs_0):
        action = torch.clamp(self.model.act_deterministic(obs_0), -1.0, 1.0)
        return self.version_number, self.memory_size, action, self.action_shape, action


def export(model: ActorCritic, obs_dim: int, act_dim: int, path: Path):
    torch.onnx.export(
        UnityPolicy(model, act_dim).eval(),
        (torch.zeros(1, obs_dim),),
        str(path),
        input_names=["obs_0"],
        output_names=["version_number", "memory_size", "continuous_actions",
                      "continuous_action_output_shape", "deterministic_continuous_actions"],
        dynamic_axes={"obs_0": {0: "batch"}, "continuous_actions": {0: "batch"},
                      "deterministic_continuous_actions": {0: "batch"}},
        opset_version=13,
        dynamo=False,   # the classic exporter; Unity's inference engine reads its output
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--install", action="store_true", help=f"also copy the model to {GAME_MODEL}")
    p.add_argument("--build", action="store_true", help="with --install: also rebuild the standalone game")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = Path(args.checkpoint).with_name("TMRacer.onnx")
    export(model, ckpt["obs_dim"], ckpt["act_dim"], out)
    print(f"wrote {out}")
    if args.install:
        GAME_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, GAME_MODEL)
        print(f"installed {GAME_MODEL}")
        if args.build:
            build_game()
        else:
            print("now rebuild the game (Track Rush > Build Game (Linux), or add --build), then turn on AI RIVAL")
    else:
        print("run again with --install to put it in the game")


if __name__ == "__main__":
    main()
