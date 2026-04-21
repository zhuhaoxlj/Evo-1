#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evo1_lerobot"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evo1_metaworld_custom.grasp_dial_env import TASK_NAME, TASK_PROMPT, make_custom_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset


MOVE_BINDINGS = {
    ord("w"): np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("s"): np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("a"): np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("d"): np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("r"): np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("f"): np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("u"): np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("o"): np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ord("j"): np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
    ord("l"): np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    ord("n"): np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
    ord("m"): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyboard teleop collection for grasp-dial-turn MetaWorld task in LeRobot format."
    )
    parser.add_argument("--repo-id", default="local/grasp_dial_turn")
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "grasp_dial_turn_lerobot"))
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-name", default="corner2")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def overlay_status(
    frame: np.ndarray,
    episode_idx: int,
    reward: float,
    info: dict[str, float],
    state: np.ndarray,
    gripper_closed: bool,
) -> np.ndarray:
    canvas = frame.copy()
    lines = [
        f"Episode {episode_idx}",
        f"Reward {reward:.2f}",
        f"Angle {info.get('dial_angle', 0.0):.2f} / target 1.20",
        f"Angle error {info.get('obj_to_target', 0.0):.3f}",
        f"Grasp {info.get('grasp_success', 0.0):.0f}  Success {info.get('success', 0.0):.0f}",
        f"Roll {state[4]:.2f}  Pitch {state[5]:.2f}  Yaw {state[6]:.2f}",
        f"Gripper {'closed' if gripper_closed else 'open'}",
        "Move: W/S A/D R/F",
        "Roll: U/O  Pitch: J/L  Yaw: N/M",
        "Toggle gripper: SPACE",
        "Reset episode: X",
        "Save success now: C",
        "Quit: Q",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def build_dataset(root: Path, repo_id: str, fps: int, width: int, height: int, resume: bool) -> LeRobotDataset:
    if resume and root.exists():
        return LeRobotDataset(repo_id, root=root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="metaworld_sawyer",
        fps=fps,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        },
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(dataset_root, args.repo_id, args.fps, args.width, args.height, args.resume)

    env = make_custom_env(
        TASK_NAME,
        render_mode="rgb_array",
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
    )
    if env is None:
        raise RuntimeError(f"Failed to create custom env '{TASK_NAME}'")

    obs, info = env.reset()
    del obs, info

    episode_idx = dataset.num_episodes
    gripper_closed = False
    last_step_time = time.perf_counter()
    neutral = np.zeros(7, dtype=np.float32)

    cv2.namedWindow("grasp-dial-turn", cv2.WINDOW_NORMAL)

    try:
        while episode_idx < args.num_episodes:
            frame = env.render()
            preview_state = env.get_proprio_state()
            preview = overlay_status(
                frame,
                episode_idx,
                0.0,
                {"dial_angle": preview_state[-1]},
                preview_state,
                gripper_closed,
            )
            cv2.imshow("grasp-dial-turn", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                gripper_closed = not gripper_closed
            if key == ord("x"):
                dataset.clear_episode_buffer()
                env.reset()
                gripper_closed = False
                continue

            action = neutral.copy()
            if key in MOVE_BINDINGS:
                action = MOVE_BINDINGS[key].copy()
            action[-1] = 1.0 if gripper_closed else -1.0

            now = time.perf_counter()
            dt = now - last_step_time
            frame_dt = 1.0 / args.fps
            if dt < frame_dt:
                time.sleep(frame_dt - dt)
            last_step_time = time.perf_counter()

            next_obs, reward, terminated, truncated, info = env.step(action)
            state = env.get_proprio_state()
            show = overlay_status(frame, episode_idx, float(reward), info, state, gripper_closed)
            cv2.imshow("grasp-dial-turn", cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
            dataset.add_frame(
                {
                    "observation.image": frame.copy(),
                    "observation.state": state.astype(np.float32),
                    "action": action.astype(np.float32),
                    "task": TASK_PROMPT,
                }
            )

            manual_save = key == ord("c")
            if info.get("success", 0.0) or manual_save:
                dataset.save_episode()
                episode_idx += 1
                print(f"[saved] episode={episode_idx} root={dataset_root}")
                env.reset()
                gripper_closed = False
                continue

            if terminated or truncated:
                print("[drop] episode terminated without success; clearing buffer")
                dataset.clear_episode_buffer()
                env.reset()
                gripper_closed = False
                continue

            _ = next_obs
    finally:
        dataset.finalize()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
