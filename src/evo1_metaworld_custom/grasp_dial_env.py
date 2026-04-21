from __future__ import annotations

from pathlib import Path
from typing import Any

import metaworld
import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils

TASK_NAME = "grasp-dial-turn-v1"
TASK_PROMPT = "Grasp the knob with the gripper and rotate it clockwise."
CUSTOM_TASK_DESCRIPTIONS = {TASK_NAME: TASK_PROMPT}
CUSTOM_TASK_POLICY_MAPPING: dict[str, Any] = {}

_GENERATED_DIR = Path(__file__).resolve().parent / "_generated"
_MODEL_PATH = _GENERATED_DIR / "sawyer_grasp_dial_turn.xml"
_OBJECT_PATH = _GENERATED_DIR / "grasp_dial.xml"


def _metaworld_assets_root() -> Path:
    return Path(metaworld.__file__).resolve().parent / "assets"


def _ensure_xml_assets() -> None:
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    assets_root = _metaworld_assets_root()
    basic_scene = assets_root / "scene" / "basic_scene.xml"
    xyz_base_dependencies = assets_root / "objects" / "assets" / "xyz_base_dependencies.xml"
    xyz_base = assets_root / "objects" / "assets" / "xyz_base.xml"

    _OBJECT_PATH.write_text(
        """<mujocoinclude>
    <body childclass="dial_base">
      <joint name="knob_Joint_1" axis="0 0 1" type="hinge" limited="true" range="-1.8 1.8"/>

      <geom material="dial_metal" pos="0 0 0.018" size="0.05 0.018" type="cylinder"/>
      <geom material="dial_metal" pos="0 0 0.055" size="0.024 0.024" type="cylinder"/>
      <geom material="dial_metal" pos="0 -0.03 0.055" size="0.028 0.008 0.012" type="box"/>
      <geom material="dial_metal" pos="0 0.03 0.055" size="0.028 0.008 0.012" type="box"/>
      <geom material="dial_red" pos="0 -0.045 0.074" size="0.004 0.012 0.006" type="box"/>

      <geom name="dial_base_col" class="dial_col" pos="0 0 0.018" size="0.05 0.018" type="cylinder" friction="2.5 0.4 0.15"/>
      <geom name="dial_head_col" class="dial_col" pos="0 0 0.055" size="0.024 0.024" type="cylinder" friction="2.5 0.4 0.15"/>
      <geom name="dial_tab_neg_col" class="dial_col" pos="0 -0.03 0.055" size="0.028 0.008 0.012" type="box" friction="3.0 0.5 0.2"/>
      <geom name="dial_tab_pos_col" class="dial_col" pos="0 0.03 0.055" size="0.028 0.008 0.012" type="box" friction="3.0 0.5 0.2"/>

      <site name="dial_center" pos="0 0 0.055" size="0.006" rgba="0 0 1 1"/>
      <site name="dial_grasp_neg" pos="0 -0.03 0.055" size="0.005" rgba="0 1 0 1"/>
      <site name="dial_grasp_pos" pos="0 0.03 0.055" size="0.005" rgba="0 1 0 1"/>
    </body>
</mujocoinclude>
""",
        encoding="utf-8",
    )

    _MODEL_PATH.write_text(
        f"""<mujoco>
  <include file="{basic_scene}"/>
  <include file="{xyz_base_dependencies}"/>
  <asset>
    <material name="dial_metal" rgba=".35 .35 .35 1" shininess="1" reflectance=".4" specular=".3"/>
    <material name="dial_red" rgba=".7 .1 .1 1" shininess="1" reflectance=".4" specular=".2"/>
  </asset>
  <default>
    <default class="dial_base">
      <joint armature="0.001" damping="2" limited="true"/>
      <default class="dial_col">
        <geom conaffinity="1" contype="1" condim="4" group="4" solimp="0.99 0.99 0.01" solref="0.01 1"/>
      </default>
    </default>
  </default>
  <worldbody>
    <include file="{xyz_base}"/>
    <body name="dial" pos="0 0.7 0.0">
      <include file="{_OBJECT_PATH}"/>
      <site name="dialStart" pos="0 -0.05 0.035" size="0.005" rgba="0 0 1 1"/>
    </body>
    <site name="goal" pos="0.0 0.74 0.07" size="0.02" rgba=".8 0 0 1"/>
  </worldbody>
  <actuator>
    <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400" user="1"/>
    <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400" user="1"/>
  </actuator>
  <equality>
    <weld body1="mocap" body2="hand" solref="0.02 1"/>
  </equality>
</mujoco>
""",
        encoding="utf-8",
    )


class GraspDialTurnEnvV1(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.15
    TARGET_ANGLE_DELTA: float = 1.2
    BASE_MOCAP_QUAT = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    handles_corner2_flip = True

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v1",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.08, 0.68, 0.0)
        obj_high = (0.08, 0.82, 0.0)
        goal_low = (-0.08, 0.68, 0.0)
        goal_high = (0.08, 0.82, 0.0)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.reward_function_version = reward_function_version
        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0.0, 0.7, 0.0], dtype=np.float32),
            "hand_init_pos": np.array([0.0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.goal = self.obj_init_pos.copy()
        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low),
            np.array(goal_high),
            dtype=np.float64,
        )
        self._target_angle = self.TARGET_ANGLE_DELTA
        self._angle_margin = self.TARGET_ANGLE_DELTA
        self._consecutive_hold_steps = 0
        self._last_grasp_success = False
        self._wrist_rpy = np.zeros(3, dtype=np.float64)
        self.action_space = Box(
            np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_dial.xml")

    def render(self) -> npt.NDArray[np.uint8]:
        image = super().render()
        if self.camera_name == "corner2":
            image = np.flip(image, (0, 1))
        return image

    def _quat_from_axis_angle(self, axis: npt.NDArray[Any], angle: float) -> npt.NDArray[np.float64]:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=np.float64), angle)
        return quat

    def _apply_wrist_rpy(self) -> None:
        roll_quat = self._quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), float(self._wrist_rpy[0]))
        pitch_quat = self._quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), float(self._wrist_rpy[1]))
        yaw_quat = self._quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), float(self._wrist_rpy[2]))

        composed = np.zeros(4, dtype=np.float64)
        temp = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(temp, self.BASE_MOCAP_QUAT, roll_quat)
        mujoco.mju_mulQuat(composed, temp, pitch_quat)
        mujoco.mju_mulQuat(temp, composed, yaw_quat)
        self.data.mocap_quat = temp[None]

    def set_xyz_action_rot(self, action_xyz: npt.NDArray[Any], rpy_delta: npt.NDArray[Any]) -> None:
        target_tcp = self.tcp_center.copy()
        self.set_xyz_action(action_xyz)
        target_tcp = target_tcp + np.clip(action_xyz, -1, 1) * self.action_scale
        self._wrist_rpy = np.clip(
            self._wrist_rpy + np.asarray(rpy_delta, dtype=np.float64) * 0.12,
            np.array([-1.6, -1.2, -1.6], dtype=np.float64),
            np.array([1.6, 1.2, 1.6], dtype=np.float64),
        )
        self._apply_wrist_rpy()
        mujoco.mj_forward(self.model, self.data)
        tcp_after_rot = self.tcp_center.copy()
        compensation = target_tcp - tcp_after_rot
        new_mocap_pos = self.data.mocap_pos.copy()
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :] + compensation,
            self.mocap_low,
            self.mocap_high,
        )
        self.data.mocap_pos = new_mocap_pos
        mujoco.mj_forward(self.model, self.data)

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        dial_center = self.get_body_com("dial").copy()
        dial_angle_rad = self.data.joint("knob_Joint_1").qpos
        offset = np.array(
            [np.sin(dial_angle_rad).item(), -np.cos(dial_angle_rad).item(), 0.0]
        )
        offset *= 0.05
        return dial_center + offset

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("dial").xquat.copy()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self.init_config["obj_init_pos"].copy()
        goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]
        self.model.body("dial").pos = self.obj_init_pos
        self.model.site("goal").pos = self.obj_init_pos + np.array([0.0, 0.03, 0.03])
        self._target_pos = self.model.site("goal").pos.copy()
        self._target_angle = self.TARGET_ANGLE_DELTA
        self._angle_margin = abs(self._target_angle) + 0.2
        self._consecutive_hold_steps = 0
        self._last_grasp_success = False
        self._wrist_rpy[:] = 0.0

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = 0.0
        qvel[9] = 0.0
        self.set_state(qpos, qvel)
        self._apply_wrist_rpy()

        return self._get_obs()

    def _dial_angle(self) -> float:
        return float(self.data.joint("knob_Joint_1").qpos.item())

    def _grasp_targets(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        body = self.data.body("dial")
        origin = body.xpos.copy()
        rotation = body.xmat.reshape(3, 3)
        grasp_neg = origin + rotation @ np.array([0.0, -0.03, 0.055])
        grasp_pos = origin + rotation @ np.array([0.0, 0.03, 0.055])
        return grasp_neg, grasp_pos

    def _finger_alignment_error(self) -> float:
        left_pad = self._get_site_pos("leftEndEffector")
        right_pad = self._get_site_pos("rightEndEffector")
        grasp_neg, grasp_pos = self._grasp_targets()
        direct = np.linalg.norm(left_pad - grasp_pos) + np.linalg.norm(right_pad - grasp_neg)
        swapped = np.linalg.norm(left_pad - grasp_neg) + np.linalg.norm(right_pad - grasp_pos)
        return float(min(direct, swapped) / 2.0)

    def _dial_contact(self) -> tuple[bool, bool]:
        leftpad_geom_id = self.data.geom("leftpad_geom").id
        rightpad_geom_id = self.data.geom("rightpad_geom").id
        target_body_id = self.data.body("dial").id

        left_contact = False
        right_contact = False
        for contact in self.data.contact:
            geom1_body_id = self.model.geom_bodyid[contact.geom1]
            geom2_body_id = self.model.geom_bodyid[contact.geom2]
            touches_dial = geom1_body_id == target_body_id or geom2_body_id == target_body_id
            geom_pair = {contact.geom1, contact.geom2}
            if leftpad_geom_id in geom_pair and touches_dial:
                left_contact = True
            if rightpad_geom_id in geom_pair and touches_dial:
                right_contact = True
        return left_contact, right_contact

    def get_proprio_state(self) -> npt.NDArray[np.float32]:
        gripper = float(np.clip(self._get_curr_obs_combined_no_goal()[3], 0.0, 1.0))
        state = np.array(
            [
                *self.tcp_center.tolist(),
                gripper,
                *self._wrist_rpy.tolist(),
                self._dial_angle(),
            ],
            dtype=np.float32,
        )
        return state

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def step(
        self,
        action: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        assert len(action) == 7, f"Actions should be size 7, got {len(action)}"
        self.set_xyz_action_rot(action[:3], action[3:6])
        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            assert self._last_stable_obs is not None
            return (
                self._last_stable_obs,
                0.0,
                False,
                False,
                {
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self._last_stable_obs = obs
        reward, info = self.evaluate_state(obs, action)
        truncated = self.curr_path_length >= self.max_path_length
        return obs, reward, False, truncated, info

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self,
        obs: npt.NDArray[np.float64],
        action: npt.NDArray[np.float32],
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_grasp,
            alignment_error,
            angle_error,
            object_grasped,
            turn_reward,
        ) = self.compute_reward(action, obs)

        grasp_success = bool(object_grasped >= 0.8)
        if grasp_success and angle_error <= self.TARGET_RADIUS:
            self._consecutive_hold_steps += 1
        else:
            self._consecutive_hold_steps = 0

        success = bool(angle_error <= self.TARGET_RADIUS and self._consecutive_hold_steps >= 5)
        self._last_grasp_success = grasp_success
        info = {
            "success": float(success),
            "near_object": float(tcp_to_grasp <= 0.025),
            "grasp_success": float(grasp_success),
            "grasp_reward": object_grasped,
            "in_place_reward": turn_reward,
            "obj_to_target": angle_error,
            "alignment_error": alignment_error,
            "dial_angle": self._dial_angle(),
            "unscaled_reward": reward,
        }
        return reward, info

    def compute_reward(
        self,
        action: npt.NDArray[Any],
        obs: npt.NDArray[np.float64],
    ) -> tuple[float, float, float, float, float, float]:
        del obs
        tcp = self.tcp_center
        grasp_neg, grasp_pos = self._grasp_targets()
        grasp_center = (grasp_neg + grasp_pos) / 2.0
        tcp_to_grasp = float(np.linalg.norm(tcp - grasp_center))
        alignment_error = self._finger_alignment_error()
        dial_angle = self._dial_angle()
        angle_error = float(abs(self._target_angle - dial_angle))

        tcp_init_dist = float(np.linalg.norm(self.init_tcp - grasp_center))
        reach_reward = reward_utils.tolerance(
            tcp_to_grasp,
            bounds=(0.0, 0.02),
            margin=max(tcp_init_dist, 0.02),
            sigmoid="long_tail",
        )
        alignment_reward = reward_utils.tolerance(
            alignment_error,
            bounds=(0.0, 0.015),
            margin=0.08,
            sigmoid="long_tail",
        )
        gripper_closed = min(max(float(action[-1]), 0.0), 1.0)
        left_contact, right_contact = self._dial_contact()
        contact_bonus = 1.0 if left_contact and right_contact else 0.0
        object_grasped = max(
            reward_utils.hamacher_product(alignment_reward, gripper_closed),
            contact_bonus,
        )
        turn_reward = reward_utils.tolerance(
            angle_error,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=self._angle_margin,
            sigmoid="long_tail",
        )

        reward = 1.5 * reach_reward + 4.0 * object_grasped
        if object_grasped >= 0.8:
            reward += 8.0 * turn_reward
        if angle_error <= self.TARGET_RADIUS and object_grasped >= 0.8:
            reward += 5.0

        return (
            float(reward),
            tcp_to_grasp,
            alignment_error,
            angle_error,
            float(object_grasped),
            float(turn_reward),
        )


def make_custom_env(task_name: str, **kwargs: Any) -> GraspDialTurnEnvV1 | None:
    if task_name != TASK_NAME:
        return None

    env = GraspDialTurnEnvV1(**kwargs)
    env._freeze_rand_vec = False
    env.reset()
    env._set_task_called = True
    return env
