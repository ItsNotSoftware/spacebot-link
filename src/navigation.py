"""Navigation state and helpers for goal/follow modes and abort flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple


from utils import panda_pose_to_ros, panda_pose_to_ros_tuple
from config import (
    FOLLOW_HPR_EPS,
    FOLLOW_POS_EPS,
    FOLLOW_REACHED_THRESH,
    FOLLOW_SAMPLE_PERIOD,
    TOPIC_CMD_PATH,
    TOPIC_GOAL,
)

PoseTuple = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


@dataclass
class NavState:
    mode: str = "Goal Mode"
    nav_publishing_enabled: bool = True
    move_target: str = "Avatar"
    last_goal_pose: Optional[PoseTuple] = None
    last_robot_pose_panda: Optional[PoseTuple] = None
    last_ros_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    last_ros_orientation: Optional[Dict[str, float]] = None
    last_robot_hpr: Optional[Tuple[float, float, float]] = None
    follow_path_points: List[PoseTuple] = field(default_factory=list)


class Navigation:
    """Encapsulates nav goal/follow logic and abort flows."""

    def __init__(self, cmd_pub: Any, task_mgr: Any) -> None:
        """Wire navigation helpers with publishers and Panda3D task manager."""
        self.cmd_pub = cmd_pub
        self.task_mgr = task_mgr
        self.state = NavState()
        self._follow_sample_period = FOLLOW_SAMPLE_PERIOD
        self._follow_hpr_eps = FOLLOW_HPR_EPS
        self._follow_pos_eps = FOLLOW_POS_EPS
        self._follow_reached_thresh = FOLLOW_REACHED_THRESH

    @property
    def follow_sample_period(self) -> float:
        return self._follow_sample_period

    # ---- mode management ----
    def set_mode(self, mode: str) -> None:
        """Set UI mode and reset state as needed."""
        self.state.mode = mode
        if mode == "Goal Mode":
            # seed so we don't immediately publish
            self.state.last_goal_pose = None
            self.state.follow_path_points.clear()
        else:
            self.state.follow_path_points.clear()

    def set_move_target(self, target: str) -> None:
        """Update the label for the move target."""
        self.state.move_target = target

    # ---- robot pose updates ----
    def update_robot_pose(self, pos_hpr: PoseTuple, ros_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None, ori: Optional[Dict[str, float]] = None) -> None:
        """Track the latest robot pose in both Panda and ROS frames."""
        self.state.last_robot_pose_panda = pos_hpr
        self.state.last_robot_hpr = pos_hpr[1]
        if ros_pose is not None:
            self.state.last_ros_pose = ros_pose
        if ori is not None:
            self.state.last_ros_orientation = ori

    # ---- follow mode helpers ----
    def should_append_follow_pose(self, pose: PoseTuple) -> bool:
        """Return True when avatar moved enough to enqueue a new follow waypoint."""
        if not self.state.follow_path_points:
            return True
        (x1, y1, z1), (h1, p1, r1) = self.state.follow_path_points[-1]
        (x2, y2, z2), (h2, p2, r2) = pose
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        dh = abs(h2 - h1)
        dp = abs(p2 - p1)
        dr = abs(r2 - r1)
        return (
            dx > self._follow_pos_eps
            or dy > self._follow_pos_eps
            or dz > self._follow_pos_eps
            or dh > self._follow_hpr_eps
            or dp > self._follow_hpr_eps
            or dr > self._follow_hpr_eps
        )

    def prune_follow_path(self) -> None:
        """Drop waypoints already reached by the robot."""
        if self.state.last_robot_pose_panda is None or not self.state.follow_path_points:
            return
        (rx, ry, rz), _ = self.state.last_robot_pose_panda
        while len(self.state.follow_path_points) > 1:
            (px, py, pz), _ = self.state.follow_path_points[0]
            dist = sqrt((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2)
            if dist <= self._follow_reached_thresh:
                self.state.follow_path_points.pop(0)
            else:
                break

    def publish_follow_path(self) -> None:
        """Send the current follow path as a nav_msgs/Path-like dict."""
        if not self.state.nav_publishing_enabled:
            return
        if not self.state.follow_path_points:
            return
        points = list(self.state.follow_path_points)
        if len(points) == 1 and self.state.last_robot_pose_panda is not None:
            points = [self.state.last_robot_pose_panda] + points
        poses = []
        for pos_hpr in points:
            ros_pose = panda_pose_to_ros(pos_hpr)
            if ros_pose is None:
                continue
            poses.append({"pose": ros_pose})
        if not poses:
            return
        msg = {"header": {"frame_id": "map"}, "poses": poses}
        try:
            self.cmd_pub.publish(TOPIC_CMD_PATH, msg)
        except Exception:
            pass

    # ---- goal publishing ----
    def pose_changed_since_last_goal(self, pose: PoseTuple, pos_eps: float = 1e-4, hpr_eps: float = 1e-3) -> bool:
        """Check if the pose moved enough to republish a goal."""
        if self.state.last_goal_pose is None:
            return True
        (x1, y1, z1), (h1, p1, r1) = self.state.last_goal_pose
        (x2, y2, z2), (h2, p2, r2) = pose
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        dh = abs(h2 - h1)
        dp = abs(p2 - p1)
        dr = abs(r2 - r1)
        return (
            (dx > pos_eps)
            or (dy > pos_eps)
            or (dz > pos_eps)
            or (dh > hpr_eps)
            or (dp > hpr_eps)
            or (dr > hpr_eps)
        )

    def publish_goal_for_pose(self, pose: PoseTuple) -> None:
        """Publish a single goal pose to the robot."""
        if not self.state.nav_publishing_enabled:
            return
        ros_pose = panda_pose_to_ros(pose)
        if ros_pose is None:
            return
        msg = {"header": {"frame_id": "map"}, "pose": ros_pose}
        try:
            self.cmd_pub.publish(TOPIC_GOAL, msg)
            self.state.last_goal_pose = pose
        except Exception:
            pass

    # ---- abort handling ----
    def abort_to_robot_pose(self, renderer: Any) -> None:
        """Abort motion: command hold at current robot pose and sync avatar."""
        if self.state.last_robot_pose_panda is None:
            return
        pose = renderer.sync_avatar_to_robot(self.state.last_robot_pose_panda)
        ros_pose = None
        if self.state.last_ros_pose is not None and self.state.last_ros_orientation is not None:
            # Prefer publishing using the freshest ROS pose to avoid frame drift.
            pos = self.state.last_ros_pose[0]
            ros_pose = {
                "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "orientation": {
                    "x": float(self.state.last_ros_orientation.get("x", 0.0)),
                    "y": float(self.state.last_ros_orientation.get("y", 0.0)),
                    "z": float(self.state.last_ros_orientation.get("z", 0.0)),
                    "w": float(self.state.last_ros_orientation.get("w", 1.0)),
                },
            }
        else:
            ros_pose = panda_pose_to_ros(pose)
        if ros_pose is not None:
            try:
                self.cmd_pub.publish(TOPIC_GOAL, {"header": {"frame_id": "map"}, "pose": ros_pose})
            except Exception:
                pass
            self.state.last_goal_pose = pose
        renderer.publish_hold_path(self.cmd_pub, pose, ros_pose_override=ros_pose)
        if self.state.mode == "Follow Mode":
            self.state.follow_path_points = []

    # ---- path helpers ----
    def set_follow_seed(self, pose: PoseTuple) -> None:
        """Clear and seed follow buffer with a starting pose."""
        self.state.follow_path_points.clear()
        self.state.follow_path_points.append(pose)

    def follow_tick(self, renderer: Any) -> None:
        """Sample avatar pose and publish follow path if in Follow Mode."""
        if self.state.mode != "Follow Mode":
            return
        self.prune_follow_path()
        pose = renderer.get_avatar_pose()
        if self.should_append_follow_pose(pose):
            self.state.follow_path_points.append(pose)
        self.publish_follow_path()

    def render_path_overlay(self, renderer: Any, poses: List[PoseTuple]) -> None:
        """Render provided poses as ghost markers."""
        renderer.render_path_markers(poses)

    # ---- status helpers ----
    def last_goal_pose_ros_tuple(self) -> Optional[PoseTuple]:
        """Return last goal pose converted to ROS tuple, if available."""
        return panda_pose_to_ros_tuple(self.state.last_goal_pose) if self.state.last_goal_pose else None
