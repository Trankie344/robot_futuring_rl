"""Toggle autonomous/PICO control from the generic /change_ctrl_mode topic."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

from hil_pico_collection.adapters.loader import load_configured_robot_adapter, load_robot_adapter
from hil_pico_collection.protocol_config import default_robot_config_path
from hil_pico_collection.recording.cache import LatestValueCache, StaleDataError

CHANGE_CTRL_MODE_TOPIC = "/change_ctrl_mode"
STATUS_TOPIC = "/arm_status"
DEFAULT_STATUS_MAX_AGE_S = 0.5


def _import_ros_runtime() -> SimpleNamespace:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool

    return SimpleNamespace(rclpy=rclpy, Node=Node, Bool=Bool)


def _make_mode_switcher_node_class(ros: Any):
    class ModeSwitcherNode(ros.Node):
        def __init__(
            self,
            adapter: Any,
            status_max_age_s: float,
            topic: str,
            status_topic: str = STATUS_TOPIC,
        ) -> None:
            super().__init__("hil_pico_mode_switcher")
            self._adapter = adapter
            self._status_max_age_s = float(status_max_age_s)
            self._status_cache = LatestValueCache()
            self._controller = adapter.create_mode_controller(self)
            self.create_subscription(adapter.status_message_type, status_topic, self._on_status, 10)
            self.create_subscription(ros.Bool, topic, self._on_toggle, 10)

        def _on_status(self, message: Any) -> None:
            try:
                sample = self._adapter.parse_status(message)
            except Exception as exc:
                self.get_logger().warn(f"invalid robot status: {exc}")
                return
            self._status_cache.update(sample)

        def _on_toggle(self, message: Any) -> None:
            if not bool(message.data):
                return
            try:
                sample, _ = self._status_cache.snapshot(self._status_max_age_s)
            except StaleDataError as exc:
                self.get_logger().warn(f"mode toggle ignored without fresh status: {exc}")
                return
            if not self._controller.toggle(sample):
                self.get_logger().warn("mode toggle ignored while another request is active or mode is unsupported")

    return ModeSwitcherNode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toggle robot control mode from a Bool topic.")
    parser.add_argument("--robot-config", default=str(default_robot_config_path()))
    parser.add_argument("--robot-adapter")
    parser.add_argument("--topic")
    parser.add_argument("--status-max-age-s", type=float, default=DEFAULT_STATUS_MAX_AGE_S)
    return parser


def main(argv: Any = None) -> int:
    args, ros_args = build_arg_parser().parse_known_args(argv)
    ros = _import_ros_runtime()
    protocol, configured_adapter = load_configured_robot_adapter(args.robot_config)
    adapter = configured_adapter if args.robot_adapter is None else load_robot_adapter(args.robot_adapter)
    ros.rclpy.init(args=ros_args)
    node = _make_mode_switcher_node_class(ros)(
        adapter,
        args.status_max_age_s,
        args.topic or protocol.change_control_mode_topic,
        protocol.status_topic,
    )
    try:
        ros.rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if ros.rclpy.ok():
            ros.rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
