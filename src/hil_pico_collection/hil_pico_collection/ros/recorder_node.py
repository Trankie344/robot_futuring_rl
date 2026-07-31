"""ROS2 recorder node for robot-independent HIL PICO correction data."""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hil_pico_collection.adapters.loader import load_configured_robot_adapter, load_robot_adapter
from hil_pico_collection.protocol_config import RobotProtocolConfig, default_robot_config_path, import_symbol
from hil_pico_collection.recording.cache import LatestImageCache, LatestValueCache
from hil_pico_collection.recording.recorder_core import RecorderCore
from hil_pico_collection.recording.v21_writer import LeRobotV21Writer
from hil_pico_collection.web.app import create_app

from .images import RosImageDecoder, prepare_configured_image

IMAGE_KEYS = ("top", "left_wrist", "right_wrist")
IMAGE_TOPICS = {
    "top": "/camera_dcw2/custom_cam_color_1M",
    "left_wrist": "/camera_dcl_left/custom_cam_color_1M",
    "right_wrist": "/camera_dcl_right/custom_cam_color_1M",
}
STATUS_TOPIC = "/arm_status"
ACTION_TOPIC = "/auto_arm_cmd"
RESET_REQUEST_TOPIC = "/pico_tele/reset_request"
DEFAULT_FPS = 30
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 8088
DATE_SUFFIX_RE = re.compile(r"_\d{8}$")


class ThreadSafeRecorderCore:
    def __init__(self, core: RecorderCore) -> None:
        self._core = core
        self._lock = threading.RLock()

    @property
    def raw_core(self) -> RecorderCore:
        return self._core

    def __getattr__(self, name: str) -> Any:
        with self._lock:
            return getattr(self._core, name)

    def start_episode(self, task: Any) -> Any:
        with self._lock:
            return self._core.start_episode(task)

    def record_tick(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._core.record_tick(*args, **kwargs)

    def end_episode(self) -> Any:
        with self._lock:
            return self._core.end_episode()


@dataclass
class RecorderRuntime:
    status_cache: LatestValueCache
    action_cache: LatestValueCache
    image_cache: LatestImageCache
    core: ThreadSafeRecorderCore
    writer: LeRobotV21Writer
    adapter: Any
    protocol: RobotProtocolConfig


def runtime_root() -> Path:
    return Path(
        os.environ.get(
            "HIL_PICO_RUNTIME",
            "/mnt/workspace/ys/futuring/openpi_runtime/hil_pico_collection",
        )
    )


def default_dataset_root() -> str:
    return str(runtime_root() / "datasets" / "hil_pico_v21")


def resolve_dataset_root(dataset_root: str | None, today: str | None = None) -> str:
    root = Path(dataset_root or default_dataset_root())
    today = today or datetime.now().strftime("%Y%m%d")
    if DATE_SUFFIX_RE.search(root.name):
        return str(root)
    return str(root.with_name(f"{root.name}_{today}"))


def create_runtime(
    dataset_root: str | None,
    fps: int,
    adapter: Any,
    protocol: RobotProtocolConfig,
) -> RecorderRuntime:
    if int(fps) != DEFAULT_FPS:
        raise ValueError("--fps must be fixed 30 Hz to match the recording contract")
    root = resolve_dataset_root(dataset_root)
    Path(root).mkdir(parents=True, exist_ok=True)
    status_cache = LatestValueCache()
    action_cache = LatestValueCache()
    image_cache = LatestImageCache(protocol.image_names)
    writer = LeRobotV21Writer(
        root=root,
        fps=DEFAULT_FPS,
        robot_type=protocol.robot_type,
        state_names=protocol.state.order,
        action_names=protocol.action.order,
        image_shapes={image.name: image.shape for image in protocol.images},
    )
    raw_core = RecorderCore(status_cache, action_cache, image_cache, adapter)
    existing = writer._active_episode_indices()
    if existing:
        raw_core._next_episode_index = max(existing) + 1
    return RecorderRuntime(
        status_cache=status_cache,
        action_cache=action_cache,
        image_cache=image_cache,
        core=ThreadSafeRecorderCore(raw_core),
        writer=writer,
        adapter=adapter,
        protocol=protocol,
    )


def _import_ros_runtime(camera_message_type: str | None = None) -> SimpleNamespace:
    del camera_message_type
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Empty

    try:
        from cv_bridge import CvBridge
    except ImportError:
        CvBridge = None
    return SimpleNamespace(
        rclpy=rclpy,
        Node=Node,
        qos_profile_sensor_data=qos_profile_sensor_data,
        Empty=Empty,
        CvBridge=CvBridge,
    )


def _make_recorder_node_class(ros: Any):
    class RecorderNode(ros.Node):
        def __init__(self, runtime: RecorderRuntime) -> None:
            super().__init__("hil_pico_recorder")
            self._runtime = runtime
            self._decoder = RosImageDecoder(ros.CvBridge)
            self._last_image_warning: dict[str, float] = {}
            protocol = runtime.protocol
            self._reset_publisher = self.create_publisher(ros.Empty, protocol.reset_request_topic, 10)
            self.create_subscription(
                runtime.adapter.status_message_type,
                protocol.status_topic,
                runtime.status_cache.update,
                10,
            )
            self.create_subscription(
                runtime.adapter.command_message_type,
                protocol.command_topic,
                runtime.action_cache.update,
                10,
            )
            for image_spec in protocol.images:
                self.create_subscription(
                    import_symbol(image_spec.message_type),
                    image_spec.topic,
                    self._image_callback(image_spec),
                    ros.qos_profile_sensor_data,
                )
            self.create_timer(1.0 / DEFAULT_FPS, runtime.core.record_tick)

        def publish_reset_request(self) -> None:
            self._reset_publisher.publish(ros.Empty())

        def _image_callback(self, image_spec: Any):
            def callback(message: Any) -> None:
                try:
                    image = prepare_configured_image(self._decoder.decode(message), image_spec)
                except Exception as exc:
                    now = time.monotonic()
                    if now - self._last_image_warning.get(image_spec.name, 0.0) >= 1.0:
                        self._last_image_warning[image_spec.name] = now
                        self.get_logger().warn(f"failed to decode {image_spec.name} image: {exc}")
                    return
                self._runtime.image_cache.update(image_spec.name, image)

            return callback

    return RecorderNode


def start_web_server(app: Any, host: str, port: int) -> tuple[Any, threading.Thread]:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=int(port), log_level="info"))
    thread = threading.Thread(target=server.run, name="hil-pico-web", daemon=True)
    thread.start()
    return server, thread


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record HIL PICO episodes in LeRobot v2.1 format.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--robot-config", default=str(default_robot_config_path()))
    parser.add_argument("--robot-adapter")
    parser.add_argument("--web-host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    return parser


def main(argv: Any = None) -> int:
    args, ros_args = build_arg_parser().parse_known_args(argv)
    ros = _import_ros_runtime()
    protocol, configured_adapter = load_configured_robot_adapter(args.robot_config)
    adapter = configured_adapter if args.robot_adapter is None else load_robot_adapter(args.robot_adapter)
    runtime = create_runtime(args.dataset_root, args.fps, adapter, protocol)
    ros.rclpy.init(args=ros_args)
    node = _make_recorder_node_class(ros)(runtime)
    app = create_app(runtime.core, runtime.writer, reset_request=node.publish_reset_request)
    server, server_thread = start_web_server(app, args.web_host, args.web_port)
    try:
        ros.rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if ros.rclpy.ok():
            ros.rclpy.shutdown()
        server.should_exit = True
        server_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
