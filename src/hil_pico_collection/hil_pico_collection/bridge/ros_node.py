"""ROS2 node that publishes RL Token absolute actions directly to /auto_arm_cmd."""

from __future__ import annotations

import argparse
import threading
from typing import Any

from hil_pico_collection.adapters.loader import load_configured_robot_adapter, load_robot_adapter
from hil_pico_collection.protocol_config import RobotProtocolConfig, default_robot_config_path, import_symbol
from hil_pico_collection.recording.cache import LatestImageCache, LatestValueCache
from hil_pico_collection.ros.images import RosImageDecoder, prepare_configured_image
from hil_pico_collection.ros.recorder_node import _import_ros_runtime

from .core import RLTokenBridgeCore
from .protocol import RLTokenPolicyClient

COMMAND_TOPIC = "/auto_arm_cmd"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8011


def _make_bridge_node_class(ros: Any):
    class RLTokenBridgeNode(ros.Node):
        def __init__(
            self,
            *,
            adapter: Any,
            protocol: RobotProtocolConfig,
            host: str,
            port: int,
            api_key: str | None,
            prompt: str,
            state_max_age_s: float,
            image_max_age_s: float,
        ) -> None:
            super().__init__("hil_pico_rl_token_bridge")
            self._adapter = adapter
            self._protocol = protocol
            self._host = host
            self._port = int(port)
            self._api_key = api_key
            self._prompt = prompt
            self._state_max_age_s = float(state_max_age_s)
            self._image_max_age_s = float(image_max_age_s)
            self._status_cache = LatestValueCache()
            self._image_cache = LatestImageCache(protocol.image_names)
            self._decoder = RosImageDecoder(ros.CvBridge)
            self._publisher = self.create_publisher(
                adapter.command_message_type,
                protocol.command_topic,
                10,
            )
            self.create_subscription(
                adapter.status_message_type,
                protocol.status_topic,
                self._status_cache.update,
                10,
            )
            for image_spec in protocol.images:
                self.create_subscription(
                    import_symbol(image_spec.message_type),
                    image_spec.topic,
                    self._image_callback(image_spec),
                    ros.qos_profile_sensor_data,
                )
            self._stop = threading.Event()
            self._worker = threading.Thread(target=self._run, name="rl-token-bridge", daemon=True)
            self._worker.start()

        def destroy_node(self) -> Any:
            self._stop.set()
            self._worker.join(timeout=2.0)
            return super().destroy_node()

        def _image_callback(self, image_spec: Any):
            def callback(message: Any) -> None:
                try:
                    image = prepare_configured_image(self._decoder.decode(message), image_spec)
                    self._image_cache.update(image_spec.name, image)
                except Exception as exc:
                    self.get_logger().warn(f"failed to decode {image_spec.name} image: {exc}")

            return callback

        def _run(self) -> None:
            client = None
            while not self._stop.is_set():
                if not self._model_control_ready():
                    self._stop.wait(0.1)
                    continue
                try:
                    if client is None:
                        client = RLTokenPolicyClient(
                            self._host,
                            self._port,
                            api_key=self._api_key,
                            protocol=self._protocol,
                        )
                        self.get_logger().info("connected to validated RL Token Stage 2 server")
                    core = RLTokenBridgeCore(
                        status_cache=self._status_cache,
                        image_cache=self._image_cache,
                        adapter=self._adapter,
                        policy_client=client,
                        publish_command=self._publisher.publish,
                        stamp=lambda: self.get_clock().now().to_msg(),
                        prompt=self._prompt,
                        protocol=self._protocol,
                        state_max_age_s=self._state_max_age_s,
                        image_max_age_s=self._image_max_age_s,
                    )
                    result = core.infer_and_execute()
                    if not result.completed:
                        self.get_logger().warn(
                            f"RL Token chunk discarded after {result.sent_count} commands: {result.reason}"
                        )
                        if result.reason and result.reason.startswith("inference failed"):
                            client.close()
                            client = None
                        self._stop.wait(0.05)
                except Exception as exc:
                    self.get_logger().error(f"RL Token bridge connection failed: {exc}")
                    if client is not None:
                        client.close()
                    client = None
                    self._stop.wait(1.0)
            if client is not None:
                client.close()

        def _model_control_ready(self) -> bool:
            try:
                message, _ = self._status_cache.snapshot(self._state_max_age_s)
                sample = self._adapter.parse_status(message)
                return sample.model_control_enabled and not sample.intervention
            except Exception:
                return False

    return RLTokenBridgeNode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge OpenPI RL Token actions to ROS2.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--robot-config", default=str(default_robot_config_path()))
    parser.add_argument("--robot-adapter")
    parser.add_argument("--state-max-age-s", type=float, default=0.25)
    parser.add_argument("--image-max-age-s", type=float, default=0.25)
    return parser


def main(argv: Any = None) -> int:
    args, ros_args = build_arg_parser().parse_known_args(argv)
    ros = _import_ros_runtime()
    protocol, configured_adapter = load_configured_robot_adapter(args.robot_config)
    adapter = configured_adapter if args.robot_adapter is None else load_robot_adapter(args.robot_adapter)
    ros.rclpy.init(args=ros_args)
    node = _make_bridge_node_class(ros)(
        adapter=adapter,
        protocol=protocol,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        prompt=args.prompt,
        state_max_age_s=args.state_max_age_s,
        image_max_age_s=args.image_max_age_s,
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
