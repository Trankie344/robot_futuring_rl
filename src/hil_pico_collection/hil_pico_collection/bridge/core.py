"""Synchronized RL Token request and interruptible chunk execution core."""

from __future__ import annotations

from typing import Any

from hil_pico_collection.protocol_config import RobotProtocolConfig
from hil_pico_collection.recording.cache import StaleDataError

from .execution import ExecutionResult, execute_action_chunk
from .protocol import build_observation


class RLTokenBridgeCore:
    def __init__(
        self,
        *,
        status_cache: Any,
        image_cache: Any,
        adapter: Any,
        policy_client: Any,
        publish_command: Any,
        stamp: Any,
        prompt: str,
        protocol: RobotProtocolConfig | None = None,
        state_max_age_s: float = 0.25,
        image_max_age_s: float = 0.25,
        frame_id: str = "rl_token_stage2",
        monotonic: Any = None,
        sleep: Any = None,
    ) -> None:
        self.status_cache = status_cache
        self.image_cache = image_cache
        self.adapter = adapter
        self.policy_client = policy_client
        self.publish_command = publish_command
        self.stamp = stamp
        self.prompt = prompt
        self.protocol = protocol
        self.state_max_age_s = float(state_max_age_s)
        self.image_max_age_s = float(image_max_age_s)
        self.frame_id = str(frame_id)
        self._execution_kwargs = {}
        if monotonic is not None:
            self._execution_kwargs["monotonic"] = monotonic
        if sleep is not None:
            self._execution_kwargs["sleep"] = sleep

    def ready(self) -> bool:
        try:
            status, _ = self.status_cache.snapshot(self.state_max_age_s)
            sample = self.adapter.parse_status(status)
            return sample.model_control_enabled and not sample.intervention
        except Exception:
            return False

    def infer_and_execute(self) -> ExecutionResult:
        try:
            status, _ = self.status_cache.snapshot(self.state_max_age_s)
            sample = self.adapter.parse_status(status)
            if not sample.model_control_enabled or sample.intervention:
                return ExecutionResult(0, False, "model control is not enabled")
            images, _ = self.image_cache.snapshot(self.image_max_age_s)
            observation = build_observation(images, sample.state, self.prompt, self.protocol)
            response = self.policy_client.infer(observation)
        except StaleDataError as exc:
            return ExecutionResult(0, False, f"observation is stale: {exc}")
        except Exception as exc:
            return ExecutionResult(0, False, f"inference failed: {exc}")

        def guard() -> bool:
            latest, _ = self.status_cache.snapshot(self.state_max_age_s)
            current = self.adapter.parse_status(latest)
            return current.model_control_enabled and not current.intervention

        execution_config = {}
        if self.protocol is not None:
            execution_config = {
                "command_hz": self.protocol.command_hz,
                "output_count": self.protocol.command_count,
                "expected_horizon": self.protocol.action_horizon,
                "expected_action_dim": self.protocol.action.dimension,
            }
        return execute_action_chunk(
            response["actions"],
            guard=guard,
            build_command=self.adapter.build_command,
            publish_command=self.publish_command,
            stamp=self.stamp,
            frame_id=self.frame_id,
            **execution_config,
            **self._execution_kwargs,
        )
