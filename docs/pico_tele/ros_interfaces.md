# PICO ROS 2 interfaces

## Nodes

- `pico_sdk_bridge`: XRoboToolkit SDK callback to synchronized `PicoState`.
- `pico_gesture_mapper`: button-edge events and safe dual-button gesture recognition.
- `pico_command_router`: robot-agnostic operator commands to standard ROS trigger topics.

All nodes are launched by `pico_tele.launch.py` and configured by `pico_tele.yaml`.

## Topics

| Topic | Type | QoS | Purpose |
|---|---|---|---|
| `/pico_tele/state` | `pico_tele_interfaces/PicoState` | sensor data | Synchronized source timestamp, head, and controllers |
| `/pico_tele/button_event` | `pico_tele_interfaces/ButtonEvent` | reliable | Digital button press/release edges |
| `/pico_tele/operator_command` | `pico_tele_interfaces/OperatorCommand` | reliable | Semantic reset or control-mode request |
| `/pico_tele/reset_request` | `std_msgs/Empty` | reliable | Generic reset request for a robot adapter |
| `/change_ctrl_mode` | `std_msgs/Bool` | reliable | `true` pulse for the existing HIL topic mode switcher |

`PicoState` contains independent pose and controller-input validity flags. Consumers must not infer valid buttons from
an invalid controller input block, and must not use a pose unless its `valid` field is true.

## Gesture contract

- Both primary buttons continuously held for `1.0s` emit one `COMMAND_RESET`.
- Both secondary buttons continuously held for `1.0s` emit one `COMMAND_TOGGLE_CONTROL_MODE`.
- A command is one-shot until both buttons in that chord are released.
- Simultaneous primary and secondary chords suppress both commands.
- A `0.25s` input gap cancels any pending hold.

The reset output is intentionally only a request. A robot-specific node may subscribe to
`/pico_tele/reset_request` and execute the appropriate reset service/action. No robot message package is required by
the PICO bridge.

## Parameters

Important parameters include:

- `device_id`: fixed device filter; empty selects the first device.
- `frame_id`: native PICO tracking frame, default `pico_tracking_origin`.
- `sdk_init_retry_s`: PC Service connection retry interval.
- `hold_duration_s` and `input_timeout_s`: gesture safety timing.
- `reset_topic`, `toggle_topic`, `enable_reset`, and `enable_toggle`: generic routing controls.

Topic names can also be remapped using normal ROS 2 launch arguments or remapping syntax.
