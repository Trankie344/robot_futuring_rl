# PICO tracking protocol

This bridge targets the official
[XR-Robotics](https://github.com/XR-Robotics) XRoboToolkit Unity client at commit
[`c9326092ff4d11e8b507b041713194b93470a8e1`](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/commit/c9326092ff4d11e8b507b041713194b93470a8e1).

## Transport boundary

The PICO application connects to XRoboToolkit PC Service over TCP port `63901`. The client protocol contains a
framed command stream, heartbeats, device registration, and tracking JSON. OpenPI does not reimplement that
transport. `pico_sdk_bridge` links the vendor `libPXREARobotSDK` and consumes `PXREADeviceStateJson` callbacks from
the locally running PC Service.

The vendor callback contains a device ID and an outer JSON envelope:

```json
{
  "functionName": "Tracking",
  "value": "{\"timeStampNs\":1700000000000000000,\"Input\":1,...}"
}
```

`value` is normally an escaped JSON string. The parser also accepts an object for compatibility with record/replay
tools. Other `functionName` values are ignored.

## Tracking value

The supported fields are:

- `timeStampNs`: non-negative Unix epoch timestamp generated on the PICO device.
- `Input`: PICO active input-device enum.
- `Head.pose`: `x,y,z,qx,qy,qz,qw`, plus integer `status`.
- `Controller.left` and `Controller.right`: pose, joystick axes/click, grip, trigger, primary, secondary, and menu.

Pose may be a comma-separated string, as emitted by the Unity client, or a seven-number JSON array. Values must be
finite and quaternions must be non-zero. Quaternions are normalized to remove small numerical drift.

Malformed outer JSON, source timestamps, or input modes reject the whole frame. A malformed or missing head or
controller marks only that device field invalid so that the remainder of the synchronized frame can still be used.

## Coordinate and time semantics

No robot coordinate conversion is performed. All poses retain the PICO native right-handed tracking coordinates:

- X: right
- Y: up
- Z: inward

The ROS frame is `pico_tracking_origin`. `PicoState.header.stamp` is the PICO source time and `receipt_stamp` is the
host ROS clock time when the SDK callback was handled. Clock synchronization is required before interpreting their
difference as network latency.

By default the bridge locks to the first device that produces an online/tracking callback. Set the `device_id` ROS
parameter to select a specific PICO when multiple devices share one PC Service.

## Provenance

The callback and JSON contracts are taken from the official
[XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) repository, including
[`PXREARobotSDK.h`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/blob/85bac4dbc1fd5cef42c74a160d9c30aa3491f122/RoboticsService/SDK/include/PXREARobotSDK.h),
its SDK demos, and the official
[PICO ROS integration toolkit](https://github.com/XR-Robotics/XRoboToolkit-Teleop-ROS). The restored Apache-2.0 ROS
package at `restored/robot/rootfs/home/zme/pico_tele_zme` was used only as the migration input. Robot-specific
`arm_interfaces`, `/tele_vr_cmd`, and ZME coordinate conversion logic are intentionally not migrated.
