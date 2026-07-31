# 机器人协议配置

HIL recorder、mode switcher 和 RL Token bridge 共用一份 YAML 配置。默认样例是：

[zme_dual_arm.yaml](../../src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml)

启动时通过：

```bash
--robot-config /path/to/robot.yaml
```

指定配置。更换机器人时，通常只需要新增 YAML，不需要修改 recorder 或 bridge 源码。

## 配置范围

`robot` 定义 ROS 接口：

```yaml
robot:
  robot_type: my_robot
  status_topic: /robot/status
  command_topic: /robot/command
  reset_request_topic: /robot/reset_request
  change_control_mode_topic: /change_ctrl_mode
  status_message_type: my_robot_msgs.msg:RobotStatus
  command_message_type: my_robot_msgs.msg:RobotCommand
  header_stamp_path: header.stamp
  header_frame_id_path: header.frame_id
```

消息类型使用 `module:attribute`。ROS 消息 package 必须已经在目标主机安装并 source。

## 状态维度与顺序

`state.order` 是发送给模型并写入 `observation.state` 的唯一顺序。维度必须与 order 长度一致：

```yaml
state:
  dimension: 3
  order: [joint_b, gripper, joint_a]
  sources:
    joint_b: joints[1].position
    gripper: gripper.position
    joint_a: joints[0].position
```

字段路径支持：

- 属性：`arm.position`
- 数组索引：`joints[2]`
- 组合路径：`right_arm[0].joint_status[0]`
- 兼容字段列表：`[right_command[0], right_joint_command[0]]`

兼容字段按顺序尝试，首个可读取或写入的路径生效。

## 动作维度、顺序和限位

`action.order` 同时决定：

- RL Token 响应向量顺序；
- recorder 写入 dataset 的 action 顺序；
- robot command 的字段映射。

```yaml
action:
  dimension: 2
  order: [joint_target, gripper_target]
  observed_sources:
    joint_target: executed_joint[0]
    gripper_target: executed_gripper
  command_targets:
    joint_target: command_joint[0]
    gripper_target: command_gripper
  limits:
    joint_target: {min: -2.0, max: 2.0}
    gripper_target: {min: 0.0, max: 1.0}
```

`observed_sources` 用于 recorder 从机器人实际执行消息中读取 action。`command_targets` 用于 bridge 生成机器人命令。配置层会在发布前执行逐维有限值和软限位检查，本体控制器仍须执行最终安全限制。

## 控制状态

```yaml
status:
  control_mode: mode.primary
  intervention:
    path: mode.primary
    equals: 5
  model_control_enabled:
    - {path: mode.primary, equals: 1}
    - {path: mode.secondary, equals: 1}
```

`model_control_enabled` 中的所有条件必须同时成立。任一条件失效，bridge 都会停止当前 chunk。

## 图像观测

图像 key、topic、ROS 消息类型和模型输入尺寸均由配置决定：

```yaml
images:
  front_rgb:
    topic: /camera/front/image
    message_type: sensor_msgs.msg:Image
    width: 640
    height: 480
    channels: 3
    resize: false
```

- key `front_rgb` 会原样出现在 WebSocket 请求的 `images` 字典和 dataset feature `observation.images.front_rgb` 中。
- 当前 policy 图像协议使用 RGB，因此 `channels` 必须为 3。
- `resize=false` 时尺寸不一致会拒绝该帧。
- `resize=true` 时 bridge/recorder 在进入 cache 前调整到配置尺寸。
- 图像数量不固定，WebUI 会根据 dataset metadata 动态生成视频面板。

## 推理频率

```yaml
inference:
  action_horizon: 20
  model_hz: 20
  command_hz: 30
```

输出命令数按：

```text
round(action_horizon / model_hz * command_hz)
```

计算。ZME 样例因此把 20 个模型动作重采样为 30 个机器人命令。

server metadata 中的 `state_dim`、`action_dim` 和 `action_horizon` 必须与 YAML 一致，否则 bridge 拒绝连接。

## 模式控制器

模式服务通常具有机器人专属语义，因此由配置指定 factory：

```yaml
mode_controller:
  factory: my_package.mode:create_controller
  options:
    service_type: my_robot_msgs.srv:ChangeMode
    service: /robot/change_mode
```

ZME 样例使用已有 `TrajFollowMode` 和 `TrajFollowSecondaryMode` 服务。其他机器人只需实现该小型 factory；状态、动作和图像映射仍保持纯配置。
