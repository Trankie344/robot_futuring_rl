# 机器人本体接入协议

## 责任边界

HIL 应用只负责观测同步、模型推理、数据记录和通用模式/复位请求。最终控制仲裁、急停、碰撞保护、关节与夹爪硬限位必须由机器人本体实现。

默认配置为：

```text
src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml
```

状态/动作字段和图像观测优先通过 `--robot-config` 配置。只有机器人模式服务或无法声明式表达的行为才需要 `--robot-adapter module:factory` 扩展。

## ZME 样例规范向量

当前 ZME RL Token checkpoint 使用：

```text
[right_joint_0..6, left_gripper, left_joint_0..6, right_gripper]
```

其中：

- index 7 是物理左夹爪，默认来自 `gripper[0]` 或 `gripper_command[0]`。
- index 15 是物理右夹爪，默认来自 `gripper[1]` 或 `gripper_command[1]`。
- 其他机器人可以通过 YAML 改变维度和顺序，但必须与对应 checkpoint metadata 和 norm stats 完全一致。
- RL Token 输出是 output transforms 之后的绝对动作，机器人侧不得再次叠加当前状态。

## ROS 接口

| 接口 | 默认类型 | 机器人侧责任 |
|---|---|---|
| `/arm_status` | `arm_interfaces/msg/ArmStatus` | 至少 30 Hz 发布；包含双臂 7 关节、双夹爪和控制模式。 |
| `/auto_arm_cmd` | `arm_interfaces/msg/AutonomyArmCommand` | 模型模式下接收 bridge 命令；人工模式下发布或回显实际执行目标，供 recorder 记录。 |
| `/change_ctrl_mode` | `std_msgs/msg/Bool` | HIL switcher 消费 `true` 事件，并通过 adapter 请求自主/PICO 模式切换。 |
| `/pico_tele/reset_request` | `std_msgs/msg/Empty` | 安全执行机器人复位。WebUI 和 PICO bridge 只发布请求，不下发复位轨迹。 |
| `/switch_arm_executer_mode` | `arm_interfaces/srv/TrajFollowMode` | 默认 adapter 用于切入 PICO 人工模式。 |
| `/switch_arm_secondary_mode` | `arm_interfaces/srv/TrajFollowSecondaryMode` | 默认 adapter 用于切回自主模型模式。 |

默认相机：

```text
/camera_dcw2/custom_cam_color_1M
/camera_dcl_left/custom_cam_color_1M
/camera_dcl_right/custom_cam_color_1M
```

ZME 配置分别映射为 `top`、`left_wrist`、`right_wrist`。其他机器人可在 YAML 中修改图像名称、topic、消息类型和输入尺寸。

## 默认 arm_interfaces 字段

`ArmStatus`：

- `left_arm[0:7]`、`right_arm[0:7]`，元素可以是标量或带 `joint_status[0]` 的对象。
- `gripper[0:2]`。
- `other_status[0]` 为 primary mode，`other_status[1]` 为 secondary mode。

`AutonomyArmCommand`：

- 兼容 `left_command/right_command`。
- 兼容 `left_joint_command/right_joint_command`。
- `gripper_command[0:2]`。

默认模式语义：

- `primary=1, secondary=1`：`model_control_enabled=true`。
- `primary=5`：`intervention=true`，表示 PICO 人工实际执行帧。

## 必须实现的仲裁规则

机器人控制器必须满足：

- 模型控制时只接受当前有效 RL Token bridge 发布的命令。
- PICO 人工介入、复位、急停、故障或模式切换期间立即拒绝残留模型命令。
- 禁止多个发布源绕过仲裁同时产生有效动作。
- 人工介入期间，仍在 `/auto_arm_cmd` 发布或回显实际执行的关节/夹爪目标。
- `intervention=true` 必须只覆盖人工实际执行的帧，并与 `/auto_arm_cmd` 精确对齐。
- 人工、复位、急停和故障期间必须报告 `model_control_enabled=false`。
- 对所有模型与人工动作实施最终关节、速度、加速度、夹爪和工作空间安全限制。
- 丢弃过期命令；建议结合 header 时间戳、当前模式和唯一控制源身份判断。

项目不会创建 `/execute_arm_cmd`，也没有 execute-to-auto relay。

## 自定义 adapter

factory 返回的对象必须提供：

```python
RobotStateSample:
    state: float32[configured_state_dim]
    control_mode: int
    intervention: bool
    model_control_enabled: bool

RobotAdapter:
    status_message_type
    command_message_type
    parse_status(message) -> RobotStateSample
    parse_executed_action(message) -> float32[configured_action_dim]
    build_command(action, stamp, frame_id) -> ROS message
    create_mode_controller(node) -> ModeController
```

`build_command` 应执行机器人特定的软限位检查；本体控制器仍负责最终硬安全。
