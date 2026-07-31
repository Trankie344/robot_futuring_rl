# HIL PICO 纠错采集

该模块是一个独立的 ROS2 Humble/Python 3.10 应用，负责三项工作：

1. 按配置频率记录机器人状态、实际执行动作、RGB 图像、`intervention` 和 `control_mode`。
2. 通过 WebUI 控制 episode 录制、浏览 LeRobot v2.1 数据和动态回放所有配置图像。
3. 使用轻量 `openpi-client` 连接 RL Token Stage 2 WebSocket 服务，并直接向 `/auto_arm_cmd` 发布模型动作。

源码位于 `src/hil_pico_collection`，不会导入 `src/openpi` 的训练依赖。机器人状态/动作维度、顺序、ROS 字段、topic 和图像观测通过 YAML 配置，模式服务保留可插拔 factory。

核心数据流：

```text
配置的相机 + robot status + executed action
                  |
                  v
        30 Hz LeRobot v2.1 recorder
                  |
                  v
          tristate labeler (-1/0/1/2)
                  |
                  v
        RL Token Stage 2 ready/cache

RL Token WebSocket -> [configured horizon, action dim]
                  -> configured-rate resampling
                  -> configured robot command topic
```

机器人接入前必须阅读 [robot_config.md](robot_config.md) 和 [robot_protocol.md](robot_protocol.md)。运行方法见 [operation.md](operation.md)。
