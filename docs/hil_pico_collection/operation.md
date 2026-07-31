# 安装与运行

## 环境

目标主机：

- ROS2 Humble。
- Python 3.10。
- `arm_interfaces` 或机器人自己的 adapter/message package。
- 配置文件声明的相机 ROS topic。
- 与 OpenPI RL Token server 网络连通。

准备运行环境：

```bash
scripts/hil_pico_collection/prepare_runtime.sh \
  --python /usr/bin/python3
```

默认外置运行目录：

```text
/mnt/workspace/ys/futuring/openpi_runtime/hil_pico_collection
```

可用 `HIL_PICO_RUNTIME` 覆盖。数据默认写入：

```text
$HIL_PICO_RUNTIME/datasets/hil_pico_v21_YYYYMMDD
```

## 单独启动

录制器和 WebUI：

```bash
scripts/hil_pico_collection/run_recorder.sh \
  --dataset-root /data/hil_pico_v21 \
  --robot-config src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml \
  --web-port 8088
```

模式切换：

```bash
scripts/hil_pico_collection/run_mode_switcher.sh \
  --robot-config src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml
```

RL Token bridge：

```bash
scripts/hil_pico_collection/run_rl_token_bridge.sh \
  --host 127.0.0.1 \
  --port 8011 \
  --robot-config src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml \
  --prompt "fold clothes"
```

整体启动：

```bash
scripts/hil_pico_collection/run_hil_stack.sh \
  --dataset-root /data/hil_pico_v21 \
  --policy-host 127.0.0.1 \
  --policy-port 8011 \
  --robot-config src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml \
  --prompt "fold clothes"
```

默认：

```text
ROS_DOMAIN_ID=30
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## WebUI

访问 `http://<robot-host>:8088`：

- 开始/结束 episode。
- 查看掉帧、缓存新鲜度和异步保存状态。
- 浏览已保存 episode。
- 根据 dataset metadata 动态播放所有配置图像视频。
- 发布通用复位请求。

WebUI 不提供机器人轨迹 replay。Reset 按钮只向 `/pico_tele/reset_request` 发布一次 `std_msgs/Empty` 并立即返回 `accepted=true`。

## 实机验收

1. 确认配置中的图像、status 和 command topic 均持续更新。
2. 人工模式下确认 `intervention=true` 且 `/auto_arm_cmd` 为实际人工执行目标。
3. 发布 `ros2 topic pub --once /change_ctrl_mode std_msgs/msg/Bool '{data: true}'`，验证自主/PICO 双向切换。
4. 在模型模式连接 RL Token server，确认 bridge 只向 `/auto_arm_cmd` 发布。
5. chunk 执行中切入人工模式，确认模型命令立即停止。
6. WebUI 发起 reset，确认机器人本体安全执行。
7. 保存真实 episode，并使用 tristate labeler 完成 `-1/0/1/2` 标注后运行 Stage 2 admission。
