# RL Token 推理桥接

bridge 只使用 OpenPI 官方 WebSocket + MessagePack NumPy 协议。

请求结构由机器人配置决定：

```python
{
    "images": {
        "<configured image name>": uint8[configured H, configured W, 3],
    },
    "state": float32[configured state_dim],
    "prompt": str,
}
```

响应：

```python
{
    "actions": float[configured horizon, configured action_dim],
    "policy_timing": ...,
    "server_timing": ...,
}
```

首次连接必须通过 metadata 校验：

- 存在 `rlt_stage2`。
- `round_complete == true`。
- `network_config.state_dim` 与配置一致。
- `network_config.action_dim` 与配置一致。
- `network_config.action_horizon` 与配置一致。

bridge 不实现旧 ZMQ 字段、request ID、domain ID、action layout、state indices 或 structured action chunk。

bridge 根据配置的 horizon、model Hz 和 command Hz 计算输出命令数量，并使用线性插值保持 chunk 首尾动作。ZME 样例把 20 个 20 Hz 动作转换为 30 个 30 Hz 命令。每条命令发送前重新读取配置的 status topic：

- 状态超过默认 0.25 秒；
- `model_control_enabled=false`；
- `intervention=true`；
- adapter 拒绝动作；
- WebSocket 断开或响应形状/有限值不符合配置；

任一条件成立时立即停止发布并丢弃当前 chunk。恢复模型控制后，bridge 必须重新读取最新配置图像和状态并发起全新请求，不续播旧 chunk。

启动 server：

```bash
python scripts/tools/rl_token/serve_actor.py \
  --base-checkpoint checkpoints/rl_token/pi05_lite0030_rltoken_only/54999 \
  --actor-checkpoint <stage2-step> \
  --mode mean \
  --port 8011
```

启动 bridge：

```bash
scripts/hil_pico_collection/run_rl_token_bridge.sh \
  --host 127.0.0.1 \
  --port 8011 \
  --robot-config src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml \
  --prompt "fold clothes"
```
