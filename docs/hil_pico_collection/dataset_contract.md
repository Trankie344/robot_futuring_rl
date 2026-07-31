# 数据集与标注契约

recorder 输出 LeRobot v2.1 目录，其中 Parquet 至少包含：

```text
observation.state  float32[configured state dimension]
action             float32[configured action dimension]
intervention       bool
control_mode       int64
timestamp
frame_index
episode_index
task_index
index
```

视频 feature 根据配置动态生成。ZME 样例为：

```text
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
```

`meta/expert_frame_index.json` 只依据显式 `intervention=true` 生成连续区间。具体机器人模式数值不会隐式产生 expert frame。

ZME 样例 schema 与当前 RL Token Stage 2 admission 所要求的 `observation.state`、`action`、`intervention`、`control_mode` 类型和 16D 顺序一致。使用其他维度时，相应训练配置和 checkpoint metadata 也必须采用相同契约。

原始 episode 必须经过 tristate labeler，才可进入 RL Token Stage 2 ready-batch/cache。ready 标注规则：

- `frame_states` 只允许整数 `-1`、`0`、`1`、`2`。
- `2` 仅能出现在 episode 最后一帧，每个 episode 最多一个。
- 只有 `2` 可以产生 `progress=1`。
- 未完成 episode 的 progress 必须严格小于 1。
- ready 数据不允许 null。
- `frame_states` 长度必须与 episode 帧数完全一致。

任何单位、左右臂顺序、夹爪映射或 norm stats 不一致都会使训练数据失效。
