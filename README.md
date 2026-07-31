# robot_futuring_rl

面向真实机器人闭环学习的开源全栈：VLA 微调、进展/奖励标注、RL Token 两阶段训练、PICO 遥操作，以及 HIL（Human-in-the-Loop）纠错数据采集。

robot_futuring_rl 在 [OpenPI](https://github.com/Physical-Intelligence/openpi) 的 π 系列模型基础上，补充了从机器人数据采集、三态进展标注、优势加权训练到在线策略部署的完整工程链路。项目强调模块隔离：模型、训练器、标注平台、ROS2 遥操作和机器人协议适配层都可以独立使用。

## 项目框架与闭环工作流

![robot_futuring_rl closed-loop robot learning workflow](docs/assets/robot_futuring_rl-closed-loop-workflow.png)

典型迭代过程是：先用示范数据训练 VLA 或 RL Token Stage 1，再部署策略并通过 PICO/HIL 收集人工纠错，完成三态标注后训练 Stage 2 actor，最后重新部署并进入下一轮采集。

## 项目能力

| 模块 | 作用 | 主要入口 |
|---|---|---|
| π₀ / π₀-FAST / π₀.₅ | 上游 VLA 训练、微调与推理 | `src/openpi/models/pi0`、`scripts/train/train.py` |
| ARM AW-BC | 根据未来 progress 增量对行为克隆样本加权 | `src/openpi/models/arm_awbc`、`scripts/train/train_arm.py` |
| ARM Value Model | CLIP + Temporal Transformer，预测进展区间和任务完成概率 | `src/openpi/models/arm_value`、`scripts/train/train_arm_value.py` |
| RL Token Stage 1 | 在冻结 PI0.5 上训练 prefix RL Token autoencoder | `src/openpi/models/rl_token`、`scripts/train/rl_token/train_stage1.py` |
| RL Token Stage 2 | 基于冻结特征、Replay 和 TD3+BC 训练 actor/critic | `src/openpi/training/rl_token/stage2`、`scripts/train/rl_token/train_stage2.py` |
| Tri-state Labeler | 多相机 episode 的 `-1/0/1/2` 进展/完成标注平台 | `src/tristate_labeler`、`scripts/tristate_labeler/run_labeler.sh` |
| PICO Teleoperation | 机器人无关的 PICO 位姿、按键和操作命令 ROS2 bridge | `src/pico_tele`、`scripts/pico_tele` |
| HIL PICO Collection | LeRobot v2.1 纠错采集、WebUI、RL Token 推理与动作执行 | `src/hil_pico_collection`、`scripts/hil_pico_collection` |
| Robot Protocol Config | 用 YAML 解耦关节/动作顺序、ROS 字段、相机和推理频率 | `src/hil_pico_collection/hil_pico_collection/config` |

### RL Token 论文

RL Token 两阶段训练方法请参考论文 **Focus to Learn More**：

- Cite as: [**arXiv:2604.23073**](https://arxiv.org/abs/2604.23073) **[cs.LG]**
- 当前版本：[**arXiv:2604.23073v2**](https://arxiv.org/abs/2604.23073v2) **[cs.LG]**
- DOI: [**10.48550/arXiv.2604.23073**](https://doi.org/10.48550/arXiv.2604.23073)

## 图形界面

### 三态进展标注平台

标注器同步展示一个 episode 的多路相机视频，支持区间选择、快捷键标注、任务锁、进度统计和结果导出。下图加载了 `hil_pico_v21_20260703` 中一条真实的 30 Hz 三相机 HIL 纠错轨迹。

![Tri-state annotation platform](docs/assets/tristate-labeler-ui.png)

标签契约固定为：

| `frame_states` | 语义 | Stage 2 reward |
|---:|---|---:|
| `-1` | 回退、有害进展或状态变差 | `-1` |
| `0` | 无明显进展 | `0` |
| `1` | 正向进展或完成一个子步骤 | `+1` |
| `2` | 最终任务完成 | `+2` |

数据校验遵守以下规则：

- `frame_states` 只能包含整数 `-1`、`0`、`1`、`2`，ready 数据不允许 `null`。
- `2` 每个 episode 最多一个，并且只能位于最后一帧。
- 只有 `2` 可以产生 `progress=1`；未完成 episode 的 progress 必须严格小于 `1`。
- terminal 表示 episode 边界，与任务是否成功相互独立。

### HIL PICO 采集平台

HIL WebUI 用于开始/结束 episode、发送通用复位请求、监控相机新鲜度与掉帧情况，以及浏览已保存的 LeRobot v2.1 数据。

![HIL PICO collection platform](docs/assets/hil-pico-collection-ui.png)

机器人相关的状态维度、动作维度、字段顺序、ROS topic、消息类型、相机名称和输入尺寸均来自 YAML，而不是写死在采集或推理框架中。

## 目录结构

```text
robot_futuring_rl/
├── src/
│   ├── openpi/
│   │   ├── models/
│   │   │   ├── pi0/
│   │   │   ├── arm_awbc/
│   │   │   ├── arm_value/
│   │   │   └── rl_token/
│   │   ├── policies/rl_token/
│   │   └── training/
│   │       ├── arm_awbc/
│   │       ├── arm_value/
│   │       └── rl_token/
│   ├── tristate_labeler/
│   ├── pico_tele/
│   └── hil_pico_collection/
├── scripts/
│   ├── train/
│   ├── tools/
│   ├── tristate_labeler/
│   ├── pico_tele/
│   └── hil_pico_collection/
├── docs/
├── packages/openpi-client/
├── checkpoints/                 # 本地权重，不提交 Git
└── data/                        # 本地数据，不提交 Git
```

## 环境要求

本项目包含三个相互隔离的运行环境：

| 环境 | 推荐系统 | Python | 用途 |
|---|---|---:|---|
| OpenPI/训练 | Ubuntu 22.04 + NVIDIA GPU | `>=3.11` | π 系列、ARM、RL Token 训练与推理 |
| HIL 采集 | Ubuntu 22.04 + ROS2 Humble | `3.10` | 机器人 ROS2 节点、WebUI、轻量 OpenPI client |
| PICO bridge | Ubuntu 22.04 + ROS2 Humble | C++/Python | XRoboToolkit SDK、PICO 状态和手势路由 |

模型训练显存需求取决于模型、batch size、LoRA/FSDP 和精度设置。上游 π 模型通常需要 24 GB 以上显存进行 LoRA 微调，全量微调通常需要 70 GB 以上显存。请安装与 GPU 架构匹配的 PyTorch/JAX CUDA 构建；Blackwell `sm_120` GPU 必须使用明确包含该架构支持的 PyTorch 版本。

## 安装

### OpenPI 与训练环境

克隆仓库时初始化 submodule：

```bash
git clone --recurse-submodules https://github.com/<YOUR_ORG>/robot_futuring_rl.git
cd robot_futuring_rl
```

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

如需使用容器环境，可参考 [Docker 安装说明](docs/docker.md)。

如需运行标注平台，在同一环境补充其 Web 依赖：

```bash
uv pip install -r scripts/tristate_labeler/requirements.txt
```

PyTorch π₀/π₀.₅ 需要应用仓库提供的 Transformers 兼容补丁，完整说明见上游保留的 PyTorch 文档与源码目录：

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

> [!WARNING]
> 默认 uv link mode 可能使用 hardlink。复制补丁前请确认该虚拟环境是否与其他项目共享 uv cache；如需撤销，可重新创建隔离环境或清理对应 Transformers cache。

### Checkpoint 与数据

`checkpoints/`、`data/`、`assets/`、`logs/` 和 `wandb/` 均被 Git 忽略。模型权重、机器人数据、XRoboToolkit 二进制和运行时产物不要提交到仓库。

ARM Value 默认离线读取 CLIP ViT-B/32：

```bash
mkdir -p checkpoints
uv run huggingface-cli download \
  openai/clip-vit-base-patch32 \
  --local-dir checkpoints/clip-vit-base-patch32
```

下载后可以离线验证：

```bash
uv run python - <<'PY'
from transformers import CLIPModel, CLIPProcessor

path = "checkpoints/clip-vit-base-patch32"
CLIPModel.from_pretrained(path, local_files_only=True)
CLIPProcessor.from_pretrained(path, local_files_only=True)
print("CLIP assets are ready")
PY
```

RL Token 默认资产布局为：

```text
checkpoints/rl_token/pi05_lite0030_base/29999
checkpoints/rl_token/pi05_lite0030_rltoken_only/54999
data/rl_token/lite0030_stage1
```

可从自有存储断点复制并校验：

```bash
uv run python scripts/tools/rl_token/prepare_assets.py \
  --source-base /path/to/base/29999 \
  --source-stage1 /path/to/stage1/54999 \
  --source-dataset /path/to/lerobot-v21-dataset

uv run python scripts/tools/rl_token/prepare_assets.py \
  --source-base /path/to/base/29999 \
  --source-stage1 /path/to/stage1/54999 \
  --source-dataset /path/to/lerobot-v21-dataset \
  --verify-only
```

## 快速使用

### 1. 训练或部署上游 π 系列模型

先为目标训练配置计算 normalization statistics：

```bash
uv run scripts/tools/compute_norm_stats.py --config-name pi05_libero
```

启动 JAX 训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train/train.py \
  pi05_libero \
  --exp-name pi05_experiment \
  --overwrite
```

启动标准 OpenPI policy server：

```bash
uv run scripts/tools/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/pi05_experiment/20000
```

仓库同时保留 π₀/π₀.₅ 的 PyTorch 实现和 `scripts/train/train_pytorch.py`。上游模型的数据转换、fine-tuning 和 remote inference 示例位于 `examples/` 与 [remote inference 文档](docs/remote_inference.md)。

### 2. 运行三态标注平台

数据集路径既可以作为 `--dataset` 参数，也可以作为第一个位置参数传入：

```bash
PYTHON_BIN=.venv/bin/python scripts/tristate_labeler/run_labeler.sh \
  --dataset /path/to/lerobot-v21-dataset \
  --db workspace/labeler.db \
  --host 0.0.0.0 \
  --port 8000 \
  --stride 10
```

浏览器访问 `http://<host>:8000`。`stride` 可选 `30`、`15`、`10`、`5`，用于控制标注任务的采样步长。

导出 JSONL 和 CSV：

```bash
PYTHONPATH=src uv run python -m tristate_labeler export \
  --db workspace/labeler.db \
  --out exports
```

标注结果进入训练前仍需经过 ready-batch 校验和不可变 metadata/hash 绑定，不能直接把未校验 CSV 当作 Stage 2 replay。

### 3. 训练 ARM Value Model

生产配置名为 `arm_value_hil_pico_v21`，CPU 单步调试配置名为 `arm_value_debug`：

```bash
uv run python scripts/train/train_arm_value.py \
  arm_value_hil_pico_v21 \
  --exp-name value_baseline \
  --data.repo-id /path/to/lerobot-v21-dataset \
  --data.progress-path /path/to/progress.parquet \
  --data.norm-stats-path /path/to/norm_stats.json
```

多卡使用 `torchrun`：

```bash
uv run torchrun --standalone --nproc-per-node=2 \
  scripts/train/train_arm_value.py \
  arm_value_hil_pico_v21 \
  --exp-name value_ddp
```

checkpoint 默认写入 `checkpoints/arm_value/<config>/<experiment>/`，包含 `latest.pt` 和编号 checkpoint。模型使用单相机历史窗口、标准化 state 和任务文本，输出 interval/success 双头结果。

### 4. 训练 ARM AW-BC

ARM AW-BC 复用 OpenPI VLA 和数据变换，但使用独立加权训练入口：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/train/train_arm.py \
  <arm-awbc-config> \
  --exp-name arm_awbc_experiment
```

对应数据配置需要提供 `arm_progress_path`。权重由当前帧与 action chunk 末端的 progress 增量计算，缺失或无效 progress 会按配置的 fallback 策略处理。

### 5. RL Token Stage 1

Stage 1 只训练 prefix RL Token autoencoder，冻结 PI0.5 VLA：

```bash
uv run python scripts/train/rl_token/train_stage1.py \
  rl_token_stage1 \
  --exp-name stage1_experiment
```

生产配置使用 batch 256、30k steps、10k warmup、LR `5e-5`、EMA `0.999`，每 5k steps 保存一次。调试配置为 `rl_token_stage1_debug`。详细说明见 [Stage 1 文档](docs/rl_token/stage1_training.md)。

### 6. RL Token Stage 2

Stage 2 transition 使用 30 Hz 原始帧、horizon 20、stride 2 和强制尾对齐；reward 是窗口内 20 个 `frame_states` 的和，允许为负数。

构建冻结特征 cache：

```bash
uv run python scripts/tools/rl_token/build_stage2_cache.py \
  rl_token_stage2 \
  --checkpoint checkpoints/rl_token/pi05_lite0030_rltoken_only/54999 \
  --batch /path/to/validated-ready-batch \
  --training-root /path/to/openpi_runtime/rl_token_stage2 \
  --round-id round_000001
```

发布 replay snapshot：

```bash
uv run python scripts/tools/rl_token/publish_replay.py \
  --cache-shard <cache-shard> \
  --admission <admission.json> \
  --output <replay.json>
```

训练一个 round：

```bash
uv run python scripts/train/rl_token/train_stage2.py \
  rl_token_stage2 \
  --checkpoint-dir <round-checkpoints> \
  --round-id round_000001 \
  --admission <admission.json> \
  --replay-snapshot <replay.json>
```

cache/replay 会绑定 Stage 1/2 配置、reward schema、标签哈希、checkpoint 参数哈希和 norm-stats 哈希；不一致时拒绝复用。详细流程见 [Stage 2 文档](docs/rl_token/stage2_manual_workflow.md)。

### 7. 启动 RL Token Actor

`mean` 用于确定性部署，`collection` 会加入 AR(1) exploration：

```bash
uv run python scripts/tools/rl_token/serve_actor.py \
  --base-checkpoint checkpoints/rl_token/pi05_lite0030_rltoken_only/54999 \
  --actor-checkpoint <completed-stage2-step> \
  --mode mean \
  --port 8011
```

服务使用 OpenPI WebSocket + MessagePack 协议，返回 `[action_horizon, action_dim]` 动作 chunk。更多信息见 [Actor 推理文档](docs/rl_token/inference.md)。

### 8. 构建并运行 PICO ROS2 Bridge

PICO 端和主机端安装包以 [XR-Robotics GitHub organization](https://github.com/XR-Robotics) 为准。请先阅读 [PICO 安装文档](docs/pico_tele/installation.md)，安装与主机架构匹配的 XRoboToolkit PC Service，并在 PICO 中安装官方 Unity Client APK。

使用系统安装的 SDK：

```bash
export PICO_ROBOT_SDK_ROOT=/opt/apps/roboticsservice/SDK
scripts/pico_tele/build.sh
```

也可以准备外置 SDK runtime：

```bash
scripts/pico_tele/prepare_runtime.sh \
  --sdk-zip /path/to/SDK.zip \
  --pc-service-deb /path/to/matching-pc-service.deb
scripts/pico_tele/build.sh
```

分别启动 PC Service 和 ROS2 bridge：

```bash
scripts/pico_tele/run_pc_service.sh
scripts/pico_tele/run_pico_tele.sh
```

默认发布：

```text
/pico_tele/state
/pico_tele/button_event
/pico_tele/operator_command
/pico_tele/reset_request
/change_ctrl_mode
```

PICO bridge 只发布原始跟踪/按键和机器人无关的操作命令，不发布具体机器人关节控制。ROS 接口和手势规则见 [PICO ROS2 接口文档](docs/pico_tele/ros_interfaces.md)。

### 9. 运行 HIL 采集与机器人桥接

HIL 包面向 ROS2 Humble/Python 3.10，独立安装轻量客户端：

```bash
scripts/hil_pico_collection/prepare_runtime.sh \
  --python /usr/bin/python3
```

先启动 RL Token actor server，再启动采集、模式切换和推理 bridge：

```bash
scripts/hil_pico_collection/run_hil_stack.sh \
  --dataset-root /data/hil_pico_v21 \
  --policy-host 127.0.0.1 \
  --policy-port 8011 \
  --robot-config src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml \
  --prompt "fold clothes"
```

浏览器访问 `http://<robot-host>:8088`。模型动作由 bridge 直接发布到配置中的 command topic；ZME 样例使用 `/auto_arm_cmd`，不存在 `/execute_arm_cmd` relay。

完整运行说明见 [HIL operation](docs/hil_pico_collection/operation.md)。

## 接入新机器人

推荐复制 [ZME 双臂样例配置](src/hil_pico_collection/hil_pico_collection/config/zme_dual_arm.yaml)，为新机器人创建独立 YAML。通常无需修改 recorder、WebUI 或 RL Token bridge。

配置可声明：

- state/action 维度、唯一顺序和 ROS 字段路径；
- status、command、reset、mode switch topic 和消息类型；
- 模型控制权、人工介入和 control mode 的判定条件；
- 任意数量的 RGB 相机 key、topic、消息类型、宽高和 resize 策略；
- `action_horizon`、模型频率和机器人 command 频率；
- 每个动作维度的有限值与软限位；
- 机器人专属模式服务的可插拔 factory。

机器人控制器仍然必须负责最终安全：控制源仲裁、急停、硬/软限位、状态超时、复位流程，以及人工介入期间实际动作的回显。详细契约见 [机器人配置](docs/hil_pico_collection/robot_config.md) 和 [机器人协议](docs/hil_pico_collection/robot_protocol.md)。

## 测试

运行完整 Python 测试：

```bash
uv run pytest
```

按模块运行核心训练测试：

```bash
uv run pytest \
  src/openpi/models/arm_value \
  src/openpi/training/arm_value \
  src/openpi/models/rl_token/test \
  src/openpi/training/rl_token
```

在安装了 `pytest` 的 HIL 开发环境中运行独立测试：

```bash
/usr/bin/python3 -m pytest src/hil_pico_collection
```

没有 proprietary SDK 的 CI 可以先关闭 SDK bridge，构建消息、手势和路由模块，再在外置 colcon runtime 中执行 `colcon test`：

```bash
PICO_TELE_BUILD_SDK_BRIDGE=OFF scripts/pico_tele/build.sh
```

GPU 验收应至少覆盖：真实数据单步训练、真实 Stage 1 checkpoint 特征提取、Stage 2 cache → replay → TD3+BC → checkpoint，以及 actor WebSocket 单步推理。机器人上线前还应执行 topic、模式切换、断线停发和急停测试。

## 文档索引

- [RL Token Stage 1](docs/rl_token/stage1_training.md)
- [RL Token Stage 2](docs/rl_token/stage2_manual_workflow.md)
- [RL Token Actor inference](docs/rl_token/inference.md)
- [PICO 安装](docs/pico_tele/installation.md)
- [PICO 通信协议](docs/pico_tele/protocol.md)
- [PICO ROS2 接口](docs/pico_tele/ros_interfaces.md)
- [HIL 模块概览](docs/hil_pico_collection/README.md)
- [HIL 安装与运行](docs/hil_pico_collection/operation.md)
- [机器人 YAML 配置](docs/hil_pico_collection/robot_config.md)
- [机器人本体协议](docs/hil_pico_collection/robot_protocol.md)
- [HIL 数据契约](docs/hil_pico_collection/dataset_contract.md)
- [OpenPI remote inference](docs/remote_inference.md)
- [Normalization statistics](docs/norm_stats.md)

## 当前限制

- 仓库不分发训练数据、PI/CLIP/RL Token checkpoint、XRoboToolkit SDK 或 PC Service 二进制。
- 生产配置中的数据路径和资产路径需要按部署环境覆盖或新增独立配置。
- RL Token Stage 2 actor 必须来自标记为 `round_complete=true` 的完整 round checkpoint。
- PICO 和 HIL 实机运行依赖 ROS2、机器人消息 package、相机 topic、模式服务及机器人侧安全控制器。
- 配置层软限位不是机器人安全控制器的替代品。

## 上游、许可与致谢

本项目基于 Physical Intelligence 的 [OpenPI](https://github.com/Physical-Intelligence/openpi)，并保留其模型实现、文档、许可和来源说明。PICO 通信依赖 [XR-Robotics](https://github.com/XR-Robotics) 提供的 XRoboToolkit 软件。ARM Value 中移植自 [FluxVLA](https://github.com/FluxVLA/FluxVLA) 的相关实现保留了源码内的来源和许可说明。

代码按 [Apache License 2.0](LICENSE) 发布。使用第三方模型、数据集、checkpoint、SDK 和机器人软件时，请同时遵守其各自的许可与使用条款。

## TODO List

- [x] 保留并整理 π₀、π₀-FAST、π₀.₅ 模型训练与推理能力。
- [x] 完成 ARM AW-BC progress 加权训练流程。
- [x] 完成独立 ARM Value Model 训练、checkpoint 和离线 CLIP 加载流程。
- [x] 完成 RL Token Stage 1 prefix autoencoder 训练流程。
- [x] 完成 RL Token Stage 2 cache、replay、TD3+BC 和 actor 推理流程。
- [x] 完成 `-1/0/1/2` 三态进展标注平台和 ready 数据校验规范。
- [x] 完成机器人无关的 PICO ROS2 位姿、按键和手势命令 bridge。
- [x] 完成 HIL PICO 纠错采集、LeRobot v2.1 writer 和 WebUI。
- [x] 完成基于 YAML 的机器人观测、动作、相机、ROS topic 和推理协议适配层。
- [x] 提供 ZME 双臂机器人的参考配置和 `/auto_arm_cmd` 集成示例。
- [ ] 适配松灵机器人硬件平台并提供参考配置。
- [ ] 适配睿尔曼机械臂硬件并提供参考配置。
- [ ] 扩展更多机械臂、移动操作机器人和末端执行器适配器。
- [ ] 适配 NVIDIA Isaac Sim 仿真器，打通仿真采集、训练、评估和策略部署流程。
