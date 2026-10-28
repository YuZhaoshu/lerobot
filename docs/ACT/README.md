# ActiveLab ACT 全量训练指南

## 从零启动：单任务

仓库：`/home/yuzhaoshu/data/ActiveLab/lerobot`

```bash
cd /home/yuzhaoshu/data/ActiveLab/lerobot
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1
uv sync --locked --extra training
uv run wandb login
# 可选：设置 W&B 用户/团队名；不设置则使用登录账号的默认 entity
# export WANDB_ENTITY=your-wandb-entity

TASK=insert_the_funnel_into_the_measuring_cylinder
MODEL_ROOT=/home/yuzhaoshu/data/model/act
uv run activelab-act prepare --task "$TASK"
CONFIG="$(find "$MODEL_ROOT/datasets/$TASK" -mindepth 2 -maxdepth 2 \
  -type f -name activelab_act.yaml -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)"
echo "训练配置：$CONFIG"
# 可选：编辑 $CONFIG 修改 run_name、steps 或 batch_size
uv run activelab-act config-check --config "$CONFIG"
uv run activelab-act preflight --task "$TASK"
uv run activelab-act train --config "$CONFIG"
```

默认把该任务的全部正常轨迹用于训练，不留验证集。已确认的时长异常轨迹继续排除；训练结束后不会计算验证分数或自动选择 checkpoint。`prepare` 会输出新生成的配置路径；旧的 39/10 划分资产不会被覆盖。

## 配置位置与可调参数

- 模板：`/home/yuzhaoshu/data/ActiveLab/lerobot/docs/ACT/activelab_act.yaml`
- 本次实际配置：`/home/yuzhaoshu/data/model/act/datasets/<task>/<fingerprint>/activelab_act.yaml`
- 训练归档：`/home/yuzhaoshu/data/model/act/runs/<task>/<run_name>/assets/activelab_act.yaml`

`prepare` 生成的 YAML 是本次运行的配置。可修改 `task`、`data_root`、`model_root`、`run_name`、`seed`、`training.steps`、`training.batch_size`、`training.log_freq`、`training.save_freq`、`training.num_workers`、`wandb.project` 和 `wandb.entity`。`validation_fraction` 必须为 `0`，`training.eval_steps` 必须为 `0`：这表示所有未被标记异常的 episode 都进入训练，训练期间不做留出集评估。修改任务或数据路径后重新运行 `prepare`。

三个合法任务名：

```text
insert_the_funnel_into_the_measuring_cylinder
insert_the_glass_rod_into_the_test_tube
insert_the_titration_tube_into_the_conical_bottle
```

数据根目录默认 `/home/yuzhaoshu/data/active_data/ActiveLab_Data`，模型与缓存根目录默认 `/home/yuzhaoshu/data/model/act`。路径也可通过 `prepare --data-root ... --model-root ...` 指定。

## 全量训练的数据范围

每个任务原有 50 条轨迹；漏斗排除 1 条、玻璃棒排除 4 条、滴定管排除 2 条已确认的时长异常，其余分别 49、46、48 条正常轨迹全部进入训练，不留验证集。manifest 会记录训练和排除 episode。状态/动作 mean/std 使用所有训练轨迹的有效帧计算。

由于没有留出集，训练日志只能反映拟合情况，不能作为泛化效果评估。若要比较泛化效果，需要另准备未参与训练的数据或做真机闭环评估。

## 固定模型与训练基准

| 配置 | 固定值 |
|---|---:|
| state/action、观测步数、action horizon | 20 / 20、1、100 |
| 输入相机 | 主动视角、左腕、右腕，RGB 640×480 |
| ResNet | ResNet18、ImageNet V1 初始化、FrozenBatchNorm |
| Transformer | hidden 512、FFN 3200、8 heads、encoder 4 层、decoder 1 层 |
| CVAE | latent 32、encoder 4 层、KL 权重 10 |
| 优化 | AdamW；模型与骨干 LR `1e-5`；weight decay `1e-4`；梯度裁剪 10 |
| 训练 | FP32、batch size 8、100,000 steps、dropout 0.1；无 scheduler/EMA/图像增强 |

状态与动作 20 维顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪、主动视角臂 6 关节。数据保持 15 Hz 原始绝对目标；100 步约 6.67 秒。ResNet18 从 ImageNet 权重初始化，其余 ACT 模块随机初始化；所有可训练参数参与全量微调。模型结构和 feature 白名单由配置校验固定。

## 多任务、恢复和评估

顺序训练三个独立任务，每个任务使用全部正常轨迹：

```bash
uv run activelab-act train-all \
  --config /home/yuzhaoshu/data/ActiveLab/lerobot/docs/ACT/activelab_act.yaml
```

恢复训练（checkpoint 路径需包含 `training_state/`）：

```bash
uv run activelab-act resume \
  --checkpoint /home/yuzhaoshu/data/model/act/runs/<task>/<run_name>/checkpoints/<step>
```

全量训练没有留出集，不能运行 `evaluate` 来做验证集 checkpoint 选择。部署时使用最后保存的 checkpoint；效果比较应另外使用未参与训练的新数据或真机闭环评估。

## 输出与 W&B

```text
/home/yuzhaoshu/data/model/act/
├── datasets/<task>/<fingerprint>/  # 配置、全量训练 manifest、归一化统计
└── runs/<task>/<run_name>/
    ├── assets/                     # 配置及数据资产副本
    ├── wandb/                      # W&B 本地日志
    └── checkpoints/<step>/          # 模型、processor、optimizer/RNG 等状态
```

W&B 使用 online 模式，默认项目 `activelab-act`，entity 可由 `WANDB_ENTITY` 指定。训练记录 loss、L1、KL、学习率、梯度范数、step 耗时和吞吐；没有验证 episode，因此不会产生验证 loss 或最佳 checkpoint 选择结果。正式训练前用 `nvidia-smi` 确认所选 GPU 有足够空闲显存；如需指定 GPU，可用 `CUDA_VISIBLE_DEVICES=<卡号> uv run activelab-act train --config "$CONFIG"`。
