# skate-sim

独立的滑板仿真项目。当前活动后端为：

- MuJoCo native viewer
- mjlab / MuJoCo Warp CUDA

Isaac Sim 已按用户决定从当前交付范围移除，不是活动依赖，也不参与阶段验收。

## 当前状态

| 阶段 | 状态 | 内容 |
| --- | --- | --- |
| 01 | ACCEPTED | MuJoCo/mjlab bench、暂停、步进、外力、重置、记录 |
| 02 | ACCEPTED | 五种参数化滑板、gallery、资产检查、鼠标扰动 |
| 03 | ACCEPTED | truck/reduced、coast、release、turn、drop、收敛、回放 |
| 04 | WAITING_REVIEW | 足底—滑板接触实验台 |
| 05 | WAITING_REVIEW | 29DoF contract、官方G1资产、BFM ONNX地面烟测 |
| 06 | WAITING_REVIEW | G1+真实滑板 MuJoCo rider slice |
| 07 | AUTO_PASS | mjlab CUDA batch、rider controller、partial reset |
| 09 | AUTO_PASS | 参数随机化、合成 CEM 校准、留出验证 |

阶段记录：

- [阶段 01](docs/stages/01.md)
- [阶段 02](docs/stages/02.md)
- [阶段 03](docs/stages/03.md)
- [阶段 04 计划](plan/04_contact.md)

## 安装

目标环境：Ubuntu 24.04、Python 3.11、NVIDIA GPU。当前实测环境为 RTX 4060 Ti、NVIDIA 575.57.08、CUDA 12.9。

```bash
cd ~/workspace/skate-sim
conda env create -f environment-mjlab.yml
conda activate skate-sim-mjlab
python -m pip install -r requirements-mjlab.txt
python -m pip check
```

主要版本：

```text
Python 3.11.16
MuJoCo 3.11.0
mjlab 1.6.0
MuJoCo Warp 3.11.0
Warp 1.17.0
PyTorch 2.7.0+cu128
```

所有命令从项目根目录运行。当前活动代码都在 `src/skate_sim/`，唯一窗口入口是根目录 `play.py`。

## 阶段 01

MuJoCo bench：

```bash
python play.py --backend mujoco --scene bench --seed 0 --record runs/01_mujoco
```

mjlab bench 使用真实 MuJoCo Warp CUDA 物理：

```bash
python play.py --backend mjlab --scene bench --seed 0 --record runs/01_mjlab
```

键盘操作：

| 按键 | 行为 |
| --- | --- |
| `Space` | 暂停/继续 |
| `N` | 一个 physics step |
| `M` | 一个 control step |
| `R` | 重置 |
| `F` | +X 方向 20 N、持续 0.2 仿真秒 |
| `E` | 释放键盘外力 |
| `Esc` | 关闭窗口 |

自动检查：

```bash
python -m skate_sim doctor --backend mujoco
python -m skate_sim doctor --backend mjlab
python -m skate_sim doctor --backend all
python -m skate_sim check --stage 01 --backend mujoco
python -m skate_sim check --stage 07 --backend mjlab --num-envs 64
python -m skate_sim check --stage all --backend all --checkpoint checkpoints/bfm-zero
```

当前活动范围只有 MuJoCo 和 mjlab。阶段 07 的 CUDA 推进、非连续 partial reset、rider pose/policy 控制链路均已通过自动检查；MuJoCo 01–06 与 mjlab 01/07 的整体检查也已通过。Isaac 相关内容不在项目范围内。

## 阶段 02：资产

五种板型：

```text
street, standard, cruiser, longboard, downhill
```

生成 XML 和 manifest：

```bash
python -m skate_sim build --asset board --style all
```

MuJoCo inspection：

```bash
python play.py --backend mujoco --scene board --style standard --inspect --record runs/02_standard
python play.py --backend mujoco --scene board --style cruiser --inspect
python play.py --backend mujoco --scene board --style longboard --inspect
python play.py --backend mujoco --scene board --style street --inspect
python play.py --backend mujoco --scene board --style downhill --inspect
python play.py --backend mujoco --scene gallery --style standard --inspect --record runs/02_gallery
```

资产检查：

```bash
python -m skate_sim build --asset board --style all
```

资产配置唯一来源是 [board.py](src/skate_sim/models/board.py)。当前模型包含圆角分段板面、前后翘起、前后 truck、四个轮子、wheel/truck/deck anchors 和独立 gallery 障碍。

MuJoCo inspection 操作：

- `Space`：暂停/继续
- `1`–`4`：暂停时检视单个轮子转动
- `V`：碰撞/透明显示
- `J`：关节轴显示
- `R`：恢复名义状态
- 鼠标扰动：按 MuJoCo viewer 的选中刚体和拖拽操作施加外力

## mjlab 白色 viewer

启动同一套 board 资产：

```bash
python play.py --backend mjlab --scene board --style standard --inspect --record runs/02_mjlab
```

终端会打印唯一的 Viser URL，例如：

```text
mjlab Viser viewer: http://localhost:45111
```

打开终端打印的地址，不要固定使用旧的 `localhost:8080`。网页面板提供：

- Pause / step / reset
- `Board asset` 下拉切换五种板型
- X/Y/Z 世界坐标力滑块
- `Release Force`
- 场景显示控制

切换 `Board asset` 时保持同一个 Python 进程、浏览器连接和 HTTP 端口，只重建当前 mjlab/Warp 仿真和 Viser 场景。

## 阶段 03：动力学

实验命令：

```bash
python play.py --backend mujoco --scene board --model truck --experiment coast --record runs/03_truck_coast
python play.py --backend mujoco --scene board --model truck --experiment release --inspect --record runs/03_truck_release
python play.py --backend mujoco --scene board --model truck --experiment turn --record runs/03_truck_turn
python play.py --backend mujoco --scene board --model reduced --experiment turn --record runs/03_reduced_turn
python play.py --backend mujoco --scene board --model truck --experiment drop --record runs/03_drop
```

阶段检查：

```bash
python play.py --backend mujoco --scene board --model truck --experiment coast --duration 10 --record runs/03_coast
```

当前实验参数位于 [dynamics.py](src/skate_sim/physics/dynamics.py)，模型说明位于 [physics.md](docs/physics.md)。窗口显示板面 roll、前后 steer、轮速、接触、机械能、外力做功和轨迹。

回放：

```bash
python play.py --replay runs/03_truck_turn --mode visual
```

回放只读取记录，不接受新的交互外力。

## 记录

`--record runs/name` 会生成：

```text
runs/name/events.jsonl
```

记录包括 backend、scene、seed、资产配置/hash、实验参数、状态、诊断、外力和事件。`runs/` 不提交到 git。

## 阶段 04

接触实验台命令：

```bash
python play.py --backend mujoco --scene contact --experiment pull --record runs/04_pull
python play.py --backend mujoco --scene contact --experiment twist --record runs/04_twist
python play.py --backend mujoco --scene contact --experiment unload --record runs/04_unload
python play.py --backend mujoco --scene contact --experiment freeboard --record runs/04_freeboard
python play.py --backend mujoco --scene contact --experiment pull --duration 5 --record runs/04_pull
```

窗口中 `A/D` 调整拉力，`T/Y` 调整扭矩，`E` 释放外力，`Space/N/M/R` 保持通用控制。

接触实验台包含：

- 可动足垫
- 法向加载
- 切向拉动
- 粘着/起滑/滑移
- 扭转
- 卸载
- 固定板/自由板对照
- 接触力、相对速度和摩擦功率记录

阶段 04 计划见 [plan/04_contact.md](plan/04_contact.md)，阶段记录见 [docs/stages/04.md](docs/stages/04.md)。当前原生 baseline 已实现，真实数据拟合尚未启用；用户窗口验收后才能进入阶段 05。

## 阶段 05：29DoF ground contract

已实现 checkpoint-independent 的 29DoF joint contract、history/reset 状态、
PD torque helper 和 ground inspection robot：

```bash
python play.py --backend mujoco --scene robot --inspect --record runs/05_robot_inspect
```

Pose baseline（不调用 BFM，使用官方 G1 资产和逐关节 PD）：

```bash
python play.py --backend mujoco --scene robot --controller pose --duration 10 --record runs/05_pose
```

真实 BFM ONNX 已下载到 gitignored 的 `checkpoints/bfm-zero/`。推理输入为
721 维 `float32`，输出为 29 维 action；每个 2 ms 物理步按当前 q/dq
重新计算逐关节 PD torque，20 ms 控制边界更新一次 action：

```bash
python play.py --backend mujoco --scene robot --controller policy \
  --policy-task reward --latent-name move-ego-0-0 \
  --checkpoint checkpoints/bfm-zero --duration 30 --record runs/05_bfm_reward

python play.py --backend mujoco --scene robot --controller policy \
  --policy-task tracking --checkpoint checkpoints/bfm-zero \
  --duration 30 --record runs/05_bfm_tracking

python play.py --backend mujoco --scene robot --controller policy \
  --policy-task goal --latent-name walk3_subject3_9044 \
  --checkpoint checkpoints/bfm-zero --duration 30 --record runs/05_bfm_goal
```

Reward 使用 `move-ego-0-0`；tracking 使用 `zs_walking.pkl` 的单个 256 维
时间帧；goal 使用 `goal_reaching.pkl` 中显式指定的公开名称。goal 文件没有
T-pose，T-pose 只能用 pose baseline 验证。

阶段 05 控制链路已通过实际记录验证：pose/reward/tracking/goal 运行均有
非零 action、非零 qfrc_applied、持续 q/dq 变化，且
`torque_injection_error=0`。policy 更新 cadence 为 20 ms，PD 在每个 2 ms
physics substep 重算。自由基座下的最终位移/跌倒仍属于当前策略与初始化
在新 ground 任务上的表现，不再被报告为接口链路错误。

详见 [docs/robot.md](docs/robot.md) 和 [docs/stages/05.md](docs/stages/05.md)。

## 阶段 06：G1 + 滑板

```bash
python play.py --backend mujoco --scene rider --controller none --preset separate --duration 5 --record runs/06_separate
python play.py --backend mujoco --scene rider --controller pose --preset push_start --duration 10 --record runs/06_push_start
python play.py --backend mujoco --scene rider --controller pose --preset steer_start --duration 10 --record runs/06_steer_start
python play.py --backend mujoco --scene rider --controller policy --policy-task reward --checkpoint checkpoints/bfm-zero --preset steer_start --duration 30 --record runs/06_policy
python play.py --replay runs/06_policy --mode visual
```

`separate`、`push_start`、`steer_start` 是唯一三种 rider 物理初始状态。
它们不生成支撑杆、支撑弹簧、固定约束或竖直辅助稳定力；启动时会输出
编译后 pelvis、脚、deck、ground 和接触对证据。
当前 BFM latent 不是滑板任务策略，摔倒会如实记录，不代表环境接口失败。
窗口验收见 [docs/stages/06.md](docs/stages/06.md)。

## 阶段 09：参数与 sim2real 准备

阶段 09 保持纯仿真。参数范围是明确标注的工程假设，不是真实测量置信区间。
每次 reset 从 nominal 模型字段重新应用参数，不在当前值上累计缩放；参数、来源和
seed 会写入记录。

```bash
python play.py --backend mujoco --scene board --experiment release \
  --params nominal --record runs/09_nominal
python play.py --backend mujoco --scene board --experiment release \
  --randomize --seed 17 --record runs/09_random
python play.py --backend mjlab --scene rider --controller pose \
  --num-envs 64 --randomize --seed 17 --view-env 7
```

合成校准使用真实 MuJoCo board rollout、独立完整 trial 和有界 CEM；结果必须同时
看 train/held-out loss，不能解释为真实 sim2real 提升：

```bash
python -m skate_sim calibrate --experiment release --data synthetic \
  --method cem --seed 17 --output runs/09_fit
python play.py --backend mujoco --scene board --experiment release \
  --params runs/09_fit/params.yaml --record runs/09_fitted
python -m skate_sim check --stage 09 --backend all --num-envs 64 \
  --checkpoint checkpoints/bfm-zero --output runs/09_check
```

有真实数据时只接受带完整 trial 划分和显式单位的 JSON；CSV 还必须有同名
`.units.json` sidecar，不能猜测单位。没有真实数据时结论限于 synthetic method
validation。阶段 09 详细结果见
[docs/stages/09.md](docs/stages/09.md)。

## 交付检查与回放

```bash
python -m skate_sim check --stage all --backend all \
  --num-envs 64 --checkpoint checkpoints/bfm-zero --output runs/final_check
python -m skate_sim report --run runs/final_check --output docs/acceptance.md
python play.py --replay runs/03_truck_turn --mode visual
python play.py --replay runs/03_truck_turn --mode dynamics --backend mujoco
```

`visual` 只显示记录状态；`dynamics` 按记录的场景配置重新运行 native MuJoCo。
Isaac Sim、真机部署、训练平台和硬件采集均不在本项目交付范围。

## 故障排查

`Xlib: extension "NV-GLX" missing` 表示当前 Xorg 没有 NVIDIA GLX 扩展，和 Warp CUDA 计算不是同一问题。物理检查仍可运行，但 native viewer 可能黑屏或回退软件渲染，需要修复系统 NVIDIA/Xorg 图形链路。

如果 Viser 页面空白：

1. 关闭旧的 `play.py` 进程；
2. 重新启动 mjlab；
3. 使用当前终端打印的 URL；
4. 不复用旧的 `localhost:8080` 页面。
