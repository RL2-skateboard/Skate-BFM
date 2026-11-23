# skate-sim

独立的滑板仿真项目。当前活动后端为：

- MuJoCo CPU（统一网页场景，保留原生检视窗口）
- mjlab / MuJoCo Warp CUDA

Isaac Sim 已按用户决定从当前交付范围移除，不是活动依赖，也不参与阶段验收。

## 当前状态

2026-11-04：用户指出统一入口误用了平板资产，原先的资产一致性结论撤回。
现已恢复本项目五种 BoardStyle 的翘头尾甲板、桥架、轮子和原参数；仅借鉴
HUSKY 的倾斜—转向运动学原理。`play.py` 默认暂停展示 G1 29DoF 机器人
和本项目滑板；两后端共用资产开关、
障碍添加/移除、平移/旋转预览、应用场景和 JSON 摆放预设。应用时暂停并重置，
录制中禁止修改场景。`--scene` 可读取 JSON；旧的命名场景仅保留历史验证入口。

当前 62 项测试通过，新增原资产几何/质量一致性、翘头尾碰撞表面与障碍预览检查。
障碍预览使用实际形状和尺寸，新增障碍错开放置。恢复资产后的 CPU 500 步、
GPU 无控制与真实 BFM 动作各 500 步独立回放通过；GPU 部分重置前后各 500 步
也通过，`qpos/qvel/time` 最大误差均为 0。两端逐帧状态与外力精确读回通过。
新证据位于不入 Git 的 `runs/asset_restore_20261104/`。
此前 GPU 无控制、BFM 动作和分段重置回放的零误差结果对应误用的平板资产，
保留在 `runs/gpu_strict_20261104/formal/`，不能代替恢复资产后的复测。
门槛仍为 `1e-7`、每段连续 500 步，数值实现身份写入记录并核验。
短于指定连续时长或缺帧的记录不能冒充完整积分验收通过。
旧 `runs/final_acceptance_20261104/` 与更早记录的失败保持原样。

确定性有明显吞吐代价：恢复原资产后，本机 N=64 的三个 10 步短测窗口中，
耗时中位数由 0.01678 s 增至 2.66574 s（约 159 倍）；N=2 同法约 36 倍。
配对计时在无其他 GPU 步进负载时运行，详见 `runs/asset_restore_20261104/performance.json`。
这是所测配置的短程结果和正确性优先版本，
不能把其吞吐当作高效批量训练能力；没有据此修改原数值门槛或增加快慢模式接口。
适配代码位于 `src/skate_sim/backends/warp.py`，依赖固定 MuJoCo Warp 3.11.0 / Warp 1.17.0。

当前主线资产与运动学来源边界见 [资产说明](assets/README.md)。
新场景尚未接入旧物理参数随机化，会明确拒绝这些选项。04 足底接触研究仍冻结，
稳定骑行及真实 sim2real 校准未验收。下方日期更早的资产清单与阶段记录为历史结果。
当前 Warp 对圆柱与圆柱/盒体/网格的多接触检测支持有限，会提示这些配对最多
生成一个接触点；这不是显示故障或缓存溢出，也未通过改变原资产规避该限制。

本机启动（同一入口打印本地网页地址）：

```bash
cd /home/hm/workspace/skate-sim
/home/hm/.conda/envs/skate-sim-mjlab/bin/python play.py
/home/hm/.conda/envs/skate-sim-mjlab/bin/python play.py --backend mjlab
```

2026-11-02 更新：05 数值合同、原生逐关节检视和真实 CUDA 控制链路已补测；
06 以后的非接触工程检查已推进。04 接触模型、数据与方法方案按用户要求冻结，
先讨论再实施。下表保留原阶段签收状态，`AUTO_PASS` 仅指历史自动检查，
不是完整交付或本次全部回归通过。当前结果与限制见
[验收记录](docs/acceptance.md)。

| 阶段 | 状态 | 内容 |
| --- | --- | --- |
| 01 | ACCEPTED | MuJoCo/mjlab bench、暂停、步进、外力、重置、记录 |
| 02 | ACCEPTED | 五种参数化滑板、gallery、资产检查、鼠标扰动 |
| 03 | ACCEPTED | truck/reduced、coast、release、turn、drop、收敛、回放 |
| 04 | WAITING_REVIEW | 足底—滑板接触实验台 |
| 05 | ACCEPTED | 29DoF、观测数值对照、关节检视、BFM推理；用户已重新验收 |
| 06 | WAITING_REVIEW | G1+真实滑板 MuJoCo rider slice |
| 07 | INCOMPLETE | CUDA batch/控制链路及新记录严格回放通过；阶段窗口验收待完成 |
| 09 | AUTO_PASS | 参数随机化、合成 CEM 校准、留出验证 |

阶段记录：

- [阶段 01](docs/stages/01.md)
- [阶段 02](docs/stages/02.md)
- [阶段 03](docs/stages/03.md)
- [阶段 04 计划](plan/04_contact.md)

## 当前可用功能与资产（2026-11-02 实测）

- 两个物理后端：原生 MuJoCo 与 mjlab/MuJoCo Warp CUDA。六种场景
  `bench/board/gallery/contact/robot/rider` 均实际加载并短步推进；接触场景的
  “能运行”不代表物理实验已验收。
- 五种程序化滑板：street、standard、cruiser、longboard、downhill，包含可动
  甲板、两组被动桥铰和四个独立转轮，参数见 `assets/board/manifest.json`。
  原生后端的 truck/reduced 模型及 coast/lean/release/turn/drop 入口已复跑。
  这些是旧路径的历史结果；公开的 `--model` 已移除，默认统一场景使用原项目资产
  和倾斜—转向耦合。
- 一个正式 G1 29DoF 机器人（含六个手腕自由度），以及与滑板组合的
  separate、push_start、steer_start 三种初始化。它们是场景组合，不是三个
  新机器人。逐关节固定基座检视仅支持原生 robot 场景。
- gallery 包含斜坡、低平台、圆柱横杆三类障碍；普通板和 rider 默认不加载。
- BFM ONNX 演员与 reward/tracking/goal 三个上下文文件可本地读取；原生
  支持命名 reward、命名 goal 与 tracking。GPU 当前仅支持默认 reward 和
  tracking：命名 goal 启动失败，自选 reward 名称被忽略，不能声称功能对齐。
- 工程接口：暂停/物理单步/控制单步/重置、按环境/刚体/作用点施力、有限时长
  外力、子集状态恢复、独立终止与超时、GPU 批量及 partial reset、版本化记录
  和视觉/动力学回放、种子复现、参数扰动和 nominal 恢复。
- 离线实验：板回正合成 CEM 辨识、接触力拟合、三自由度可微轨迹拟合、受约束
  接触网络的既有模型均可运行或加载复评。网络没有超过解析基线，真实目标材料
  校准尚无证据，04 的后续研究方案仍冻结待讨论。

本次重新运行 30 项单元测试全部通过；完整入口/参数组合复测与资产编译结果
在 `runs/inventory_20261102/validation.json` 和 `assets.json`，包含拒绝、失败和
沿用证据的区别，不能把测试行数当作功能数。当前主要已知限制：04 固定板仍保留
自由关节、卸载未清理加载力；GPU 命名上下文不完整；GPU 机器人策略严格回放
仍有超差，GPU 无控制滑板连续 1 秒严格回放本次通过。详见
[当前验收记录](docs/acceptance.md)。

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
python -m skate_sim check --stage 07 --backend mjlab --num-envs 64 --checkpoint checkpoints/bfm-zero --scope noncontact
python -m skate_sim check --stage all --backend all --checkpoint checkpoints/bfm-zero
```

当前活动范围只有 MuJoCo 和 mjlab。阶段 07 的 CUDA 推进、非连续 partial reset、
rider 控制子矩阵已有自动证据；历史 MuJoCo/mjlab 汇总只覆盖当时已实现的检查。
当前总验收为 `INCOMPLETE`：严格 GPU 动力学回放未通过、用户窗口尚未签收，
04 接触研究保持冻结。Isaac 不在项目范围内。

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
python play.py --backend mujoco --scene board --experiment coast --record runs/03_truck_coast
python play.py --backend mujoco --scene board --experiment release --inspect --record runs/03_truck_release
python play.py --backend mujoco --scene board --experiment turn --record runs/03_truck_turn
python play.py --backend mujoco --scene board --experiment drop --record runs/03_drop
```

阶段检查：

```bash
python play.py --backend mujoco --scene board --experiment coast --duration 10 --record runs/03_coast
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

阶段 04 计划见 [plan/04_contact.md](plan/04_contact.md)，阶段记录见
[docs/stages/04.md](docs/stages/04.md)。已有可运行入口和合成离线实验，但本次编译
发现固定板仍保留自由关节，卸载也未清理加载力，不能把入口可运行当作实验有效。
真实目标材料拟合未启用。当前按用户最新决定冻结 04，先讨论接触创新方案；
05 及后续非接触工程已获单独授权推进，不代表 04 已签收。

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

Reward 使用 `move-ego-0-0`；tracking 按官方 walking 配置对 `zs_walking.pkl`
最多三帧做 gamma=0.8 的折扣平均与范数归一化；goal 使用
`goal_reaching.pkl` 中显式指定的公开名称。goal 文件没有
T-pose，T-pose 只能用 pose baseline 验证。

2026-11-02 已修正重力投影、角速度坐标及上一策略命令缩放，并补齐独立
721 维数值对照、history/reset、动作延迟和动态子步 PD 检查。三种 policy
模式均有完整 30 仿真秒、1500 次真实 ONNX 推理记录；本次选定地面案例
均未触发 pelvis 高度低于 0.35 m 的判据。记录位于
`runs/05_observation_fix/verification.json`，不是滑板技能或泛化验证。
policy 每 20 ms 更新，PD 每 2 ms 按最新状态重算。真实 CUDA 的 robot
N=1/N=64 及 rider N=64 reward 各完成 30 秒接口检查；推理使用
CPUExecutionProvider，物理由 CUDA 推进。2026-11-02 用户重新验收后明确确认
“好了，验证完成”，05 状态为 `ACCEPTED`；不代表其他阶段或 GPU 严格回放通过。

逐关节检视（明确标注固定基座夹具）：

```bash
python play.py --backend mujoco --scene robot --controller joint --inspect \
  --checkpoint checkpoints/bfm-zero --record runs/05_joint_review
```

`[` / `]` 选择关节，`-` / `+` 小幅调整目标；`P` 切自由基座 pose，
`O` 切 policy，`E` 释放夹具和外力。`Space/N/M/R` 保持暂停、物理单步、
控制单步、重置语义。窗口显示实际关节值、重力、角速度和模型标识。

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

### 阶段09离线参数训练（新增）

`src/skate_sim/physics/train_parameters.py` 支持解析接触参数拟合、受约束接触网络，以及三自由度
足底模型的多步可微轨迹拟合。没有真实数据时须显式选择合成验证：

```bash
python -m skate_sim.physics.train_parameters --data synthetic --task force_fit --output runs/sole_force
python -m skate_sim.physics.train_parameters --data synthetic --task trajectory_fit --output runs/sole_trajectory
python -m skate_sim.physics.train_parameters --data runs/sole_force/dataset.json --model contact_net \
  --init runs/sole_force/model.pt --output runs/sole_network
python play.py --contact-run runs/sole_trajectory
```

依赖使用已有 `skate-sim-mjlab` 环境，或安装 `pip install -e '.[contact]'`。
详细数据合同、参数含义、扩展接口和 TactileStep 参考见
[离线接触训练](docs/contact_training.md)。输出为离线候选，不自动替换原生接触，
也不代表 MuJoCo/mjlab 引擎已支持反向传播或真实 sim2real 已验证。

### 全项目验收

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
