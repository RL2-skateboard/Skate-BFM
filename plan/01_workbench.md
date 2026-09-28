# 阶段01：纯仿真工程与可视化工作台

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行；本阶段先安装/创建环境，再 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：用户已清空半成品，在 `workspace/skate-sim` 空目录（除 `plan/` 计划文档外）从零开始。不要恢复旧工程或创建旧仓库的worktree。

## 目标与可见成果

在 MuJoCo、mjlab、Isaac Sim 中分别打开真实仿真窗口：地面、坐标轴、带质量与碰撞体的测试立方体。用户可暂停、单步、重置、选择并推动立方体，能看到接触与状态数值。不是截图展示器，也不是后端名称不同但都调用 MuJoCo。

本阶段目的：尽早暴露 GPU、显示服务器、Isaac 运行时、Python 版本和依赖冲突；建立以后每阶段复用的可视化与验收工具。

## 实现内容

### 先完成实际安装

1. 在目标电脑检查OS/架构、GPU型号与显存、驱动、可用磁盘、显示会话、conda与Python；Linux额外检查GLIBC。保存简短事实表，不将本助手沙盒的配置当作用户电脑配置。
2. 没有conda时，使用官方用户级安装方法安装Miniconda或兼容发行版；校验安装来源，初始化当前shell。已有conda则复用，不覆盖其他环境。
3. 先查对应发布版本的官方安装说明和依赖声明，再确定兼容版本组合。可从Python 3.11的稳定组合尝试；这是候选起点，不是无视依赖的硬编码。选定Isaac Sim、Isaac Lab、mjlab、MuJoCo/Warp、torch版本后写入版本表。优先稳定版，不为了最新功能混装beta/develop。
4. 若尚无同名环境，创建 `conda create -n skate-sim python=3.11 pip`；已有同名环境则先检查用途和已安装内容，不删除重建。若兼容性分析要求其他Python版本，在安装前解释并在同名环境方案中落实。
5. 激活 `skate-sim`，按选定版本的官方来源安装MuJoCo、mjlab及匹配依赖，再安装Isaac Sim与实际需要的Isaac Lab组件。安装选择由本阶段实测确定，不让用户自己补装。官方要求的索引、插件、资源缓存也由OpenCode配置和验证。
6. 依赖冲突先检查约束并选取共存版本，不使用 `--no-deps` 或强制覆盖来掩盖缺依赖；不擅自创建另一个conda环境。各后端在独立进程中进行导入/启动验证，Isaac按应用生命周期初始化。
7. 第三方安装自带的demo可用于内部诊断，但阶段交给用户的三个窗口必须全部由本项目根目录 `play.py` 启动。不能把官方demo可运行当成本项目wrapper验收通过。
8. 安装失败继续排查网络、索引、包版本、系统依赖和资源路径；只有确实无法自行处理的外部条件才提交具体阻塞。GPU/驱动不满足时不能用CPU空实现伪装成功。
9. 通过后整理 `environment.yml` 与必要版本约束；README写实际安装命令、实测版本、conda激活、缓存位置及复现步骤。保留简明安装结果，删除临时诊断代码；不交付大型自制安装框架。

安装依据：[Isaac官方pip安装](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)、[mjlab安装](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html)、[Conda安装](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)。执行时核对所选版本，不混用不同版本页面的命令。

### 再完成工作台与接口

1. 解析用户实际项目绝对路径，确认当前工作目录就是 `workspace/skate-sim`，阅读其中的 `plan/` 与适用指导文件；直接在此建立纯仿真工程，不克隆旧项目树、不执行旧分支清理。不要把 `workspace` 擅自解释为根目录 `/workspace`。
2. 制定依赖矩阵：操作系统、GPU/驱动、Python、MuJoCo、mjlab、MuJoCo Warp、Isaac Sim/Isaac Lab。使用彼此兼容且实测的固定版本，不能把各项目 main 任意组合。
3. 创建或复用唯一指定的 conda 环境 `skate-sim`。先求解兼容的Python/引擎/torch/NumPy版本组合，再安装。三个后端在独立进程运行，但由同一环境及同一份项目源码启动；禁止默认新建其他名称的环境。
4. 实现根目录 `play.py` 这个唯一可视化入口、内部非可视化工具、最小场景、viewer 生命周期、事件队列和记录。Isaac 使用其实际启动器启动应用后再导入依赖，退出时正常关闭。
5. 设计最小 backend 合同：reset(seed, env_ids)、step(action)、get_state、set_state、apply_wrench、diagnostics、render、close；batch shape、device、坐标系、力作用点、terminated/truncated 必须写清。
6. 使用 body/joint 名称映射实体，不假定机器人永远排在 qpos 开头。无机器人时 action 维度为0或明确禁用，不伪造29维控制。
7. 先实现观测/动作合同类型与接口，不用全零张量伪装尚未实现的机器人后端。
8. 固定记录 schema：commit、版本、配置/hash、seed、场景、时间、状态、事件、外力、环境ID。没有录像编码器时安装并验证，不静默失败。
9. 显示模拟时间、真实时间、实时倍率、后端、运行/暂停、受力对象；渲染频率和物理频率解耦。

## 要实现并实测的命令

以下以及后续阶段均是要求实现的目标命令。先完成依赖安装和项目入口，再运行；不能在空目录直接运行后把找不到文件当作完成。OpenCode须提供用户机器上验证过的具体版本。

```bash
conda activate skate-sim
python -m pip install -e .
python play.py --help
python -m skate_sim doctor --backend mujoco --display
python play.py --backend mujoco --scene bench --seed 0 --record runs/01_mujoco
python -m skate_sim doctor --backend mjlab --display
python play.py --backend mjlab --scene bench --num-envs 1 --seed 0 --record runs/01_mjlab
python -m skate_sim doctor --backend isaacsim --display
python play.py --backend isaacsim --scene bench --num-envs 1 --seed 0 --record runs/01_isaac
python -m skate_sim check --stage 01 --backend all --output runs/01_check
```

`--backend all` 由同一个 `skate-sim` 环境中的检查器依次派生独立进程并汇总，不能在同一进程提前导入全部引擎。所有窗口仍由 `python play.py` 启动，不要求用户运行第三方demo或另外的viewer脚本。

## 用户窗口验收（每后端约3分钟）

1. 打开窗口，旋转镜头看地面和碰撞体，确认角标中的真实后端与版本。
2. Space 暂停，等待2秒：模拟时间、物体位置保持不变。
3. N 单步：时间恰增一个 physics_dt；M 控制步：恰增 control_dt。
4. 选择立方体，施加例如1 N、0.2 s的水平外力；查看力箭头、位移及接触变化。这里仅验证链路，不用统一加速度证明跨后端物理等价。
5. 释放并 R 重置；位置、速度、外力缓存、显示一致回到初始值。
6. 关闭窗口再启动一次，不残留进程或 GPU 资源错误。

## 自动指标与证据

- dt 累计误差符合数值精度，暂停不变；自由落体非接触段与解析运动对照。
- 重置10次不出现非法四元数/NaN；固定 seed 的初始参数可复现。
- 外力事件有大小/坐标系/持续时间/作用点记录，释放后不残留。
- 三后端截图各1张；每后端一段包含暂停、施力、重置的视频或用户现场操作记录。
- docs/stages/01.md 写完整安装、命令、平台、结果和未通过项。

## 阶段关卡与修复

缺少MuJoCo/mjlab/Isaac Sim、conda或普通依赖时先自行安装、验证和排错，不能直接标BLOCKED交差。只有经过安装/兼容性排查，确认硬件不支持、权限/许可受限、磁盘不足或显示链路无法配置时才报告具体阻塞、已尝试步骤和需要的外部条件。离屏视频可帮助诊断，但不能替代本阶段用户窗口要求；经用户接受的远程桌面/流式窗口可用。

禁止继续堆后端占位代码。修复窗口/运行链路后复跑本阶段；用户确认三个窗口与基本交互后标 ACCEPTED，再进入02。

参考：[MuJoCo Python/viewer](https://mujoco.readthedocs.io/en/stable/python.html)、[mjlab](https://github.com/mujocolab/mjlab)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab)。
