# skate：G1 29DoF环境、训练接口与回放

## 共同项目上下文（每份独立可用）

这是 **2026-11-23 迁移交接快照**，供另一台电脑、另一个账号的新对话读取，不是完整聊天历史导入。来源为本轮实际 Git/测试检查、仓库文档，以及原聊天最近可读的用户消息和最终答复；原聊天仍有更早分页未读取。历史答复中的实验数字不是本次复测结果。

- 原机 BFM：`/home/hm/workspace/skate-bfm`，分支 `train`，本地提交 `ea4de54`（43 文件变更）。
- 原机仿真：`/home/hm/workspace/skate-sim`，分支 `sim`，本地提交 `1a72110`（76 文件变更）。两目录的 origin 都指向 `RL2-skateboard/Skate-BFM`，不要误认为两个独立远端仓库。
- 本轮只完成上述本地快照提交，**没有 push**；新机 clone 不保证有这些提交。先确认分支与提交可用。这里所有绝对路径都是原机路径，新机要替换并核实存在。
- 本轮定向回归：BFM 29 项、Sim 33 项，共 62 项通过；未运行完整测试集、GPU/UI、长期训练或闭环技能验收。使用原机 Python 3.11 环境 `skate-sim-mjlab`；旧 `skatebfm` Python 3.12 因缺少 `skate_sim` 曾收集失败，不能继续按旧 README 环境盲跑。
- 快照未包含 BFM `logs/`、`MUJOCO_LOG.TXT` 和 Sim `2/`；数据、权重、runs、视频、外部研究报告也不保证由 Git 携带。Sim 参考 YAML 尾空格和一个 Python 文件末尾空行保留原样，不声称全部 diff 格式检查通过。
- 目标是原生 G1 29DoF 的人—滑板交互与模型学习。所有在线训练、评估、采集、回放统一使用独立 `skate-sim`。禁止启动、安装为活动依赖或回退到 `HuskyBfmOnlineEnv → HuskyLiteEnv` 23DoF；原生 29 维动作不得加六腕置零投影。
- 历史 HUSKY 23DoF 数据只作带来源标记的离线参考；不得称为 native29 实测，不得把源 raw qpos/qvel/source_physics 直接恢复到新模型或称为同帧物理对齐。
- 先读两仓根 `AGENTS.md`。Astra=`gpt-6-astra`负责设计/分析/最终复核；Sol=`gpt-6.1-sol`负责调试验证；Terra=`gpt-6.1-terra`负责外部资料/UI；DS 按具体设计通过 `scripts/deepseek_agent.py --model deepseek-flash` 生成待审查补丁。模型不可用就报告，不得只换角色名冒充；不能假定新账号拥有同样工具、模型或配置。密钥在新机另配，不复制身份文件。
- 一个文件同一时刻一个执行者；保留用户改动。结构、数值、闭环行为、泛化四层证据分开；有限 loss、checkpoint 可加载、短暂未摔倒都不等于任务成功。
- BFM 文档遵守 `DOCUMENTATION_RULES.md`：未经新文档任务，不自动改实验文档；计划不能记为成果。Sim 先读 `plan/12_execution.md`。04 接触研究方案仍冻结，先讨论再实施；原生接触用于环境任务不等于解冻接触研究。
- `skate-bfm/docs/PROJECT_CONTEXT.md` 是 **2026-09-25 历史23DoF快照**；BFM README 也仍有旧环境/重置说明。只用它们追溯历史，不照搬为当前运行指令。以较新用户决定、根 AGENTS 和现有代码为准。
- 本文件交接上下文，不自动启动实验、上传、发消息、commit/push，也不延续旧对话一次性的额度、运行时长或停止条件。收到新任务后先核对当前代码和所需资产。

## 原对话与最新决定

原聊天：`skate`，ID `01a19a9e-aa4f-7550-adbd-8adc8d1fbc85`。读取最近8个 turn 的用户消息/最终答复，仍有早期分页。最后明确要求“全部切换为29DoF，之前23DoF环境不再使用”。这是优先约束，不保留活动旧环境兼容回退。

## 当前代码与历史验证的界线

- 当前 `SkateEnv`（Sim `src/skate_sim/env.py`）提供外部动作和训练 transition；BFM `integration/sim.py` 的 `SkateSimOnlineEnv`包装它，latent由学习器管理，native preset重置明确 `source_physics_aligned=False`、`tracking_aligned=False`。
- 现有BFM训练参数只支持 `--online-env-backend skate-sim-cpu|skate-sim-mjlab`、`--native-reset-preset steer_start|push_start`；`--aligned-online-tracking` 是要求不支持语义时的拒绝入口，不能当已实现同帧reset。
- 原聊天最终答复报告 BFM29、Sim74项测试及CPU/GPU真实更新/重载、受控严格回放零误差；报告在 `skate-sim/runs/native29_20261104/result.json`。这些是历史证据引用，本次迁移没有重跑GPU。
- 更早训练适配报告称 N64 比旧确定性实现快37.6倍、真实采样约408 transition/s，路径 `runs/training_adapter_20261104/result.json`。仅能对当时配置成立；README还留有更早159倍慢的历史数字，勿混用或称为新机性能。
- 真实策略仍会摔倒重置，短片段标记 INCOMPLETE；训练链路可跑不代表已学会稳定骑板。BFB/RFB在 `train/train.md` Experiment5仍为计划，不宣称已实现。

## 场景与资产约束

两活动后端为MuJoCo CPU和mjlab/MuJoCo Warp CUDA；Isaac Sim不在当前交付范围。唯一可视化入口 `play.py`。默认统一场景机器人+原项目滑板、暂停启动；JSON场景与两后端共用，障碍编辑先预览再应用暂停重建，录制中锁定编辑。命名旧场景本身仍可为原生29DoF验证入口，不能因“旧”字一概删除。
滑板必须保留本项目 BoardStyle 的翘头尾、桥轴、轮子及物理来源，仅借鉴HUSKY倾斜转向原理。mimic trial中的尺寸候选需独立验证，不据此悄悄替换主线默认资产。
04接触研究仍冻结，05已由用户重新验收。稳定骑行、真实材料校准和sim2real尚未验收。原聊天指出接触台释放/加载、扭转轴、接触点滑移和带符号功率、真实参数随机化等问题；新任务开始前重定位现有代码，不能假设旧行号或所有问题仍原样存在。

## 入口与操作

- Sim：`src/skate_sim/{env,viewer,record,replay}.py`、`backends/{base,mujoco,mjlab,warp}.py`、`models/{scene,board,robot}.py`；`docs/acceptance.md`、`docs/stages/05.md`、`06.md`、`07.md`。
- BFM：`src/skate_bfm/integration/{sim,source,actions,observations}.py`，`train/scripts/{train_skate_bfm,train_runner,evaluator}.py` 与采集目录。历史 `online.py`已删除；不要恢复它。
- 原机README列出的启动命令（本次未启动UI；新机先核实环境路径）：
```bash
cd /home/hm/workspace/skate-sim
/home/hm/.conda/envs/skate-sim-mjlab/bin/python play.py
/home/hm/.conda/envs/skate-sim-mjlab/bin/python play.py --backend mjlab
```

## 下一步与验收

先核对当前入口、依赖与动作/观测/reset合同，再按用户指定任务选择训练、回放或场景工作；不擅自解冻04。测试保留原生29维腕部有效、终止与超时区分、partial reset历史一致、记录版本/模型身份和受控回放。GPU连续500步/分段回放的既定门槛不能被短片段替代；跨后端接触一致性与每个后端自回放一致性必须分开。长期技能和泛化须用独立行为指标及留出条件证明。
