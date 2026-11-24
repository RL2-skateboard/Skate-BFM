# 迁移与项目协作总控

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

## 本对话负责什么

用户先询问把本地项目与 Codex 对话带到另一电脑、另一账号；项目可经 GitHub 交接，真正需要保留的是各聊天上下文。随后授权将两个项目未提交的合法改动本地提交，再要求拆成对应聊天可用的 AGENTS.md。此目录是后一个请求的交付，尚未提交。

## 已完成与交接方式

- 两个提交已独立复核：tracked 工作树与 HEAD 一致，余下只是在共同部分列出的本地产物；没有替用户改写实现。
- 当前新建 `handoff/` 会成为 BFM 的新增未跟踪目录，这是本轮文档产物，并非此前提交遗漏。
- 每个子目录的 AGENTS.md 给一个对应新对话；可以上传文件或全文粘贴。不要依次覆盖目标仓库根 AGENTS.md，也不要认为保存在 handoff 子目录会自动约束整个仓库。
- `01-data-prep` 对应“data prep”；`02-mimic-trial` 对应“mimic trial”；`03-skate` 对应“skate”；`04-dataset-and-world-model` 对应“确认数据集使用限制”。最后一份兼有可微世界模型讨论。
- 原聊天导出的片段仅供写作参考，没有将原始完整聊天或工具日志复制进本目录。附件、旧 localhost 页面、视频和账号状态不会随说明迁移。

## 新机下一步

1. 在两目录分别核对 `git status --short --branch`、`git log -1 --oneline`；若上述提交未在新机，先完成用户选择的代码传输流程，不能声称已同步。
2. 盘点模型、数据、runs 与视频，尤其 mimic trial 的失败记录；确认环境 Python 3.11 与本地 skate-sim 依赖。重新配置所需凭据，不转存旧账号 token。
3. 把对应文件交给对应新聊天，要求先复述当前目标、已知失败和下一步；再按用户新的具体任务继续。

## 本轮测试命令（原机已实际执行）

BFM 原机目录中：
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:/home/hm/workspace/skate-sim/src /home/hm/.conda/envs/skate-sim-mjlab/bin/python -m pytest -p no:cacheprovider -q tests/test_integration.py tests/test_observations.py tests/test_collection.py tests/test_evaluator.py tests/test_sim_training.py
```
Sim 原机目录中：
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/hm/.conda/envs/skate-sim-mjlab/bin/python -m pytest -p no:cacheprovider -q tests/test_scene.py tests/test_replay.py tests/test_training.py tests/test_native_reset.py tests/test_board_asset_preservation.py
```
这些是针对性回归，不是完整可复现环境安装教程。
