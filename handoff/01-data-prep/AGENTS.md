# data prep：人板共同参考、HOI retarget 与 simdata

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

## 原对话与当前目标

原聊天：`data prep`，ID `01a1bec3-9391-7302-9e41-d7ca1a88853d`。交接读取最近 8 个 turn 的用户消息/最终答复，仍有更早历史未读。用户主要要获得 BFM 和 SuperMimic 都能利用的 simdata，最近明确要求调研带 HOI 的 retarget，**足部和手部都要覆盖**。

## 已形成的分析与约束

- 建议管线是“历史 HUSKY 离线记录 → 统一人板运动参考 → 分别导出 BFM/SuperMimic 参考”；两个学习器的在线物理建议接到 native skate-sim。这是设计路线，不代表 SuperMimic 滑板适配已经实现。
- 要区分运动参考、源环境动作/动力学转移、目标环境真实执行转移。世界模型应使用目标环境实际采集的转移；历史23维 action不能直接当目标29维动作教师。
- 原对话报告的重点问题：源 `board_root_*` 是浮动底架，而目标同名字段可能是倾斜甲板；旧新 `board_heading_delta` 符号相反；历史 action[t] 表示产生 state[t] 的前次动作，目标采集记录执行前状态。重用前必须逐项核实、修正，并禁止跨 reset 拼接。
- 明确记录基架、甲板、足底锚点各自的坐标系、单位、四元数顺序、时间戳、接触对象/模式/置信度。缺少六腕记录是“不支持/推断”，不是实测零值。距离或时钟 phase 不能当接触受力真值。
- 即使动作都是29维，默认姿态、目标角缩放、clip、PD、物理子步和历史排列也可能不同；不得直接做数组拷贝或 action MSE。Base 数据没有板标签，不可补零冒充交互真值。
- 原对话检查过独立 SuperMimic 的平地单浮动根假设、71维机器人状态及不同历史编码；这些属于当时报告，新机需重新核对 `/home/hm/workspace/vibed_mujoco_supermimic`，该第三个项目不在本轮两仓快照内。
- 源 episode 级划分再派生 Phase/Continuous，防止窗口泄漏；已有命令重叠不能支持“未见命令泛化”。前进应与同初速不蹬地滑行比较，并验收完整蹬地—回板—滑行周期。

## 方法研究交接

原对话最近整理了23篇 HOI retarget 工作、17篇蒸馏/模仿工作；本交接未重新联网核验整份文献。
可继续核查的候选包括 HOI-Retarget（物体系手脚锚点）、OTRetarget（联合调整物体轨迹）、OmniRetarget（人/物/地形关系）、Weave/DexWeave/InterMimicGen（手部适配）、DynaRetarget/SBTO 与 SPIDER（动力学修正）。不能把踩固定物、踢物或手部操作的成功推广为足板移动承重已经解决。
Dyna/SPIDER需要可信、同步的人/机器人与物体参考及模型；不替代视频前端恢复。GRAIL的视频生成世界模型与SuperMimic状态动力学世界模型用途不同。

## 文件与证据入口

- BFM：`src/skate_bfm/integration/source.py`、`actions.py`、`sim.py`；`train/scripts/data_collection/{rollout_split,convert_phase,convert_continuous}.py`；`train/scripts/evaluator.py`；`train/evaluation_protocol.json` 与 `.legacy23.json`。
- Sim：`src/skate_sim/env.py`、`record.py`、`robot/contract.py`；共同方法索引 `docs/METHOD_REFERENCES.md`（两仓有版本，先核对）。
- 原机报告目录：`/home/hm/.codex/visualizations/2026/11/09/01a1bec3-9391-7302-9e41-d7ca1a88853d/`，包含 `simdata-cross-environment-analysis.html`、`policy-distillation-sim-data.html`、`HOI-retargeting-frontier.html`；它们不在本轮提交中。

## 下一步与验收

先复核1–3条完整推进周期，加失败例及左右转例；冻结共同参考 schema，校验时间、甲板坐标、偏航符号与关节支持范围，再做目标G1几何可达性检查。随后分别适配学习器并采少量 native 转移；完整闭环通过再批量转换。不得因旧建议“从HUSKY获取”重新启动已退役23DoF环境。本轮没有转换数据、训练世界模型或证明该路线闭环有效。
