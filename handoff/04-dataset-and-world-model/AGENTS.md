# 确认数据集使用限制：用途说明与可微世界模型

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

## 原对话范围

原聊天：`确认数据集使用限制`，ID `01a1a9c5-4fb7-7100-91ca-9384b4296d9e`。本交接只读取最近2个 turn 的可见用户消息和答复；较早内容及原截图条款未完整获取，**不能据此确定数据集完整许可或账号批准状态**。

用户希望复现开源项目并训练自己的模型，曾询问BONES-SEED用途填写；随后问可微世界模型、SuperMimic是否属于此类及相关工作。

## 数据用途交接

旧答复仅在“确实为非商业学术研究且公开成果”前提下建议以下文字：

> I plan to use BONES-SEED to reproduce open-source research projects and to train and evaluate my own models for human motion modeling and control. The dataset will be used for non-commercial academic research, and the research findings will be made publicly available.

这是一份条件式填写示例，不是用户已确认的实际用途，也不是许可结论。新会话若继续申请/使用/再分发问题，先取得当前官方条款和实际用途，区分数据、代码、模型权重各自的许可；不要编造商业授权、再分发权或批准状态。本轮没有提交申请、接受条款或上传数据。

## 可微世界模型的已讨论概念

- 神经动力学 `s_next = f_theta(s, a)` 可计算动作梯度；模型自身可训练，不等于策略一定通过动力学反传。
- 旧答复依据SuperMimic论文描述：MuJoCo收集转移拟合世界模型；冻结模型参数，但保留对动作的梯度，通过短时想象轨迹回报优化策略，critic补足截断未来。讨论过8步动力学拟合、32步策略展开；这些是论文设定，不代表本地项目当前实现配置。
- 区分学习的可微代理模型与可微物理模拟器（如DiffMimic），以及仅用模型生成样本再训练策略的方式；无需把MuJoCo采集过程本身变成可微。
- 相关候选：SuperTrack、ControlVAE、SuperDyno、Dreamer、DiffMimic。本轮未重新核验论文或最新代码；后续精确技术比较请查原文。项目已有索引 `skate-bfm/docs/METHOD_REFERENCES.md`，SuperMimic来源记录为 `https://arxiv.org/abs/2610.28480`。

## 下一步与禁止事项

如讨论方法，先明确要学人体状态还是人+板+接触状态、训练数据来自哪个环境，再检查多步预测、动作梯度与模型漏洞；模型内回报上升必须回到原生G1 29DoF环境检验闭环。不能把可微、单步准确或有限loss写成可靠接触控制，更不能写成真实sim2real成功。
如讨论许可，先补齐具体版本/原文条款与真实用途；不能从这份交接推断截图内容或个人身份、机构、商业属性。本文件不携带申请凭据。
