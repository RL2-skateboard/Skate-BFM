# mimic trial：静态踩尾、接板、持板与放板

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

## 原对话与目标

原聊天：`mimic trial`，ID `01a1a412-f6ae-75a0-a190-560a655e62a9`。读取最近8个 turn 的用户消息/最终答复，未覆盖全部早期探索。用户选择“脚踩板尾 → 板弹起 → 手接板”，并要求继续完成放板、踩尾夹板实验。静态启动便于衔接和稳定；接板位置应参考真实姿势夹板末端。

## 用户决定

- 先在 **右腕**安装现成夹爪，最终使用 MuJoCo Menagerie 的 **Robotiq 2F-85**；身体29维与夹爪控制分开，不能把新增夹爪当身体变成旧23DoF。
- 不限运动学方法，可研究 reward+PPO 或其他方法；旧对话允许利用GPU，但本文件不自动开启新实验或长训练。
- 对这个试验，用户允许在保留滑板基本属性的前提下调整尺寸，注意板长/宽/轴距/翘头比例；只有拿板、踩尾接板、双脚站位和滑行都通过后才替换默认。这个局部尺寸探索许可不授权全局替换成HUSKY板或改变任意质量/摩擦参数。

## 最后可见结果：仍未成功

最后答复报告72cm短板候选使板鼻达到 **68.4°**，机器人仍站稳，随后板面撞到左膝触发保护。**夹爪仍张开，接板、悬空持板、放板均未完成**。短板仍是候选，原尺寸仍为默认。
候选报告尺寸72×20cm、轮轴中心距46cm，质量和接触参数保持原样；这些是历史试验描述，继续前需从保存配置核对。
此前报告 Robotiq 开口约85.2mm、37项回归通过及一次回放误差零；它们不证明踩尾接板成功，也不属于2026-11-23的62项复测统计。
旧任务“到剩余额度50%停止”已结束，**不能把该额度条件或运行预算带到新聊天继续执行**。

## 文件和必须带走的本地产物

- Sim：`assets/robot/grippers/robotiq_2f85/`（含 LICENSE/provenance，已提交且本轮hash匹配）；`src/skate_sim/models/gripper.py`；`validation/tail_catch.py`、`place.py`、`toe_stages.py`、`gpu_robotiq.py`、`balance.py`、`clearance.py`；`tests/test_gripper.py`、`test_gpu_robotiq.py`。
- BFM：`scripts/skate_hover_press.py`、`skate_skill_ppo.py`、`skate_skill_search.py`。先读各脚本接口和保存配置，不编造“一键成功”命令。
- 最后一版原机目录：`/home/hm/workspace/skate-bfm/logs/2026-11-09_18-17-29-compact-contact24-singlefactor/`，含 `report.json`、`events.jsonl`、`control_override.py`、`overview.mp4`、`foot-board-close.mp4`，**未提交**，须另行带走才可重放原例。
- 夹爪开合原机视频：`/home/hm/workspace/skate-sim/runs/tail_catch/robotiq_opening/opening-closeup.mp4`，亦非 Git 保证携带。

## 最后一版重跑参数（本轮只读核查）

`report.json` 的 `single_factor_probe` 和 `control_override.py` 保存了精确差异：把 `skate_hover_press.py` 中唯一的 `pulse_local-.012*` 改成内存中的 `pulse_local-.024*`，接触后参考位移从12mm变24mm；调用参数是 `--pregrasp --double-support --style standard_compact_072 --pulse`。包装器校验基准脚本 SHA256 `dad69853263734ac720b7fd351378511f9ea5b5bbdd31d029f47966e81c04b81`，不要绕过校验伪称同配置重跑。
报告保存的原机命令为：
```bash
/home/hm/.conda/envs/skate-sim/bin/python /home/hm/workspace/skate-bfm/logs/2026-11-09_18-17-29-compact-contact24-singlefactor/control_override.py
```
它是历史复现记录，**本轮未执行**；环境名与本轮测试所用 `skate-sim-mjlab` 不同，新机需检查依赖和路径。包装器会新建输出目录；`scripts/skate_hover_press.py:22` 还读取 `logs/2026-11-09_17-00-09-edge-retract-pulse/events.jsonl`，此依赖也要迁移。原始 stop_reason 为 `unexpected side/underside/non-toe contact`，左膝碰撞来自历史可见答复，若继续诊断须结合事件/视频定位具体接触对。

## 新会话下一步

先读取最后失败的配置、报告及视频，确认模型/夹爪/板尺寸，再按新任务复现撞膝保护触发。分离板弹起轨迹、手可达区、闭爪时序和稳定站立约束，先完成可保持接板，再串接放板。正式验收必须撤除虚拟固定、外部扶持或预置持板假象，检查实际接触、夹持持续时间、无碰撞保护失败及完整滑行兼容。几何可达、37/62项回归或零误差回放都不能替代这个行为验收。
公开项目 VAIC/GRAIL/HDMI 等在旧对话仅为候选参考，未确认有现成可运行的完整踩尾接板专家；不得假定下载权重即可解决。
