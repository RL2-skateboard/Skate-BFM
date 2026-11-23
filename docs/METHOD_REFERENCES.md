# skate 方法参考上下文

用户于 2026-11-02 指定以下四篇文章作为后续方法参考。后续讨论 skate-bfm / skate-sim 的世界模型、接触表征、策略训练或规划时，先读取本文件，再按需要回到相应原文。这里记录参考依据与待验证的迁移想法，不将论文成果计作本项目成果，也不据此自动确定实现路线。

阅读日期：2026-11-02。已阅读四篇 v1 的正文、方法图、实验与关键附录；没有逐篇追读其参考文献。缓存原文共 84 页，SHA256、版本和来源记录在本机缓存的 `manifest.json`。

本机原文目录：`/home/hm/.codex/visualizations/2026/11/02/01a19a9e-aa4f-7550-adbd-8adc8d1fbc85/method-references/`。目录包含四份 PDF、提取文本和核对用方法图；远端可复现来源为下面的版本化 arXiv 链接，不能依赖此机器路径在其他机器上存在。

## 1. SuperMimic

**SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**，Liao et al.，arXiv:2610.28480v1，2026-10-20，19 页。

- [原文](https://arxiv.org/pdf/2610.28480v1) · [项目页](https://supermimic.github.io/)
- 缓存：`supermimic-2610.28480v1.pdf`。
- 定位：用可微代理动力学提高全身控制的训练效率。
- 核心：循环进行仿真采集、replay 上的世界模型拟合、模型内策略优化；世界模型接受状态、动作和物理随机化上下文。默认用 8 步自回归拟合，策略通过 32 步想象 rollout 做 BPTT，critic 补足截断后的回报。策略更新时模型参数冻结，但仍通过模型对动作求导。
- 边界：不对 MuJoCo/Warp 引擎求导。论文主要研究平地，未验证地形/物体交互；大规模 motion benchmark 是训练集覆盖，非未见动作泛化。H100 上的算力结论不能直接套到本机 GPU。
- 回查：Fig.2、§3.1–3.3、§6；附录 A–D 的状态/控制、架构和训练；附录 E–F 的数据筛选与评估。

## 2. TactileStep

**TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion**，Wang et al.，arXiv:2609.28959v1，2026-09-24，18 页。

- [原文](https://arxiv.org/pdf/2609.28959v1) · [HTML](https://arxiv.org/html/2609.28959v1) · [项目页](https://tactilestep.github.io/)
- 缓存：`tactilestep-2609.28959v1.pdf`。
- 定位：让可部署足底触觉反馈参与接触质量调节。
- 核心：把引擎法向合力按几何距离/法向分给每足 60 个 taxel，经空间扩散后提取总力、接触面积比例和 CoP。actor 使用紧凑触觉历史；按 swing / pre-landing / landing / stance 划分接触阶段，配置着地冲击与支撑奖励，双 critic 分开估计稠密/稀疏回报。
- 边界：这是触觉观测映射及策略学习，不是由形变学习接触力或替换引擎求解器。可变形/颗粒表面尚未验证。复现前需核对表 9/10 的触觉延迟概率差异（0.10/0.08）。
- 回查：Fig.2–3、§3.1–3.3、§6；附录 B 的奖励及训练、C 的噪声/归一化/采样率。不要将足部接触阶段等同于本项目 push/steer 任务阶段。

## 3. 用户提供链接：Attention-based Contacts

**Humanoid Mobility via Attention-based Contacts**，He et al.，arXiv:2610.31608v1，2026-10-22，25 页。

- [原文](https://arxiv.org/pdf/2610.31608v1) · [HTML](https://arxiv.org/html/2610.31608v1)
- 缓存：`attention-contacts-2610.31608v1.pdf`。
- 定位：以接触目标连接地形感知与全身控制，减少对完整参考轨迹的依赖。
- 核心：planner 为九个身体部位预测 1.5 秒内的下一接触位置/占用标志；attention 在观测地形点上选择接触目标。三阶段为 oracle 接触 tracker、planner 与 tracker 联训、无匹配参考轨迹地形上的 RL 微调；成功 rollout 经 hindsight 标注反哺 planner。控制器为 23DoF，部署不需要运动参考输入。
- 边界：主要面向地形中心的 2.5D 接触和局部目标控制；相同接触序列不能区分所有腾空动作。它没有验证自由运动滑板的耦合动力学。
- 回查：Fig.2–3；Methods 中 tracker、planner、hindsight labeling；Discussion；补充表 S3–S6。具体占用输出不等于已经预测完整接触力或连续时间表。

## 4. BeJEPA

**BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL**，Yan et al.，arXiv:2610.23127v1，2026-10-16，22 页。用户所写“bejepa”对应此文，不是 EB-JEPA。

- [原文](https://arxiv.org/pdf/2610.23127v1) · [HTML](https://arxiv.org/html/2610.23127v1) · [项目页](https://ylyem9x.github.io/BeJEPA-project/)
- 缓存：`bejepa-2610.23127v1.pdf`。本次项目页未成功打开，理解以原文为准。
- 定位：共同表征身体姿态与世界位姿，结合短期物理预测和长期行为概括。
- 核心：Dynamic `D(z,a)` 预测下一 embedding；Successor `S(z,zg)` 学习目标条件下的折扣未来特征。actor 通过二者组合评价动作，训练不依赖长动力学想象。保留 CPR/AUX；观测历史编码器向状态表征蒸馏。Dynamic 还支持冻结模型下的在线动作残差优化。
- 边界：Successor 不是未来轨迹生成器，Dynamic 也非长期准确仿真器。附录 E.2 显示开环位姿预测随时域增长漂移。全球姿态/时间对齐实验不构成机器人—滑板交互证据。
- 回查：Fig.2、§3.2–3.4 的 Eq.7–13；附录 A 的 TD 解释，B 的动作修正，C 的训练与网络，E.2 的预测时域。

## 后续方法讨论的项目关联：以下是迁移假设

这些条目是根据 skate 当前状态提出的研究问题，尚未实现或验证。

1. **世界模型的状态应覆盖机器人、板和接触。** 仅预测机器人姿态可能漏掉板的速度、姿态、truck/wheel 状态，以及足—板相对运动。需要明确可观测量、特权量和未知物理上下文如何进入表征。
2. **触觉特征可先作为接口与诊断。** 比较法向载荷、有效面积、CoP、滑移和支撑切换，不把现有离线 contact_net 参数拟合直接称为 TactileStep 复现。CoP 在低载荷下必须标记无效；切向滑移、摩擦功和扭转仍需单独定义。
3. **接触目标需要明确对象及运动坐标系。** 足—地面、足—板应分开标注；移动 deck 上的目标需要物体标识、板相对坐标、速度或时序语义。静态地形点选择不能原样解决移动支撑接触。
4. **短期预测与长期目标可以分开研究。** 可比较显式物理 rollout、Dynamic–Successor 组合、以及二者支持的短时动作/latent 修正；先验证各自是否保留接触切换相关信息，再决定训练目标和结构。
5. **先定义评价，再讨论组合。** 继续区分完整率、任务成功、轨迹/耦合误差、接触质量、训练算力和部署延迟；在固定 Val 场景及独立 Test 上验证，防止把较短失败轨迹上的小误差当成改进。
6. **保留当前工程边界。** 旧训练使用 HUSKY23，新 sim 是 G1 29DoF；参考论文采用 23 或 29DoF 并不自动解决本项目的 action/obs/reset/PD 迁移。BFB/RFB、JEPA 及这些文献是可比较的参考，不因为列入上下文就视为已选型或已完成。

## 后续会话的恢复方式

1. 先读本文件和当前项目状态，避免只凭论文摘要或旧聊天结论设计方法。
2. 涉及公式、损失、时间尺度或控制参数时，回到上面列出的版本和章节；需要原图时打开缓存 PDF。
3. 保留“论文事实 / 本项目推断 / 已验证实现”的区分。代码可用性和后续论文修订需要在实际复现时重新核对。
