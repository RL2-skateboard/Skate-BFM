# 参考资料、资产与旧实现风险清单

本文件供所有阶段共用。参考资料帮助执行，不取代固定版本源码核查与实际验证。
官方项目主页及核心接触论文页面于2026-09-26复核；以下历史commit来自前轮源码检查，执行时必须重新解析并固定本次使用版本，不能默认latest与旧checkpoint兼容。

## 1. 主项目与资产

| 资料 | 链接 | 使用边界 |
| --- | --- | --- |
| BFM-Zero | https://github.com/LeCAR-Lab/BFM-Zero | 模型、控制、观测和对应资产的权威实现 |
| BFM权重 | https://huggingface.co/LeCAR-Lab/BFM-Zero | 仅测试所需权重；核对配置/hash/许可 |
| BFM参考机器人配置 | https://github.com/LeCAR-Lab/BFM-Zero/tree/318cf44a3262e5bdec5944f82f1a5f509b95d09b/humanoidverse/config/robot/g1 | 区分不同waist/effort版本，不能混用 |
| BFM参考资产 | https://github.com/LeCAR-Lab/BFM-Zero/tree/318cf44a3262e5bdec5944f82f1a5f509b95d09b/humanoidverse/data/robots/g1 | 检查MJCF/USD/mesh与LFS完整性 |
| BFM控制/观测实现 | https://github.com/LeCAR-Lab/BFM-Zero/blob/318cf44a3262e5bdec5944f82f1a5f509b95d09b/humanoidverse/envs/legged_base_task/legged_robot_base.py | 控制子步、坐标变换、history拼接；配套history_handler一起查 |
| BFM wrapper | https://github.com/LeCAR-Lab/BFM-Zero/blob/318cf44a3262e5bdec5944f82f1a5f509b95d09b/humanoidverse/agents/envs/humanoidverse_isaac.py | 模型接口与reset语义 |
| HUSKY作者代码 | https://github.com/TeleHuman/humanoid_skateboarding | 借鉴滑板物理、实验与viewer，不继承23DoF控制接口 |
| HUSKY参考板模型 | https://github.com/TeleHuman/humanoid_skateboarding/blob/d93833e80deff7f927c0b80ef9c435d8b5c488fe/src/mjlab_husky/asset_zoo/robots/skateboard/xmls/skateboard.xml | 分析真实拓扑、tilt/truck关系与碰撞 |
| Unitree MuJoCo | https://github.com/unitreerobotics/unitree_mujoco | 几何/命名对照，不自动替换BFM资产 |
| MuJoCo Menagerie | https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1 | 对照资产，注意机器人版本和许可 |

自建内容：三样式参数化滑板、载荷/倾斜夹具、可动足底实验台、有限障碍、坐标/轮轴/接触诊断标记。碰撞代理与动态参数一起生成，不能仅生成漂亮mesh。

## 2. 引擎与窗口

安装由OpenCode在阶段01实际完成，不以“未安装”替代执行。优先选择支持同一conda环境的稳定版本，版本/Python约束以对应发布版官方文档为准，不混用main/develop或不同发行版的安装命令。

- Isaac Sim/Isaac Lab pip安装：https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- mjlab安装：https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html
- mjlab依赖声明：https://github.com/mujocolab/mjlab/blob/main/pyproject.toml
- Conda安装：https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

上述链接是安装依据，不是已经在用户机器验证成功的版本组合。

- MuJoCo Python / viewer：https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo Modeling：https://mujoco.readthedocs.io/en/stable/modeling.html
- MuJoCo XML Reference：https://mujoco.readthedocs.io/en/stable/XMLreference.html
- mjlab：https://github.com/mujocolab/mjlab
- mjlab架构：https://mujocolab.github.io/mjlab/main/source/architecture_overview.html
- MuJoCo Warp：https://mujoco.readthedocs.io/en/latest/mjwarp/
- Isaac Lab：https://github.com/isaac-sim/IsaacLab
- Isaac Sim物理：https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html
- Isaac Lab执行器：https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html

mjlab是基于MuJoCo Warp的独立运行/批量环境框架，不是Isaac Sim的另一个窗口皮肤。可微能力也不能仅从“使用GPU/Warp”推断；需核对实际算子、状态路径与梯度检查。

## 3. 方法论文与代码

| 文献/项目 | 链接 | 本计划中的用途 |
| --- | --- | --- |
| HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control | https://arxiv.org/abs/2602.03205 | 倾斜/转向耦合、被动物理与实验对照 |
| BFM-Zero | https://arxiv.org/abs/2511.04131 | 基座背景；精确输入输出以权重与代码为准 |
| ContactNets: Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations，Pfrommer等，CoRL/PMLR | https://proceedings.mlr.press/v155/pfrommer21a.html | 接触结构学习参考，不当通用即插即用摩擦网络 |
| ContactNets代码 | https://github.com/DAIRLab/contact-nets | 研究性参考；不整包引入本项目 |
| Contact Models in Robotics: a Comparative Analysis | https://arxiv.org/abs/2304.06372 | 摩擦/接触模型与求解器比较 |
| ContactBench | https://github.com/Simple-Robotics/ContactBench | 微基准和指标设计 |
| NeuralSim | https://arxiv.org/abs/2011.04217 | 物理与神经项结合的背景 |
| Differentiable Physics Simulations with Contacts: Do They Have Correct Gradients w.r.t. Position, Velocity and Control?，Zhong等，2022 | https://arxiv.org/abs/2207.05060 | 可微接触梯度验证；ICML AI4Science workshop工作 |
| SPI-Active | https://arxiv.org/abs/2505.14266 | 辨识与激励设计参考，不要求引入完整系统 |
| DROPO | https://arxiv.org/abs/2201.08434 | 离线数据驱动随机化分布，不代替真实数据 |

研究选型结论：原生接触＋正确几何/时序＋微基准＋可解释参数/DR是交付主线；CEM是小型辨识基线；可微与接触网络是有证据再进入的改进路线。不能把方法名堆齐当作降低真实gap的结果。

## 4. 旧实现缺陷应转化为回归测试

以下是前轮审查归纳出的回归风险，作为新项目测试设计输入。新项目不依赖已清空的半成品，不要求恢复旧仓库或旧分支。

| 已发现风险 | 新阶段防线 |
| --- | --- |
| actor重力/角速度常量 | 05非零状态与oracle |
| history时序/字段布局错误 | 05逐元素黄金样本 |
| privileged heading/head偏移错误 | 05旋转状态样本 |
| 每控制步只算一次PD | 05物理子步tau验证 |
| 轮子90被按弧度解释 | 02实际轴向检查 |
| 样式名称变化但尺寸不变 | 02编译量测＋窗口对照 |
| tilt是空载荷路径 | 02拓扑＋03动力学 |
| foot场景实际没有脚 | 04自由度/模型存在＋起滑实验 |
| 摩擦配置无效/诊断无力与滑移 | 04有效接触参数与力/速度显示 |
| mjlab写错root、history全零 | 07实体映射/partial reset/oracle |
| Isaac永远抛不可用、USD仅metadata | 01运行探针＋08实际加载/step |
| DR每次reset累计缩放 | 09 nominal恢复与重复seed |
| 校准train/validation复用、无条件PASS | 09独立留出＋forward重放 |
| UI按键不执行、replay没有读取 | 各阶段窗口人工验收＋06回放 |
| 检查只测shape/属性/finite | 10行为和物理指标全量回归 |

## 5. 每阶段验收记录统一模板

建议集中在docs/stages/NN.md，不为每轮新建一个报告副本：
- 阶段、状态、commit、日期、用户确认记录。
- 平台、GPU/驱动、Python/引擎版本、checkpoint/hash。
- 已实测启动命令与工作目录/环境激活；预期输出路径。
- 用户操作步骤、预期可见现象、明确异常信号。
- 自动结果（指标、阈值、值、退出码），截图/视频/运行记录位置。
- 发现问题、复现步骤、修复记录、回归结果。
- 未通过项与阻塞；下一阶段是否允许开始。

命令示例不是完成证据；实际窗口可运行、行为可解释、用户确认才构成阶段交付。
