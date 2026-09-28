# 阶段04：足底—滑板接触、数据与学习实验

版本：2026-09-28 / v4.1，替换原 `plan/04_contact.md`。
项目与conda环境均为 `skate-sim`；开发位置为用户的 `workspace/skate-sim`。
前置：阶段03已 ACCEPTED。先读 [计划总览](00_overview.md) 与 [公共执行规范](12_execution.md)。所有窗口、交互与可视化回放仍从根目录 `play.py` 启动。

## 1. 本阶段新增要求及边界

完成原生接触实验台、公开数据审计、可微辨识与受约束接触网络实验。不要把可微/网络只写成阶段09的TODO。

按12的架构，新增或完善唯一接触实验主文件：
**`src/skate_sim/physics/contact.py`**。

小型接触网络拟合属于本次明确要求的仿真实验，是对12中最小研究范围的补充；不扩展为机器人策略训练、ROS部署或真机采集平台。OpenCode负责实现、下载检查数据、测试、展示和持续修复。本文件中的命令是待实现接口，不是声称已有代码。

四种结果必须分开：
- 原生物理自洽：不依赖真实数据，但不能证明逼真度。
- 合成数据方法验证：验证梯度、参数恢复、学习流程。
- 公开异材质真实数据实验：可以验证方法，不能等同目标脚板校准。
- 目标足底＋目标滑板材料验证：才支持目标材料范围内的sim2real精度结论。

本次检索未核实到直接覆盖“选定G1足底＋指定滑板砂纸＋目标载荷和速度”的现成公开数据集；不代表此类数据绝不存在。下面的可用来源各有用途，不能因都写着rubber或friction就混合当真值。

阶段09复用本文件产生的模型、参数和报告，开展全环境校准/随机化；不得另复制一份接触拟合实现。

## 2. 三步实验共用同一套接触模型

| 层级 | 具体实验 | 执行阶段 |
| --- | --- | --- |
| A 固定板＋可动足底 | 加载、拉动起滑、持续滑动、扭转、卸载，隔离摩擦 | 本阶段 |
| B 自由板＋可动足底 | 保持A的材料与模型，释放板；区分脚板滑移、轮地滑移、滚动 | 本阶段 |
| C 完整机器人＋板 | 由全身控制产生双脚载荷，检查接触切换和滑移 | 05机器人合同确定后在06接入，07/08复验 |

足垫必须在编译后的模型中真实存在，具有质量和可运动自由度。先简单几何打通，再用最终机器人足底碰撞代理重验；不能用固定装饰脚或不存在的foot body。

固定板、力控制夹具、位置控制牵引与辅助约束必须在窗口中标明。修改模型后回归A/B，不能在每一步随意调大摩擦使画面稳定。HUSKY提供滑板/整机任务参考，不自动提供适合本机器人和材料的摩擦参数。

## 3. 数据及资料来源：调研结论

核查日期：2026-09-28。已阅读作者/官方数据说明、下载入口及部分仓库原始字段说明。**本次没有下载并解析所有完整原始数据**；OpenCode必须完成小样本下载、单位/字段/许可核验后才能训练。

### D1 — GTF：机器人足底静摩擦先验

Brandão、Hashimoto、Takanishi，Humanoids 2016，Friction from Vision。

- [作者项目](https://www.martimbrandao.com/friction-from-vision/)
- [作者下载入口](https://drive.google.com/file/d/1tJ9ES2kYcl6kNVjqOG6oqXN_bPdQJHTt/view)
- [论文PDF](https://www.martimbrandao.com/papers/Brandao2016-humanoids-friction.pdf)

43种表面的机器人足底摩擦系数、图像、材料标签及数据划分；论文GTF测量的是拉至起滑的静摩擦。适合静摩擦先验与实验设计，不足以辨识动态阻尼/速度规律。不是G1足底或指定滑板砂纸；下载集合中的OSA+F含人类主观判断，不能当测量真值。

状态：作者页、论文和Drive入口已核查；原始文件与数据许可待执行者检查。优先级：参数先验高，动态网络监督低。

### D2 — SENS3：砂纸接触、力与运动数据

- [作者项目](https://www.sens3.net/)
- [字段与下载说明](https://www.sens3.net/surfaces)
- [砂纸子集](https://www.sens3.net/surfaces/sandpaper)

包含真人裸指与砂纸等表面的实验；Sliding记录有力/力矩、位置和速度。可先选砂纸小子集，比较解析摩擦与小网络的受控运动力预测。

限制：皮肤—砂纸不是机器人橡胶—砂纸。不同传感器速率需同步；力矩单位N·mm须转换；部分加速度换算说明需进一步核对。运动若由人或夹具施加且输入未知，只能做条件力预测/回放，不能冒充自由动力学rollout成功。

状态：作者字段与材料入口已核查，具体文件/许可待下载审计。优先级：力监督方法实验高；目标参数直接迁移不允许。

### D3 — MIT MCube Push Dataset：真实动力学方法对照

Yu等，IROS 2016，More than a Million Ways to Be Pushed。

- [数据、格式、下载和处理代码](https://web.mit.edu/mcube/push-dataset/)
- [论文](https://arxiv.org/abs/1604.04038)

平面推物的位姿与推杆受力，JSON/H5；包含PU、胶合板等表面。主页定义tip_pose、obj_pose、ft_wrench，提示原始时间轴需同步。

选择单一物体/表面的少量完整试验，验证参数辨识或网络动力学预测。ft_wrench是推杆处的受力，不是物体底面完整摩擦力；必须重建推杆接触、质量/几何和底面动力学，不能直接改名foot_deck。PU/胶合板也不等于滑板砂纸。

状态：作者主页及格式已核查，外部下载可用性和文件质量待检查。不要整包迁移其机器人采集工程。

### D4 — ContactNets真实抛块数据：接触切换方法基准

- [论文](https://proceedings.mlr.press/v155/pfrommer21a.html)
- [作者仓库与数据说明](https://github.com/DAIRLab/contact-nets)
- [轨迹目录](https://github.com/DAIRLab/contact-nets/tree/main/data/tosses_processed)
- [参数目录](https://github.com/DAIRLab/contact-nets/tree/main/data/params_processed)

预处理的真实抛块位姿/速度轨迹，释放后控制为零。README给出位置与线速度乘0.0524恢复米制、四元数实部在前；采样间隔须另外核对。可用于可微/结构化接触轨迹实验，不能作为脚板材料参数真值。

已核对README和API目录，未逐文件解析。旧仓库标记弃用，且包含不同第三方许可；仅取必要数据/方法，不运行旧setup覆盖skate-sim环境。无接触力传感标签时不得伪造force loss。

### D5 — Cluster Haptic Texture Dataset：橡胶工况资料

Eguchi等，Scientific Data，2026。

- [论文](https://www.nature.com/articles/s41597-026-06760-z)
- [代码](https://github.com/cluster-lab/Cluster-Haptic-Texture-Dataset)
- [原始字段说明](https://github.com/cluster-lab/Cluster-Haptic-Texture-Dataset/blob/main/documents/dataset_details.md)
- [数据DOI](https://doi.org/10.6084/m9.figshare.29438288)

聚氨酯探头在纹理表面滑动，包含法向力、位置、振动等及材料摩擦表；主实验载荷0.5/1 N，速度20—60 mm/s。

关键限制：滑动序列force主要是**法向力**，不是三轴接触力，不能直接监督切向摩擦；记录速率也不等于力传感器独立测量带宽。探头形状/材料及低载荷不能直接推广到机器人足底。

已读取作者原始字段说明；数据页有mini子集说明，需确认下载版本与材料清单。用途：工况/材料先验、数据处理与实验设计，而非默认动态力监督。

### D6 — 橡胶—粗糙硬基底摩擦/磨损数据

Tanaka等，Spectral wear modelling of rubber friction on a hard substrate with large surface roughness。

- [论文DOI](https://doi.org/10.1098/rspa.2023.0587)
- [Dryad数据与README](https://doi.org/10.5061/dryad.ttdz08m4v)

发布了实验和数值数据、CSV/Excel及代码，可按章节下载。适合粗糙度/磨损影响的物理先验和参数扫描设计，不等于目标胶底—砂纸测量。混合的实验/数值结果必须分开标记；部分章节采用N、mm、MPa，不能全按SI解析。

发布页约3 GB，已核对文件说明；先取README与小章节，避免全量盲下。首版不据此扩展完整磨损求解器。

### D7 — 足式机器人参数估计方法补充

- [Physical Terrain Parameters Learning，作者代码与数据链接](https://github.com/leggedrobotics/physical_terrain_parameter_learning)
- Chen等，Identifying Terrain Physical Parameters From Vision — Towards Physical-Parameter-Aware Locomotion and Navigation，RA-L 2024。

作者提供足端摩擦/刚度估计方法与数据入口，模型面向ANYmal D，并提示估计噪声及适用范围。可参考历史状态估计参数，但估计器不是接触力求解器，其输出不能当绝对测量。

数据各部分的真实/仿真属性尚未核验，仅列候选。不要引入ROS部署或完整视觉训练框架。

### 数据选择结论

- 优先审计D2做有力数据的异材质方法比较；D4或D3择一做真实轨迹方法实验，避免移植两个完整基准。
- D1用于机器人足底静摩擦先验，D5/D6用于橡胶与粗糙材料参考，D7非主线。
- 不同材质来源不能直接拼成一个通用脚板数据集；每个源域单独训练/评价。
- 下载受限就记录具体问题，尝试合法公开入口；不要伪造数据或未经授权联系作者。
- 目标胶底配方/硬度、砂纸型号/磨损、接触面积与载荷范围仍需目标测量补齐。材料名称相近不构成迁移证据。

## 4. 数据审计与目标材料补测

接触数据统一放在项目根目录下的 `data/contact/`，即用户的 `workspace/skate-sim/data/contact/`。OpenCode负责创建目录并将下载数据保存到这里，不默认改用外部路径或runs目录。按数据来源组织：

- `data/contact/<source>/raw/`：下载的原始文件，保存后只读，不覆盖原始内容。
- `data/contact/<source>/processed/`：单位统一、时间对齐后的数据及划分清单。
- `data/contact/manifest.json`：各来源、版本、文件hash、许可、处理记录与审计状态。

`<source>`使用简短来源标识，例如sens3、contactnets、mit_push。下载与处理数据默认不入Git，配置使用项目相对路径；加入对应忽略规则。拟合参数、模型权重和实验指标仍放在 `runs/04_fit/`，与数据分开保存。

在上述manifest中记录：SOURCE_VERIFIED、DOWNLOADED、PARSED、QUALIFIED。当前调研支持说明页/字段核查，不能替OpenCode标数据已QUALIFIED。

每份数据必须记录：
- URL/DOI、版本/commit、许可、文件hash、大小、trial数与排除原因；代码许可不自动代表数据许可。
- 接触双方材质与几何、干湿/温度/磨损、载荷或压力、速度/方向范围。
- 受力对象、坐标系、单位、采样间隔、同步与有效带宽；区分命令、测量、估计。
- 真实/合成/数值标签，支持的任务：coefficient_prior、force_fit、trajectory_fit。
- train/validation/test的完整试验ID；重复试验、同一采集session不得泄漏。

内部Trial包含source_id、trial_id、material_pair、origin、units、frame、可用字段mask；按任务提供time、pose/velocity、normal_force、tangent_force/wrench、external_input、contact_state及质量/惯量/几何元数据。缺失量不填零冒充测量；没有时间轴的摩擦表只能作prior。

原始数据只读；处理时保留转换记录：
1. 米/秒/牛顿/牛米统一，检查角度与四元数顺序。
2. 时间戳对齐，降采样先抗混叠；插值不制造测量带宽。
3. 偏置、滤波延迟、尖峰和姿态跳变处理可追溯，不能删除难拟合片段美化结果。
4. 按完整trial/试件/session划分；归一化只使用train；另留速度或载荷工况检查泛化。
5. 在play.py展示原始/处理后曲线，人工确认量纲、方向与数据含义，再训练。

目标材料缺口的最小补充建议：
- 实验室在实际足底材料/几何与指定砂纸上，测法向载荷、水平拉力、相对位移/速度、加载方向及时间戳。
- 起始设计可用3档载荷×3档速度×2方向×至少3次独立重复，再加起滑、卸载和扭转。量程按机器人单脚/双脚和动态工况确定，不用探头低载荷直接缩放。
- 拉力只有在准静态/匀速且其他作用可忽略时才近似摩擦；动态时考虑惯性和夹具输入。
- 这是需要实验室另行提供的数据规格，不要求在本纯仿真项目中编写硬件采集驱动。

## 5. 三条模型路线怎么实现

### 5.1 原生接触：默认基线

设置有效材料对foot_deck、wheel_ground、foot_ground、body_ground；核实编译和运行时实际参数。MuJoCo明确condim、摩擦、solref/solimp和参数合并，不能只写YAML。几何使用有限面积足底代理，比较简单/最终形状；扭转实验验证接触力矩。

接触点相对滑移使用两刚体在接触位置处的点速度差，再取切向分量。读取法向/切向力、力矩、分离状态和耗散功率；不把世界脚速当滑移，不把冲量当力。

默认由引擎求接触，不再额外叠加一份μN。参考 [MuJoCo建模](https://mujoco.readthedocs.io/en/stable/modeling.html) 与 [ContactBench](https://github.com/Simple-Robotics/ContactBench)；后者是模型/求解器基准，不是目标材料测量库。

### 5.2 可微辨识：既要有力拟合，也要区分真正的动力学rollout

两种任务必须分开：
- force_fit：给定测得载荷/运动，预测接触力，对参数求导。它是可微接触规律拟合，不是完整动力学预测。
- trajectory_fit：给定初值、几何/质量、已知输入，积分预测运动并对轨迹误差求导。真正报告“可微仿真实验”必须实现这一项。

首版在contact.py中用PyTorch实现匹配实验的低维足垫滑块模型：例如已知法向载荷下的平滑滑动摩擦与显式时间积分。明确可支持自由度、加载方式与适用速度；平滑模型不等于零速精确静摩擦。先解析/合成验证，再用有足够输入的真实轨迹。D4抛块不是滑块，若选其数据就须实现匹配源几何/自由度的模型，不能把轨迹压成1D冒充复现。

可选完整MJX/JAX路径须核实锁定版本的梯度支持与环境兼容；不能因为用了GPU/Warp就声称可微。简单力函数求导也不能伪装成引擎rollout求导。

执行要求：
1. 只拟合可辨识的少数参数，例如摩擦；缺法向变形时不同时强拟合刚度、阻尼、质量。
2. 预测值映射到实际测量量，按单位/噪声尺度定义loss，不给缺失量构造虚假标签。
3. 同一前向模型、数据划分、参数边界下，比较梯度优化与CEM/有界搜索。
4. 用多档有限差分步长检查平滑段梯度；切换处单独报告。参考 [Zhong等，2022](https://arxiv.org/abs/2207.05060)。
5. 输出独立留出误差、参数、模型版本和适用域。
6. 只把语义一致参数映射到实际MuJoCo；再跑原生实验检查映射损失。平滑模型参数不自动等于solref/solimp，映射失败保留离线实验身份。

合成实验要有不同激励/噪声与留出参数，避免同设置自拟合。公开异材质的真实拟合结果只能宣称源任务有效。

### 5.3 接触网络：受约束参数模型先行

首版用小MLP预测工况相关有效摩擦，作为本项目候选，不称为复现ContactNets：
- 输入：相对切向速度/方向、可取得的载荷、材料编码；数据支持时增加短历史。
- 输出：通过有界变换限制的摩擦参数。边界由数据/工程假设注明，不能默认所有μ都小于1。
- 输入不得包含待预测的当前切向力。载荷若由引擎求出，明确使用上一物理步值或外部已知值，不能形成未求解的循环依赖。
- 有真实切向力时做force loss；没有力但有足够动力学条件时才做rollout loss。
- 与常数μ及简单速度相关解析模型比较，不能只和未校准模型比。
- 非接触不出力、切向耗散合理；有弹性历史时单独计储能。检查输出范围、OOD、重置与长时稳定性。

ContactNets实际学习距离/接触雅可比等隐式结构，并有专门接触损失，**不等于MLP预测摩擦系数**。[论文](https://proceedings.mlr.press/v155/pfrommer21a.html)作为结构化学习参考；如额外复现须明确标注。D1的静摩擦表和D5的法向力序列不能直接当动态切向力标签。

网络没有留出提升也可形成完整研究结果，但不替换默认原生模型。合成数据训练只能证明程序流程，不能宣称真实材料改进。

### 5.4 接入场景与三个后端

先离线对照，再尝试在线候选。参数网络在明确步边界更新目标材料对，由引擎计算接触力，不在原生摩擦上另加完整预测摩擦力。注意逐接触/逐环境参数范围，不能改共享材料导致全部环境一起变化。

某后端不支持必要更新时明确标记候选不支持，保留原生基线。自定义力替代牵涉切向求解器、法向力获取和力矩分配，不能作为未验证捷径。网络未接入时，界面不能留neural标签实际跑native。

先通过固定板，再自由板，在06接入机器人，07/08验证后端差异。阶段04不预先承诺三后端网络结果一致。

## 6. 文件组织：一个contact.py负责实验核心

| 位置 | 职责 |
| --- | --- |
| `src/skate_sim/physics/contact.py` | 接触数据规范/最小适配、接触规律、可微前向、小网络、拟合与评价 |
| `src/skate_sim/models/scene.py` | 固定板、自由板、源实验fixture构建，不复制公式 |
| `src/skate_sim/backends/*.py` | 引擎接触读取、有效参数设置、坐标/状态转换 |
| `src/skate_sim/cli.py` | 非可视化contact子命令的薄调度 |
| 根目录 `play.py` 与共享viewer | 唯一窗口入口，场景/数据/拟合曲线/回放 |
| `configs/contact.yaml` | 数据源、划分、模型、参数边界、拟合和显示配置 |
| 现有tests | 针对单位、梯度、泄漏、耗散等风险的短小持久回归 |
| `data/contact/` | 按来源分raw/processed保存接触数据，统一审计清单，下载与处理数据默认不入git |
| `runs/04_fit/` | 参数、模型权重、指标、manifest |

contact.py建议提供load_trials、predict_contact、rollout_contact、fit_contact、evaluate_contact、ContactModel等少量清楚接口，按实际需要实现，不预造插件框架。

同一公式用于拟合、评价和play对照；不新建contact_train.py/contact_diff.py/contact_network.py等重复实现。已有calibrate.py仅复用优化工具/薄调度，不重复损失与模型。文件过大先去冗余，再有依据拆分通用功能，不能为了单文件牺牲可读性。

模块不在import时训练/下载/打开窗口，不导入全部引擎。缺少依赖由OpenCode安装到既定skate-sim环境并验证兼容；不照搬旧研究仓库的环境安装脚本。

## 7. 实施顺序

1. 04A：完成原生pull/twist/unload/freeboard实验与诊断。
2. 04B：实际下载审计最小真实数据，字段/单位/许可/划分合格后，用play.py展示。优先D2；轨迹路径D4或D3择一。
3. 04C：可微rollout合成真值验证必做，再进行合格真实数据实验。force_fit单独记，不代替rollout。
4. 04D：完成小网络拟合、简单物理基线对照、留出评价；实验可以无提升，但不能是空实现。
5. 04E：映射候选到原生实验台并复验A/B；不支持映射时保留明确的离线候选，解释差异。
6. 评审：原生、合成方法、公开真实方法、目标材料验证分开列状态。

合格真实数据不足时报告BLOCKED及缺失字段，继续其他可做工作。用户可明确批准先用原生基线进入05，但不能自行把研究任务删除或标通过。目标材料暂缺不等于公开方法实验无需做。

## 8. 要求实现并实测的命令

所有命令从项目根目录执行。路径为目标示例，交付时替换实际数据/记录路径。内部contact子命令是本阶段对12工具入口的扩展；可视化入口不增加。

### 原生交互

```bash
conda activate skate-sim
python play.py --backend mujoco --scene contact --experiment pull --contact-model native --record runs/04_pull
python play.py --backend mujoco --scene contact --experiment twist --contact-model native --record runs/04_twist
python play.py --backend mujoco --scene contact --experiment unload --contact-model native --record runs/04_unload
python play.py --backend mujoco --scene contact --experiment freeboard --contact-model native --record runs/04_freeboard
python -m skate_sim check --stage 04 --backend mujoco --suite native --output runs/04_check
```

### 数据与实验：CLI调用contact.py

```bash
python -m skate_sim contact audit --config configs/contact.yaml --output data/contact
python -m skate_sim contact fit --method cem --config configs/contact.yaml --output runs/04_fit/cem
python -m skate_sim contact fit --method diff --config configs/contact.yaml --output runs/04_fit/diff
python -m skate_sim contact fit --method neural --config configs/contact.yaml --output runs/04_fit/neural
python -m skate_sim contact evaluate --config configs/contact.yaml --runs runs/04_fit/cem runs/04_fit/diff runs/04_fit/neural --split test --output runs/04_eval
```

配置显式指定force_fit或trajectory_fit、源数据和固定划分；命令必须拒绝不满足数据要求的任务，不能自动换成合成数据继续PASS。只比较同任务/同观测定义下的误差，不能把力误差与位姿误差混合排名。

### 数据与拟合结果窗口

```bash
python play.py --scene contact --experiment dataset --config configs/contact.yaml
python play.py --scene contact --experiment compare --config configs/contact.yaml --runs runs/04_fit/cem runs/04_fit/diff runs/04_fit/neural
python play.py --backend mujoco --scene contact --experiment pull --contact-model diff --model-path runs/04_fit/diff
python play.py --backend mujoco --scene contact --experiment pull --contact-model neural --model-path runs/04_fit/neural
python play.py --replay runs/04_freeboard --mode visual
python -m skate_sim check --stage 04 --backend mujoco --suite all --output runs/04_full
```

dataset/compare显示曲线及适用源实验几何，不把裸指数据无标识渲染成G1脚。diff在线表示加载经映射验证的拟合参数，不表示普通后端正在反向传播。仅有离线候选时，两个在线候选命令须明确拒绝，compare仍可用；不静默回退native。

进入07/08后，用同样的play.py参数复验各后端原生contact场景；学习候选的上线状态另行确认。

## 9. 用户可视化验收

- 足底压板、缓慢加拉力：看到起滑、力与相对滑移；高低摩擦改变真实响应。
- 切载荷、速度、方向、几何：看接触分布和力矩变化；卸载接触消失。
- 自由板展示作用反作用，不只播放固定板结果。
- 面板显示数据来源/材料对、测量范围、模型版本、在线/离线、外力/夹具。
- dataset显示原始/处理曲线、单位、缺失字段、划分。
- compare在同一留出试验显示测量、解析基线、可微拟合、网络预测；未测量量不画伪真值。
- 力拟合标“实测运动驱动”，动力学rollout标“自由预测”，不能混称。
- 相同初值和输入切模型，观察误差；公开源域效果不写成目标机器人精度。
- R清空接触/网络历史与外力；暂停不推进history，N/M分别物理步/控制步。
- 修复当前问题再复测，用户确认后才能推进。

## 10. 自动关卡与交付

原生接触：
- 三档载荷/三档摩擦/至少两档速度，检查实际行为而非字段存在。
- 准静态起滑与μN对照，初始工程门槛可取15%；先冻结加载与起滑判据，不当真实材料精度承诺。
- 检查有效材料、力/冲量、坐标、分离、功率与dt敏感性；修脚板后回归03轮地滚动。

数据与学习：
- manifest与实际文件一致，许可/单位/受力对象可追溯，未测量数据不补标签。
- 分组留出、归一化无泄漏，train与validation不是同一轨迹。
- 可微路线有真正rollout、有限差分和合成参数恢复；列不可辨识参数及接触切换梯度限制。
- 网络比较常数及简单解析模型，检查边界、耗散、无接触、OOD和长时稳定性。
- 报告有单位的test指标、重复seed变化、拟合耗时与适用域；训练loss下降不等于验收。
- 目标材料未测量时不报告真实脚板提升百分比；方法无提升可如实完成实验，代码占位不可。
- 必需项未执行/阻塞时all检查非零退出。物理基线通过不能替代研究项完成。

最终交付：
1. 一个可维护的physics/contact.py，按12复用场景/后端/CLI/viewer。
2. 一份configs/contact.yaml、最小数据适配及运行记录；不提交无关数据或大权重。
3. 原生实验台、可微rollout、接触网络对照的实际结果和适用边界。
4. 全部play.py命令与操作说明经过实测。
5. 在已有docs/stages/04.md汇总数据审计、指标、失败/修复和人工确认，避免大量重复报告。
6. README补充数据准备/实验入口；09及06—08复用本模块，不复制实现。

可以得到“公开材料无法迁移”“网络无收益”“参数不可辨识”的研究结论；必须如实展示。不能把下载入口、方法名称或空接口当作完成。
