# 阶段09：参数可解释性、随机化与sim2real准备

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段08已 ACCEPTED。项目保持纯仿真，不新增真机部署、训练平台或硬件采集链路。

## 可见成果

用户在相同场景中对比名义参数、扰动参数及校准候选，观看滑行/回正/脚滑移轨迹与曲线。界面显示实际生效参数、seed、参数来源和是否真实数据校准。

目标是尽可能覆盖主要误差来源并建立可验证改进链路，不承诺无真实数据即可“解决sim2real gap”。

## 误差清单与处理次序

| 误差源 | 本轮机制 | 验证 |
| --- | --- | --- |
| 尺度/质量/惯量/质心 | 参数化资产、配置与编译值核验 | 尺寸和惯量检查、摆动/响应 |
| truck几何/衬套 |03两模型对照、恢复/阻尼参数 | lean、release、曲率 |
| 轮地/轴承损耗 |区分滑动、滚阻和轴阻 | 滑行/轮速/能量 |
| 足底材料/面积/柔顺 |原生接触与04微基准 | 起滑、滑动、扭转、卸载 |
| 执行器/控制时序 |PD、饱和、延迟、可选结构化执行器 | 关节响应与oracle |
| 观测与状态估计 |观测噪声/延迟的明确注入点 | 干净/噪声对照与重置 |
| 求解器差异 |dt收敛与跨后端物理指标 |03/04/08复测 |
| 未建模接触规律 |有真实数据才启用受约束残差 | 独立真实留出评价 |

先隔离执行器与几何误差，再拟合摩擦；不能让摩擦系数吸收所有控制错误。

## 域随机化：必做

1. 区分nominal参数和episode参数。每次reset先恢复nominal再采样；不得在当前值上累乘。
2. 支持有物理边界的质量/质心、摩擦、truck参数、轮损耗、PD/延迟/噪声。惯量随质量/几何调整须保持正定与物理可行性。
3. 配置必须实际生效；对每个开关做行为敏感性测试，不保留只写进YAML的wheel_mass、noise等。
4. 独立按env_id reset/randomize，不改变其他环境。固定seed可复现，采样参数入日志。
5. 参数之间有物理相关性时联合采样，而非无限独立组合。范围标明“工程假设”或“真实辨识”，不虚构可信区间。
6. 窗口显示nominal/当前参数；可重置为同一seed做对照。

## 离线辨识：必做最小实验，有数据分层

- 没有真实数据：用合成但分离的激励/噪声/参数生成测试数据，只验证拟合管线、参数可辨识性和留出泛化。必须标“合成验证”。
- 有用户提供的真实测量：仅导入离线文件；记录单位、时间戳、测量误差、同步/滤波、缺失值和使用许可。按独立试验或整条轨迹划分train/validation/test，禁止随机切同一轨迹点造成泄漏。
- 拟合目标至少能对应板回正或滑行的真实forward模型，不能拟合无关弹簧玩具后宣称改善了滑板。
- 首选有界无导数优化（如CEM）作可靠基线；可微方法为对照，需自动微分与有限差分多步长检查，并覆盖接触切换附近。
- train与held-out使用不同轨迹/激励；训练和验证值完全相同必须解释且检查数据复用，不能直接报告PASS。
- 拟合参数写回同一资产/物理配置，在实际后端重跑03/04并可视化对比。不可只输出一个优化loss。

## 接触网络的进入门槛

同时满足才进入正式候选：有足够多载荷/速度/方向/材料的真实数据；原生模型已校准；真实留出误差仍存在稳定模式；网络改进超过测量噪声且不伤害长期稳定性。

若启用，优先结构化参数估计/受约束残差，满足非负法向力、摩擦界、耗散约束、无接触不出力；明确它与引擎原生摩擦如何组合，避免双计。检查OOD输入、功率/能量和长时rollout。三后端若不能等价接入，只作为清楚隔离的实验候选，不替换已验收基线。

没有数据时结论为“暂不启用”，不是“该功能占位完成”。本轮核心交付不依赖训练接触网络。

## 测试命令

```bash
python play.py --backend mujoco --scene board --experiment release --params nominal --record runs/09_nominal
python play.py --backend mujoco --scene board --experiment release --randomize --seed 17 --record runs/09_random
python play.py --backend mjlab --scene rider --controller pose --num-envs 64 --randomize --seed 17 --view-env 7
python -m skate_sim calibrate --experiment release --data synthetic --method cem --output runs/09_fit
python play.py --backend mujoco --scene board --experiment release --params runs/09_fit/params.yaml --record runs/09_fitted
python -m skate_sim check --stage 09 --backend all --output runs/09_check
```

有真实数据再追加：
```bash
python -m skate_sim calibrate --experiment release --data /absolute/path/measurements --split trial --method cem --output runs/09_real
```

OpenCode须明确合成数据怎样生成、哪些试验留出；不能让同模型同参数同激励的自拟合冒充真实验证。若未实现可微对照，不在README列为可用功能。

## 用户验收与关卡

- 切高低摩擦、阻尼/质量，观察对应变化并能回到nominal。
- 连续同seed reset 100次，参数不累计漂移，曲线复现；不同seed分布符合声明。
- 显示校准前后曲线及留出误差，不仅训练曲线。
- 用合成真值检验参数误差/可辨识性，拟合未达到预登记门槛应FAIL，不能无条件PASS。
- 校准质量由独立留出误差、参数合理性、forward重放以及物理稳定性共同判定。没有真实数据时不生成“真实sim2real提升x%”结论。

用户确认参数响应、可信边界和所选默认模型后进入10。

参考：[ContactBench](https://github.com/Simple-Robotics/ContactBench)、[ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html)、[梯度可靠性研究](https://arxiv.org/abs/2207.05060)、[SPI-Active](https://arxiv.org/abs/2505.14266)、[DROPO](https://arxiv.org/abs/2201.08434)。

