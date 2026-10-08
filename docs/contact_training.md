# 离线接触参数训练接口

`src/skate_sim/physics/train_parameters.py` 是小型离线辨识入口；实现复用
`src/skate_sim/physics/contact.py`。这是用户在阶段09明确要求增加的参数训练，
不包含机器人策略训练、硬件采集或部署。MuJoCo/mjlab 的原生接触保持现有路径。

## 可以运行的两条路径

在项目根目录、已有 `skate-sim-mjlab` 环境运行；单独安装时可用
`pip install -e '.[contact]'`。`--help` 不要求安装 Torch 或物理引擎。

```bash
conda activate skate-sim-mjlab

# 显式合成数据：由已知形变/相对速度拟合接触力。
python -m skate_sim.physics.train_parameters --data synthetic --task force_fit \
  --model analytic --seed 17 --epochs 150 --output runs/sole_force

# 显式合成数据：从初始状态积分整段运动，反向传播拟合轨迹。
python -m skate_sim.physics.train_parameters --data synthetic --task trajectory_fit \
  --model analytic --seed 17 --epochs 150 --output runs/sole_trajectory

# 在已校准的解析基线上训练受约束的小型参数网络。
python -m skate_sim.physics.train_parameters --data runs/sole_force/dataset.json --task force_fit \
  --model contact_net --init runs/sole_force/model.pt \
  --seed 17 --epochs 150 --output runs/sole_network

# 唯一窗口入口，默认显示独立 test 集的第一条完整试验。
python play.py --contact-run runs/sole_force
python play.py --contact-run runs/sole_trajectory --contact-trial test-000
python play.py --contact-run runs/sole_network
```

输出目录必须为空；不覆盖旧实验。默认 CPU/float64，150轮正常训练；窗口不会
自动关闭。`play --contact-run` 显示目标/基线/训练结果的法向力、切向力、压力中心
或足底高度，以及训练/验证曲线和 test loss。这是离线数据对照窗口，不是原生
机器人接触仿真。没有测量力时不绘制虚假力标签。

## 方法与物理语义

- `force_fit`：输入每个触觉单元的有符号 gap、足底相对平移速度，拟合法向单元力
  和切向合力。gap<0 表示压入。刚度/阻尼是整只足底的总值，均匀分摊到单元。
- `trajectory_fit`：固定水平板、无旋转足底、已知质量、三维平移；由初始位置/
  速度和每步已知外力，用半隐式 Euler 积分，切向阻力使用线性化隐式更新，避免
  减速接近零时显式步进导致速度反向。中间步不注入测得状态，不作 teacher
  forcing。此任务可以没有任何力标签，但必须有完整位置/速度轨迹与外力。
- 法向力为 `max(k * penetration - c * vz, 0) / taxel_count`，gap>=0 时严格为零；
  切向力沿速度反向，使用 `sqrt(vx²+vy²+0.01²)` 平滑零速奇点。它没有精确静摩擦、
  扭转接触、足部转动、板的自由运动或复杂碰撞切换。
- 学习参数用 sigmoid 约束：刚度 2000–30000 N/m、阻尼 5–150 N·s/m、摩擦系数
  0.05–1.5。边界为工程假设，并非真实材料置信区间。
- `contact_net` 为 6→16→3 MLP，输入 gap、三维相对速度、单元 xy 位置；不读
  当前目标力。输出刚度/阻尼/摩擦的最多±20%修正并裁剪到物理边界。
  全局参数冻结在解析拟合结果，先比较校准基线再评估网络增益。
- 法向力非负、分离无接触力、摩擦界以及切向/阻尼耗散都被验证。
  状态相关网络刚度尚无统一储能函数，因此不能据此宣称整个网络模型严格被动。

拟合用 Adam；固定损失尺度（单元力0.5 N、切向力5 N、位置0.005 m、速度0.1 m/s），
每条试验等权。训练只使用 train；validation 选择最佳 epoch；test 不参与优化或
选模，仅用于最终候选/基线比较。网络初始化须来自同一数据与划分的解析拟合，
防止基线曾见过 test 数据。每个 trial/session 只能属于一个 split。

梯度检查对多步 rollout 使用 float64 自动微分，对照 1e-3/1e-4/1e-5 三档中心
差分（raw 参数方向）；绝对误差阈值 `1e-6 + 1e-3 * abs(autodiff)`。
接触切换是分段可微边界，gap=0 两侧另做约束测试，不宣称边界有唯一导数。

`PASS` 要求所有指标有限、物理约束通过、梯度检查通过且 test loss 小于相应
基线；否则输出 `FAIL` 并返回1，但仍保留研究结果。该状态只是离线方法检查，
没有真实数据时不能解释为实际材料参数被辨识或 sim2real 得到改善。真实数据的
应用精度门槛仍需根据测量误差另行约定。网络没有提升也是允许保存的实验结果。

## 后续真实数据接口

可先查看自动生成的 `runs/sole_force/dataset.json`，它是完整可加载样例。
JSON 顶层为 `schema: skate-sim.contact.v1`、`manifest`、`splits`、`trials`。

`manifest` 必需：source_id、source（URL/出处）、version、license、task、
`force_semantics: force_on_foot_deck_frame_z_up`、`taxel_positions: [M,2]`。
`splits` 必须是非空的 train/validation/test trial ID 列表，覆盖所有试验且不重叠。

每条 trial 必需：

| 字段 | 形状/含义 |
| --- | --- |
| trial_id / material_pair | 非空试验ID、足底与板面材料对；每次拟合只用一个材料对 |
| origin | synthetic / public_real / target_real；不混用来源类型 |
| frame | fixed_deck_z_up；力为作用于足底的力，z轴朝上 |
| time | [T]，秒，有限且严格递增 |
| gap | [T,M]，米，压入为负；未知不能填零 |
| foot_velocity | [T,3]，相对固定板的平移速度，m/s |
| taxel_force | [T,M]，法向单元力，N；force_fit必需，不能直接填Pa |
| tangent_force | [T,2]，足底所受切向合力，N；force_fit必需 |
| foot_pose | [T,3]，位置，m；trajectory_fit必需，v1不是6DoF pose |
| external_input | [T-1,3]，每个积分区间外力，N；不含接触力和重力 |
| metadata.session_id | 独立采集session；禁止跨集合拆分 |
| metadata.mass | trajectory_fit必需，已知质量，kg |

`units` 必须精确声明：time=s、gap=m、foot_pose=m、foot_velocity=m/s、
taxel_force=N、tangent_force=N、external_input=N、taxel_positions=m、mass=kg。
所有字段都采用同一单位合同，即使某个可选测量为null，也不能改变其含义。
真实 trial 还必须记录 metadata.measurement_error、synchronization、preprocessing、
field_roles（哪些量是测得、命令或估计）。接口验证必需项存在，不替用户审核这些
文字声明是否准确；实际单位转换、传感器校准和时间同步须在上游完成并记录。

轨迹路径要求均匀 dt<=0.002秒、gap等于足底高度（不支持转动或倾斜板），不能把
任意机器人轨迹压成这个模型。缺失/NaN/单位不符/重复试验/session泄漏均报错。
只有鞋垫压力而没有切向力标签时，不足以启用 force_fit 摩擦训练；应补齐标签或
使用满足模型条件的 trajectory_fit，不把法向压力当切向力。

```bash
python -m skate_sim.physics.train_parameters --data data/contact/my_source/processed/trials.json \
  --task trajectory_fit --model analytic --output runs/sole_real
```

输出 `dataset.json`（带出处与划分的快照）、`model.pt`（模型类型/配置/权重）、
`contact_parameters.json`（带单位的离线参数）和 `report.json`（指标/梯度/历史/hash）。
模型使用 `torch.load(..., weights_only=True)` 读取；数据hash用于发现错配。
输出故意不叫原生 `params.yaml`，不会被 `play --params` 默认为可部署配置。

Python入口为 `load_trials`、`audit_dataset`、`ContactModel.predict/rollout`、
`fit_contact`、`evaluate_contact`、`save_contact_run`、`load_contact_model`。
后续引擎可微适配应实现对应 forward/rollout，明确状态、力和坐标语义，并重新做
多步梯度检查。当前没有 MJX/Warp 自动微分后端，不因用了 Torch 就声称引擎可微。

## TactileStep参考与区别

阅读了 [项目页](https://tactilestep.github.io/) 和
[论文§3.2–3.3](https://arxiv.org/html/2609.28959v1)。它将引擎合力按距离/法向分配
给足底触觉单元，进行空间扩散，提取总力、接触面积、压力中心，并结合接触阶段
设计策略奖励。本文借鉴的是触觉单元数据合同和低维物理特征对齐。

当前实现从局部形变预测单元力，再求总力/面积/压力中心；不是论文的射线投影、
扩散或策略训练复现，也未采用其引擎依赖。压力中心仅在有载荷时有效，离地时
显示缺口；接触面积按每单元0.1 N阈值计算，属于离散诊断，不作为可微损失。
原始鞋垫电信号到力的标定网络需真实标定数据，当前不伪造这个映射。

下一步可用真实样本训练传感器标定/工况参数网络，再比较触觉特征与轨迹误差；
原生引擎接入需通过材料对参数映射和跨后端验证，避免叠加两份摩擦力。

## 回归

```bash
python -m unittest discover -s tests -v
```

覆盖单位/标签/session/重复试验、力约束、自由飞行、分离边界、OOD、参数逐项
梯度、多步梯度及数据/权重往返。合成训练仅验证同一简化模型的参数恢复；不同
试验采用独立激励和力噪声，材料真值保持一致，不声称已经覆盖结构模型误差。
