# 阶段03：被动滑板动力学与可交互微基准

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段02已 ACCEPTED。不得先上机器人来掩盖板单体问题。

## 可见成果

用户在窗口中推板、给板面加载/倾斜、释放后观察回正；同时看前后轮轴投影、转向角、板面倾角、轮速与轨迹。可切换约化模型 reduced 和显式 truck 模型，比较相同输入下的差异。

## 方法对比与本阶段决策

| 方法 | 优点 | 主要风险 | 本阶段用途 |
| --- | --- | --- | --- |
| 借鉴 HUSKY 的约化倾斜—转向耦合 | 参数少、容易稳定、便于对照 | 关系可能过强；不完整描述实际 truck 几何 | 作为有来源的对照模型 |
| 显式倾斜转向架轴＋恢复力矩 | 刚体拓扑清楚，可跨引擎表达 | 接触/约束求解器、衬套参数敏感 | 优先候选主模型，须经测试选择 |
| 多连杆详细 kingpin/bushing 模型 | 更接近机械结构 | 参数难辨识、数值刚性大、资产成本高 | 第一版不默认采用 |
| 直接控制 yaw 让板转弯 | 演示简单 | 板不再被动；掩盖动力学错误 | 禁止作为物理模型 |

先从作者源码/论文重建约化关系和载荷路径，记录其变量定义。不能只抄一个 -0.577 系数，也不能把约束施加给不承载板面/轮组的空 body。前后符号必须通过统一坐标与实际轮轴方向验证。

显式模型从几何确定 steering axis，配置恢复力矩，例如 τ=-k1·q-k3·q³-c·qdot；k1、k3、c 为起始/待辨识参数。需关节限位和扭矩连续性检查，避免高刚度与大步长组合引入抖振。是否保留立方项由数据/实验决定，不放无效配置。

## 必做实验

1. 直线滑行：给初速，不施驱动轮矩，观察速度、轮速、路程和停止过程。
2. 倾斜—转向：夹具施加左右对称板面倾斜，显示 axle 的地平面投影角；不能拿 inclined joint 的 q 值当真实 steer。
3. 动态转弯：有限初速下用明确的载荷/侧向加载激励倾斜，观察左右转弯轨迹。自由空板不一定维持固定 roll，因此夹具测试与自由测试必须分开。
4. 回正/衰减：施加初始倾角或关节位移，释放后测恢复与阻尼。
5. 四轮接地、失去接触、重接触：看法向力、轮速跳变与约束稳定性。
6. 时间步收敛：dt 与 dt/2 比较同一实验，优先排除数值伪差再调参数。

轮损耗分清轴承阻力矩、轮地滑动、滚动阻力；原生模型已有项不能再无意叠加。使用非负耗散设计，近零速平滑/静止处理，禁止符号错误使板自动加速。

## 测试命令

```bash
python play.py --backend mujoco --scene board --model reduced --experiment coast --seed 0 --record runs/03_reduced
python play.py --backend mujoco --scene board --model truck --experiment coast --seed 0 --record runs/03_truck
python play.py --backend mujoco --scene board --model truck --experiment lean --record runs/03_lean
python play.py --backend mujoco --scene board --model truck --experiment release --record runs/03_release
python -m skate_sim check --stage 03 --backend mujoco --styles all --output runs/03_check
python play.py --replay runs/03_lean --mode visual
```

窗口须允许调外力幅值/时长、左右加载、释放、显示路径；参数变化要重建或可靠热更新，角标展示有效值。

## 用户验收操作与异常信号

- 平地推一下后释放：板应靠惯性运动，不自行持续提速。
- 左右施加等幅加载：实际前后轮转向关系及路径应符合几何定义，并呈合理镜像。
- 倾斜后释放：能看到恢复/衰减，不持续增幅。
- 切滚动损耗：增加损耗在同初值下应增加衰减，不出现反向驱动。
- 若“轮不转板却滑”“板转向但轮轴完全不变”“静止就抽搐”“释放后无限加速”，标 FAIL 而不是调镜头/提高摩擦掩盖。

## 定量关卡

以下为初始工程门槛，不是已证实真实误差范围；阶段首次跑测前写入固定配置，之后变更须说明并重新验收。
- 关节与接触状态无NaN；三样式各至少60 s被动试验。
- 直线名义对称实验无显著无输入偏航；具体偏航限按模型对称性和求解误差预先登记。
- 左右转向曲率符号正确；同幅值镜像实验的幅值差建议不超过10%，若不满足先排查几何/接触。
- 无滑滚动段轮缘速度与沿轮滚动方向接触相对速度对应；接触残差定义不能把转弯几何速度错当滑移。
- dt减半后的短时轨迹/回正指标差建议小于5%，否则继续定位，不能当作真实参数差。
- 机械能统计包含重力势能/弹性能，扣除外力做功；不把重力引起的动能增长判为非被动。
- 选择主模型时给证据：可见行为、稳定性、可解释参数、跨后端可实现性。若 truck 未胜过 reduced，可保留 reduced 为默认，但不能称其高保真。

修复后重跑02资产回归及所有03实验；用户确认主模型和动态效果后进入04。

参考：[HUSKY 论文](https://arxiv.org/abs/2602.03205)、[作者代码](https://github.com/TeleHuman/humanoid_skateboarding)、[MuJoCo Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)。

