# 阶段07：mjlab 实际移植、GPU批量与窗口交互

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段06已 ACCEPTED，阶段01已证实目标机器可运行mjlab。不是把MuJoCo窗口改个标题。

## 可见成果

mjlab后端中重现board/contact/robot/rider；可以看单环境，也能从批量仿真中选择一个环境检查、施力和单独重置。显示GPU设备、环境ID、局部坐标和有效参数。

mjlab使用自身环境/实体/场景生命周期；可复用MuJoCo资产与共享控制语义，但reset、device和批量接触处理必须重新验证。

## 实现要求

1. 固定实测mjlab与MuJoCo Warp版本，依据官方Entity/Scene/生命周期接入，不猜API。
2. 为board与robot分别解析root/joint indices；不要默认qpos[:,0:7]是机器人。
3. 保存完整nominal初始state；reset仅写指定env_ids，恢复合法四元数、初始高度、关节姿态、控制器/历史/延迟。
4. batched动作/观测留在正确device，shape与dtype符合合同；history和privileged真实计算，不能填零。
5. 逐物理子步更新PD与被动truck/轮损耗；不存在“GPU所以固定tau”的例外。
6. set_state/get_state支持子集；apply_wrench使用明确env/body/frame/point/duration；清理外力必须逐环境。
7. 随机化使用每环境名义参数与随机数流，reset不累计缩放、不污染未reset环境。
8. 接触诊断实现环境内筛选，有限接触缓存溢出要检测/告警，不静默遗漏。
9. viewer直接消费实际mjlab运行状态。若使用外部显示桥接，它只负责绘制，禁止另跑独立CPU动力学来冒充GPU结果。
10. N=1视觉验证与N=64真实GPU批量验证都做；单环境通过不能代替partial reset。

## 测试命令

```bash
python play.py --backend mjlab --scene board --num-envs 1 --record runs/07_board
python play.py --backend mjlab --scene contact --experiment pull --num-envs 1 --record runs/07_contact
python play.py --backend mjlab --scene rider --controller pose --preset free --num-envs 1 --record runs/07_rider
python play.py --backend mjlab --scene rider --controller policy --checkpoint /absolute/path/model.pt --num-envs 64 --view-env 7 --record runs/07_batch
python -m skate_sim check --stage 07 --backend mjlab --num-envs 64 --checkpoint /absolute/path/model.pt --output runs/07_check
```

批量窗口记录默认选择环境的可视化、全部环境必要统计；避免逐帧同步全部GPU张量到CPU。吞吐量作为实测报告，不预设脱离硬件的性能承诺。

## 用户操作

- 先board做推板、lean/release，再contact做拉动，不只看机器人站着。
- 选环境7施力，切到环境8确认没有同样外力。
- 暂停批量、记录环境8状态，单独reset环境7；环境8的状态/历史不能被清空。恢复后其他环境正常推进不算污染。
- 选环境7/8/0检查位置：viewer可自动平移镜头，但诊断明确世界/环境局部坐标。
- policy实时推理，观察非零姿态/运动下观测变化。

## 自动关卡

- N=64；随机选多个非连续env_ids做reset/set_state，验证其他环境在不推进物理期间保持不变。
- 单独终止一个环境，不能让整个batch都terminated。
- 随机化恢复nominal，100次reset不发生累计漂移；固定seed与env划分可复现。
- action/obs对oracle回归、history、子步PD全部通过；不能只复查形状。
- 对照MuJoCo的滑行、回正、摩擦微基准，比较预先定义的统计/物理指标；相近不等于长时轨迹逐位一致。
- 检查07时必须真实执行mjlab，不能调用硬编码MuJoCo实验后贴mjlab标签。
- 接触显示、外力、回放、窗口关闭均真实实现，不允许NotImplementedError留在交付路径。

出现差异先依次查模型参数、state indexing、控制时序、接触设置与求解器。修共享逻辑后复测MuJoCo。用户确认后进入08。

参考：[mjlab代码和官方示例](https://github.com/mujocolab/mjlab)、[架构](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)、[MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/)。

