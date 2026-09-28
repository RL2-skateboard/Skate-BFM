# 阶段08：Isaac Sim 真实物理与交互交付

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段07已 ACCEPTED。Isaac Sim/Isaac Lab运行条件已在01检查，不能本阶段才发现无法安装。

## 可见成果

Isaac窗口中加载实际机器人与有物理碰撞/质量/关节的滑板。board/contact/robot/rider/gallery全部可展示；能推板、推人、查看接触、暂停重置，BFM持续推理。Isaac不是“安装了依赖就算实现”。

## 资产与物理任务

1. 用参数源生成/转换USD：真实mesh/primitive、碰撞shape、mass/inertia、rigid body、joint、material和物理scene。只写custom metadata的USDA绝不算资产。
2. 核对metersPerUnit、upAxis、四元数顺序、joint frame、关节轴、驱动与限位；必要时用官方USD检查工具验证。
3. 机器人资产必须匹配05锁定的BFM版本，不随便换成外观相同但惯量/限位不同的G1。
4. 确定滑板articulation/刚体组织，wheel关节被动，truck恢复力与阻尼明确。
5. 对约化模型的倾斜—转向耦合，不假设MuJoCo equality可原样转换。验证版本支持的约束/耦合方式；如不能支持，优先使用03已接受的显式truck模型，或经物理微基准验证的约束实现。不得用每帧写yaw位置冒充被动动力学。
6. 使用explicit effort控制或经过对齐的执行器方式；避免原生position drive与自定义PD重复施力。解释PhysX子步回调时机与有效力矩。
7. 材料静/动摩擦、combine mode、接触偏移、求解迭代与碰撞代理单独标定。MuJoCo condim/solref数字没有逐项等价的PhysX解释。
8. root/body查询依据实体与名称；reset、history、外力清理与mjlab同标准。
9. 共享skate_sim action/obs定义，独立写后端状态适配，保证world/body/heading frame一致。
10. 实现窗口施力UI或原生drag操作；若一个API不支持，提供真正可用替代入口，不留空按钮。

## 测试命令

在阶段01配置好的 `skate-sim` conda环境内运行；`play.py` 内部管理Isaac应用初始化，README不要求用户改用另一入口。若软件丢失或依赖损坏，先按01记录修复安装后继续。

```bash
python -m skate_sim build --asset board --format usd --style all
python play.py --backend isaacsim --scene board --experiment lean --record runs/08_board
python play.py --backend isaacsim --scene contact --experiment pull --record runs/08_contact
python play.py --backend isaacsim --scene robot --controller joint --inspect
python play.py --backend isaacsim --scene rider --controller policy --checkpoint /absolute/path/model.pt --num-envs 1 --record runs/08_rider
python play.py --backend isaacsim --scene gallery --inspect
python -m skate_sim check --stage 08 --backend isaacsim --num-envs 16 --checkpoint /absolute/path/model.pt --output runs/08_check
```

N=16是初始批量目标；硬件不足须记录并由用户确认调整，不得悄悄改成N=1还报告批量通过。

## 用户窗口验收

1. Stage面板/物理调试显示板、四轮、truck与G1实际刚体/关节。
2. 看USD碰撞体、轮轴、质量，与02参数表对照。
3. 推板与左右加载，观察真轮转动及自由转向，不仅动画关节。
4. foot fixture显示拉动滑移，修改材料确实改变接触行为。
5. 板上机器人推人/推板、暂停/单步/重置；控制器切换后数值与物理响应改变。
6. gallery可见障碍；默认rider无障碍。
7. 显示接触、记录回放功能真实有效。实际窗口录像对应命令、commit和参数。

## 自动验收与差异处理

- 在实际Isaac进程编译加载USD并执行step，不能只做文件存在/字符串检查。
- 实测29DoF映射、观测oracle、子步PD、state roundtrip、partial reset和持续checkpoint推理。
- 完整复跑滑行、转向、回正、脚板起滑等与MuJoCo同定义微基准。
- 不强求PhysX和MuJoCo长时轨迹逐点重合；采用固定初值和输入，比较物理量、趋势、范围，报告差异原因。
- 初始跨引擎工程门槛建议：被动滑行减速度、回正主频/衰减、起滑力等关键指标相对差不超过15%；接近零的指标改用预先定义绝对门槛。仅用于工程对齐，不等于真实误差15%。
- 若差异超限，先查单位/驱动/接触和模型拓扑，再调整后端求解器。若必须改模型，回到03/04重新确认，而非专门造一个“看起来相同”的动画。
- 缺实现/NotImplemented/强制BackendUnavailable均判FAIL或BLOCKED；绝不是依赖检查通过。

用户确认三后端都有完整实际能力后进入09。

参考：[Isaac Lab](https://github.com/isaac-sim/IsaacLab)、[Isaac Sim物理说明](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html)、[Isaac Lab执行器](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html)。

