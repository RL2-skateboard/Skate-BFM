# 阶段06：MuJoCo 机器人＋滑板完整交互切片

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段05已 ACCEPTED。此阶段形成第一个完整后端交付，不等其余后端结束才让用户试玩。

## 可见成果

同一窗口呈现G1与滑板，显示脚板接触、足底相对滑移、重心、轮轴、板轨迹、控制模式和外力。用户可分别选板与机器人施力、暂停逐步检查、重置、录制并回放。

提供 board、robot、rider、contact、gallery 场景；正式 rider 默认平地，不含障碍。

## 实现任务

1. 通过命名实体拼装，不依赖根关节数组顺序。初始化板和机器人独立根位姿/速度，四元数归一化；脚底相对板顶位置由实际碰撞几何计算。
2. 检查初始脚底穿透/悬空、板与机器人自碰撞、左右足与板碰撞过滤。
3. 将阶段04的有效接触对接入真实机器人足底，重新测载荷与滑移；不能因脚的真实名字含ankle而漏掉材料赋值。
4. 控制模式分离：none（不施机器人驱动力矩）、pose（PD姿态）、policy（加载指定小脑基座checkpoint的真实推理）。动作/事件回放统一通过 `play.py --replay ... --mode dynamics`，不另设replay控制器入口。zero action 不等于 none，零动作可能对应默认姿态PD。
5. 场景preset分离：free（无辅助）、supported（明确的安全夹具演示）、fixture（隔离微基准）。窗口醒目标注，不允许录像裁掉标识。
6. 全局可见量与actor输入分离：board状态、接触诊断不未经模型许可塞进已训练BFM actor。
7. 定义跌倒/脱板/超范围等条件，区分terminated与时间上限truncated。检视模式允许不自动reset，展示失败原因与接触。
8. 实现真实记录与回放：状态用于视觉回放；动作、事件、初值用于动力学重放。记录版本和环境参数，不能只存board_speed。
9. 添加转向/推行/脚滑移场景脚本，重用03/04实验逻辑，不复制另一套动力学。

## 测试命令

```bash
python play.py --backend mujoco --scene rider --controller pose --preset supported --record runs/06_supported
python play.py --backend mujoco --scene rider --controller pose --preset free --record runs/06_free
python play.py --backend mujoco --scene rider --controller policy --preset free --checkpoint /absolute/path/model.pt --record runs/06_policy
python play.py --replay runs/06_free --mode visual
python play.py --replay runs/06_free --mode dynamics --backend mujoco
python -m skate_sim check --stage 06 --backend mujoco --checkpoint /absolute/path/model.pt --output runs/06_check
```

supported用于便于查看与操作，不计入无辅助动力学通过证据。若无任务策略导致policy不能稳定上板，应展示实际失败并记录策略边界，不虚构成功，也不把训练工程重新塞回本项目。

## 用户验收流程

1. supported中看脚与板接触，推板/推躯干分别响应，确认作用对象正确。
2. 释放辅助转为free或重新启动free，窗口清楚显示辅助已撤销。
3. 给板一个小脉冲，观察板、轮与脚相对运动；修改摩擦后对照滑移变化。
4. 推机器人看脚板载荷变化；若摔倒，接触/终止诊断指出实际原因。
5. 暂停、N/M单步、切接触和碰撞体；确认暂停不会积累历史或力作用时间。
6. R恢复整个场景及控制状态，反复10次不出现板飞走或机器人根位姿归零。
7. 回放刚才记录，核对受力时刻、动作与可见运动。视觉回放不可接受新的动力学交互，UI须说明。

## 自动与人工关卡

- board与robot的state roundtrip通过；根状态按名字查找。
- 无穿透爆炸/NaN；脚板材料配置对实际接触有效。
- 连续运行与重置压力测试至少10轮；在无任务策略情况下允许合理摔倒，但不允许数值爆炸被忽略。
- 用设计的跌倒状态测试terminated，而非只检查finite；时间上限独立测试。
- 同后端固定版本的动力学重放误差阈值预先登记；接触长期混沌不能要求逐位一致，优先比较短期状态和事件。
- 录屏＋可交互入口＋状态/动作记录齐全；用户确认此MuJoCo切片后才进入07。

此阶段交付的是可信环境与BFM接口，不保证未训练模型自主滑板技能。稳定骑行动画不能替代上述物理验收。
