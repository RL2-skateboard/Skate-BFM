# 阶段02：滑板、机器人来源与障碍资产

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段01已 ACCEPTED。主开发窗口：MuJoCo；mjlab/Isaac 的实际资产移植分别在07/08验收。

## 目标与可见成果

显示可拆解检查的滑板：板面、前后转向架、四轮、关节轴和碰撞体。提供 standard、cruiser、longboard 三种几何配置，并在独立 gallery 中展示坡道、低台、窄杆等资产；默认正式环境依旧只有平地，不自动加障碍。

机器人本阶段只选定 BFM 29DoF 资产来源、许可与 checkpoint 对应关系，不提前宣称机器人控制完成。

## 建模选择

优先参数化生成滑板刚体与低复杂度碰撞体，既便于跨后端导出，也便于批量扫描质量、轴距、轮径。若采用现成 mesh，先查许可、比例和可用碰撞代理，mesh 漂亮不等于动力学可用。

配置至少包括：板长/宽/厚、wheelbase、track、wheel radius/width、各刚体质量与惯量、truck 轴线/安装角、关节限位、bushing 参数。禁止三样式仅改颜色或名称。

可用工程起始值：standard 板长0.80 m/宽0.20 m、cruiser 0.72/0.22、longboard 1.00/0.24；轮径、轴距和质量另列合理初值。这些是暂定仿真设计值，不是已测真实滑板参数；如有目标实物则替换成其测量值并注明来源。

## 实现步骤

1. 单位统一 SI；模型导出显式声明角度约定。内部角度用 rad，UI可显示 degree，但转换只做一次。
2. 模型生成器与配置共用唯一参数源；build/play/check 必须读取同一配置/hash。
3. 四轮真实旋转轴与圆柱长轴一致；显示二者的调试箭头。不允许在 radian 模型中写 euler=90。
4. 前后 truck 连接确实位于 deck—truck—wheel 的载荷路径。约化模型的 tilt 不能是没有承载几何的孤立空子体。
5. 显式定义正质量、正定惯量、质心与碰撞代理；避免从高分辨率装饰 mesh 推断全部动力学。
6. 碰撞分组屏蔽结构内部不应接触的相邻零件，但不关闭必要的脚板、轮地碰撞。不要用滤碰撞掩盖错误尺寸。
7. 定义 named anchors：deck top、轮心、truck pivot、机器人候选放脚点。
8. 障碍用有限实体几何；低摩擦区域采用可验证的有限接触区域方案，不能用无限 plane 冒充局部小块地面。
9. 给每个资产记录 source、license、units、version、生成参数、碰撞代理方式和质量来源。

## 测试命令

```bash
python -m skate_sim build --asset board --style all
python play.py --backend mujoco --scene board --style standard --inspect --record runs/02_standard
python play.py --backend mujoco --scene board --style cruiser --inspect
python play.py --backend mujoco --scene board --style longboard --inspect
python play.py --backend mujoco --scene gallery --inspect --record runs/02_gallery
python -m skate_sim check --stage 02 --backend mujoco --output runs/02_check
```

inspect 模式默认暂停，可切正常动态；关节扫描必须标“检视驱动”，退出时清除。不能把检视驱动留到自由滑板实验。

## 用户窗口验收

- 切三样式，看尺寸标尺和参数面板，确认不同轴距/轮径/板长在几何与碰撞体中都生效。
- 显示半透明碰撞体，检查轮是否穿板、车桥是否接错、轮轴是否歪斜。
- 逐个低速旋转四轮，确认绕横向轮轴旋转而不是摆动。
- 展示 front/rear truck 铰轴与 tilt 载荷路径；此时不验收完整转弯动力学。
- 在 gallery 逐件查看实际碰撞外形。正式 board 场景中确认没有意外障碍。

## 自动验证与通过条件

- 三样式的解析配置和实际模型量测一致（误差限以导出精度为准），不是只比较配置字符串。
- 轮轴与几何轴夹角默认门槛小于0.1°；质量为正，惯量正定且符合刚体惯量三角不等式。
- 所有关节/刚体名称唯一，关节数与拓扑符合各模型定义。
- 初始化无明显穿透：允许求解器规定的接触裕量，不允许用任意大容差隐藏错位。
- 同一配置重复 build 的语义参数一致；mesh 路径/LFS 文件真实存在。
- 形成资产总览截图与一段样式/关节演示；用户确认后进入03。

参考：[HUSKY 作者资产](https://github.com/TeleHuman/humanoid_skateboarding)、[BFM-Zero 资产](https://github.com/LeCAR-Lab/BFM-Zero)、[MuJoCo XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html)。借鉴出处进入来源清单，公开模块不保留旧名称。
