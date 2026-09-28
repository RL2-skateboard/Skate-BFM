# skate-sim：公共执行规范

版本：2026-09-26 / v3。适用于阶段01—10；本文件是工程和验收约束，不是额外开发阶段。

## 1. 名称、位置与职责

- 项目名、发行包名、README主标题、窗口主标题统一 `skate-sim`；Python导入包名为 `skate_sim`。
- conda环境名固定 `skate-sim`。不使用bfm、husky或其他项目名作本项目名称/前缀。BFM-Zero/HUSKY等仅保留必要技术说明、文献链接和第三方版权。
- 工作目录是用户现有 `workspace/skate-sim`，其中 `plan/` 是本套计划。先解析实际绝对路径，再执行文件操作；不要假定它等于本助手的沙盒或根目录 `/workspace/skate-sim`。
- 用户已清空半成品：直接从空项目建立，不创建旧项目分支/worktree、不恢复已删除代码、不要求迁移/合并旧sim分支。最新目录要求覆盖旧的skate-bfm分支建议。
- 本助手负责研究比较、技术选择和计划；OpenCode负责安装依赖、生成资产、代码实现、整理、测试、演示和持续修复；用户负责阶段窗口验收。
- 每次按当前获准阶段执行。用户将计划放好后由OpenCode从01开始，00只作计划说明，不包含安装/执行脚本或要求一次实现全部阶段的提示词。

## 2. 范围与目录

只保留仿真环境、资产、配置、可视化、交互、微基准、必要回归、离线小型辨识和测试checkpoint所需最小推理。不引入项目级训练代码、真机驱动、ROS/SDK部署、采集平台、无关数据集或权重。引擎自身官方依赖可以包含训练相关库；不为精简破坏官方依赖，但不把其训练工程复制进项目。

项目根目录：`play.py`、`README.md`、`pyproject.toml`、`environment.yml`、`plan/`、`src/skate_sim/`、`assets/`、`configs/`、`tests/`、`docs/`。本套计划仅在 `plan/`，不另复制到docs/plans。

建议模块：
- `src/skate_sim/__main__.py` 与 `cli.py`：非可视化工具。
- `viewer.py`、`record.py`、`config.py`：共用可视化/事件/记录/配置。
- `backends/{base,mujoco,mjlab,isaacsim}.py`：后端wrapper及生命周期。
- `models/{board,robot,scene}.py`：参数化资产与场景。
- `control/{actions,observations,policy}.py`：控制与基座模型最小适配。
- `physics/{contact,randomize,calibrate}.py`：确实需要时再创建。
- 物理检查保留在一个共用checks模块组或tests中，不同时实现两份相同检查。

资产归类robot/board/terrain；运行记录在不入git的 `runs/`，权重在外部路径或不入git的 `checkpoints/`，依赖只读资源在安装位置/缓存中。第三方源码仅在必要时最小提取并保留许可，不把完整训练仓库带进来。

文件命名尽量两个英文单词以内。一个功能一份实现；共享动力学与控制不得在play、check和各场景重复定义。只保留覆盖真实风险的简短测试，不留下临时调试/多版play/重复生成器。不以“简洁”为由删掉验证关键物理语义的回归。

## 3. 唯一可视化入口：play.py

所有阶段、所有后端、所有实时交互、资产检视、接触实验、参数对比与可视化回放都由根目录 `play.py` 启动。不得要求用户直接运行mjlab原生play、Isaac独立示例、另一份viewer.py或单独的replay.py。内部可以复用原生viewer和启动器；浏览器型viewer也由play.py启动服务并给出地址，不要求第二个启动脚本。

`play.py` 只做参数解析/后端懒加载/调用共享执行逻辑，不能把全部项目堆进一个长脚本。Isaac需先初始化应用再导入特定模块，未选中后端不导入它。三个后端同处conda环境，但每次窗口使用独立进程，避免运行时污染。

目标命令（不是空目录当前已有能力）：

```bash
conda activate skate-sim
python play.py --backend mujoco --scene board
python play.py --backend mjlab --scene board
python play.py --backend isaacsim --scene board
python play.py --backend mujoco --scene rider --controller policy --checkpoint /absolute/path/model.pt
python play.py --replay runs/example --mode visual
python play.py --replay runs/example --mode dynamics --backend mujoco
```

- `--backend`：mujoco/mjlab/isaacsim，默认mujoco。
- `--scene`：bench/board/contact/robot/rider/gallery，逐阶段实现，未实现时明确报错；默认board在02后可用。
- `--controller`：none/pose/joint/policy；policy读取checkpoint，项目名称不随模型变化。
- `--replay`配合`--mode visual|dynamics`；回放后端/场景默认读取记录，显式指定冲突时拒绝并说明。
- 各阶段需要的style/model/experiment/inspect/preset/num-envs/view-env/params/randomize/seed/record参数实际接入，不保留空开关。
- `--help`在没有任何引擎或GPU时仍能显示，不能先导入Isaac再打印帮助。

非可视化工具用 `python -m skate_sim doctor|build|check|report|calibrate`。它们可以生成数字、文件和静态报告，但凡要求用户看场景/曲线或操作窗口，仍使用 `play.py`。如校准曲线需要交互查看，在play.py对应experiment面板或回放叠加中显示；不另造仪表盘入口。

`--backend all`仅供doctor/check依次派生同一环境中的后端子进程，不作为三个GUI混开功能。所有工具支持help；缺后端、缺文件、未知选项都给明确提示与非零退出。check：0=所请求必需项全部通过；1=失败；2=排查后仍阻塞；3=未完成，不把SKIP/BLOCKED算成功。

交付每阶段时，OpenCode需将模板中的工作目录/权重路径换成实际值，给完整可复制命令、预期现象、输出位置和运行结果。窗口命令必须实测，不能把文档写好等同功能完成。

## 4. 缺依赖时安装并迭代

普通软件/依赖缺失已经属于执行任务：检查→选择同环境兼容版本→安装→pip check/导入→真实应用启动→play.py集成验证。阶段01负责conda、MuJoCo、mjlab、Isaac Sim及必要Isaac Lab；后续阶段负责修复新引入/损坏的依赖。不是发现未安装就交给用户自行解决。

先查硬件/OS/驱动/GLIBC/磁盘/显示和官方兼容表；不要盲目混装最新版本。默认只用conda `skate-sim`，不默认新增环境、不改系统Python、不改其他项目环境。版本发生冲突时先求解兼容版本并验证；确实无法满足同环境要求时提交准确冲突证据与最小调整建议，不偷偷改名或丢弃后端。

优先用户级安装。没有conda时按官方流程安装用户级发行版；可自动完成的下载、包安装、缓存预热和显示配置不反复索要确认。需要管理员驱动变更、硬件替换、许可交互或访问权限时，说明具体限制并走实际可用授权流程；用户的软件安装要求不等于有权改写任意系统或绕过许可。

安装完成不等于wrapper完成：须由play.py真实运行。保留版本/命令/错误摘要，失败时继续定位修复，不能用模拟输出替代。OpenCode不为每项安装生成独立临时脚本，最终保留简洁environment.yml、必要版本约束及README复现命令。

## 5. 固定交互语义


| 操作 | 行为 |
| --- | --- |
| Space | 暂停/继续；暂停不推进物理、策略、history |
| N | 暂停下推进一个物理子步，不偷偷更新一次策略 |
| M | 推进一个完整控制步，包含规定子步数 |
| R | 重置当前环境，并同步刷新观测、延迟、历史和窗口 |
| 鼠标选中＋面板施力 | 对指定刚体施加世界/局部坐标下明确的有限时长外力 |
| 面板“释放” | 清除鼠标弹簧、外力与辅助约束 |
| C / V | 切接触力/碰撞体显示；若快捷键冲突须在面板显示实际映射 |
| 面板记录/停止 | 写入可回放状态、动作、事件、配置，明确录屏开关 |
| 面板环境ID | 批量后端只选中/操作该环境；显示 local/world frame |

优先复用原生 viewer。某后端不支持原生鼠标拖拽时，必须提供同窗或相邻控制面板的选择刚体＋数值力/力矩/脉冲输入，不能只留键盘 TODO。不强求三个 viewer 相同 UI，但要求相同物理语义。拖拽/辅助夹具始终醒目标识“非自然外力”，释放后才能评价自由动力学。

## 6. 阶段完成状态与迭代协议


只允许以下状态：
- IN_PROGRESS：正在实现或修复。
- AUTO_PASS：自动指标已通过，尚未完成窗口人工验收。
- WAITING_REVIEW：已交付窗口入口、证据和说明，等待用户确认。
- FAIL：已复现不合格，必须继续修复当前阶段。
- BLOCKED：已排查仍无法自行解决的硬件、依赖冲突、checkpoint访问、授权或输入缺口。软件尚未安装本身不构成停止理由；写明已尝试什么及需要谁提供什么。
- ACCEPTED：自动验证与用户窗口验收都通过，记录对应 commit。

OpenCode 不得自行把 WAITING_REVIEW 改为 ACCEPTED。用户已明确确认相应可见行为时可记录确认。一次交付只推进到当前允许关卡，不能提前生成大量未验证的后续模块。

每次迭代在同一份阶段记录追加：问题/触发命令/seed/截图时间点/原因/修复 commit/复测结果。修模型后重跑受影响的上游回归；修共享 action/obs 后三个后端都复测。不得以“先跳过”“以后补上”关闭必需项。

计划中的数值门槛是待阶段开工时冻结的工程起点，不是测量事实。OpenCode须在正式测试前登记指标、单位、绝对/相对误差算法、样本与阈值。测试失败后不得单纯放宽阈值获取通过；有物理理由需要调整时，向用户说明并取得确认，再重跑相关验收。

遇到阻塞，不静默降级成别的后端或虚拟数据。可继续当前阶段不受影响的诊断，但不得宣布该阶段或总交付通过。

## 7. 总交付边界


必须交付三后端的可交互板单体和板＋机器人、29DoF BFM 数据链路、多个板样式、独立可展示的障碍资产、可用的记录回放、工程级参数与随机化机制。真实 BFM checkpoint 必須实际运行，不能只用零动作冒充模型兼容。

“不承诺未经训练的 BFM 自动掌握滑板技能”：环境适配与策略能力分开验收；失败/摔倒可用于检验碰撞和终止，但不得用它冒充稳定骑行演示。稳定演示可用明确标注的姿态控制/辅助夹具；正式物理验收必须去除辅助。要交付自主稳定骑行须另有任务策略或明确扩展训练范围。

没有真实测量时只能交付 sim2real-ready 的工程环境，不宣称已经消除或量化真实 gap。阶段09的原生接触、参数识别接口、随机化与合成验证仍必做；神经接触和真实校准有数据门槛，不作为虚假必达指标。

## 8. 变更原则

本套v3的目录、名称、play.py及conda约定覆盖上一版；原始29DoF/action适配、三wrapper、多板样式、被动动力学、脚板摩擦、障碍资产、sim2real方法研究、README与代码整洁要求保持有效。不得为了满足单入口而复制物理逻辑，也不得为了项目独立命名而删除基座适配或引用来源。
