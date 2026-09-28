# 阶段10：全量回归、干净复现与最终签收

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段01—09全部 ACCEPTED。此阶段不再用新大功能替代收尾；发现缺陷回到责任阶段修复并回归。

## 交付目标

用户拿到 `skate-sim` 完整项目后，按README在声明硬件上安装，能执行三后端的滑板单体和机器人＋滑板窗口命令，完成交互、重置、记录、回放和物理检查。代码整洁、功能真实、限制明确，没有必需路径中的占位实现。

## 交付验收矩阵

| 必需项 | MuJoCo | mjlab | Isaac Sim |
| --- | --- | --- | --- |
| board三样式/被动滑行/转向/回正 | 必测 | 必测 | 必测 |
| contact拉动/卸载与材料有效性 | 必测 | 必测 | 必测 |
| G1 29DoF与BFM action/obs oracle | 必测 | 必测 | 必测 |
| rider姿态/实际checkpoint持续推理 | 必测 | 必测 | 必测 |
| 窗口暂停/步进/施力/重置/诊断 | 必测 | 必测 | 必测 |
| 记录/视觉回放/动力学重放说明 | 必测 | 必测 | 必测 |
| partial reset与逐环境外力 | N=1语义 | GPU N=64 | 目标N=16 |
| 随机化nominal恢复与参数响应 | 必测 | 必测 | 必测 |
| gallery障碍展示且默认场景不加载 | 必测 | 必测 | 必测 |

工程校准管线与方法报告必需；真实参数精度取决于外部真实数据，不能与三后端代码完整性混为一谈。没有真实数据必须清楚披露，但不是允许后端缺实现的理由。

## 清理与可复现

1. 从干净项目副本按README安装，不依赖开发者个人路径、缓存mesh或未提交文件。
2. 依赖锁定，conda环境统一为 `skate-sim`，项目名为 `skate-sim`，所有窗口入口统一为根目录 `play.py`；记录目标OS/GPU/驱动及实测兼容版本。验证缺依赖时的安装/修复流程。
3. checkpoint不入git；给来源、实际测试hash、配置与放置方式。若下载需许可，明确由用户取得，不能绕过访问限制。
4. 项目从空目录建立，只交付仿真、资产、最小推理与必要验证代码；不引入无关训练、硬件、部署代码。保留 `plan/` 文档、README和许可证；清理仅针对本项目生成的临时产物，不删除用户数据。
5. 名称检查：项目/发行包/窗口标题为 `skate-sim`，Python包为 `skate_sim`；无旧项目名前缀。BFM-Zero、HUSKY仅用于必要技术适配说明、参考链接与合法来源标注。
6. 只保留统一 `play.py` 可视化入口、内部工具、必要生成器和持久回归，不保留一次性debug/重复play/旧副本。
7. 检查所有README命令实际运行；按钮、开关、配置无“有入口没功能”情况。
8. 资产、代码、论文来源有链接/版本/许可；自建滑板注明生成参数，不冒充实测资产。
9. 同一代码文件内没有重复定义覆盖；无未使用关键配置；无except吞错后继续PASS。
10. 所有必需check失败/阻塞时退出非零，记录矩阵不把SKIP当PASS。

## 用户最终命令清单

下面是目标形式。交付时用真实环境激活方式、有效checkpoint路径替换占位，并逐条附实测结果。

```bash
conda activate skate-sim
python -m skate_sim doctor --backend all --display --checkpoint /absolute/path/model.pt
python play.py --backend mujoco --scene board --style standard --record runs/final_mujoco_board
python play.py --backend mujoco --scene rider --controller policy --checkpoint /absolute/path/model.pt --record runs/final_mujoco_rider
python play.py --backend mjlab --scene board --num-envs 1 --record runs/final_mjlab_board
python play.py --backend mjlab --scene rider --controller policy --checkpoint /absolute/path/model.pt --num-envs 1 --record runs/final_mjlab_rider
python play.py --backend isaacsim --scene board --num-envs 1 --record runs/final_isaac_board
python play.py --backend isaacsim --scene rider --controller policy --checkpoint /absolute/path/model.pt --num-envs 1 --record runs/final_isaac_rider
python -m skate_sim check --stage all --backend all --checkpoint /absolute/path/model.pt --output runs/final_check
python -m skate_sim report --run runs/final_check --output docs/acceptance.md
```

all检查必须在 `skate-sim` 环境中逐后端启动独立进程，使用实际安装的后端；可以复用内部调度逻辑，不增加用户可视化入口。

## 最终人工验收

用户在每后端至少实际完成：
- 切换滑板样式，推板、加载倾斜、释放看回正。
- 打开机器人＋滑板，区分pose/policy/free/supported，推板或推人并看接触。
- 暂停、两种步进、重置、录制、回放。
- 打开contact场景看摩擦响应；打开gallery看障碍。
- 按README复现一个固定seed实验，观察与报告一致。

视频是证据，不替代可运行窗口；用户只确认某后端时不能代表另两个后端也已签收。若远程交互符合用户需求，应记录接入方式。

## 交付文件

- README：纯仿真范围、安装、六条核心窗口命令、交互表、场景/模式、测试、故障排查。
- docs/acceptance.md：各阶段commit、自动证据、人工确认、验收矩阵、最终平台、残余限制。
- docs/physics.md：模型选择、参数来源、接触/控制解释、跨引擎差异。
- docs/robot.md：checkpoint/hash、29DoF映射、action/obs合同与oracle证据。
- assets来源/许可证清单；生成器与最终可重现资产。
- 必要回归测试；示例小型记录或下载路径，避免将大视频/日志全部塞入git。
- `workspace/skate-sim` 完整项目文件、依赖锁定和配置；若项目采用git则记录验收commit。没有回旧仓库合并sim或恢复旧工程的步骤。

## 最终判据与返工

“已交付”必须同时满足：三个后端真实可运行、所有必需自动项通过、用户已确认各窗口效果、README干净复现成功、无必需项BLOCKED/NOT_RUN。

发现缺陷：定位责任阶段→修复→相关上游/下游回归→重新演示→更新确认记录。没有“最后一天先交半成品”的例外。资源或权限不足时诚实列阻塞与下一步，不能把目标悄悄改为仅MuJoCo。

真实gap尚未测量、无专用滑板策略等边界可以作为明确限制交付；它们不能掩盖错误物理、错误BFM观测、缺失Isaac实现或无效窗口交互。

