# 阶段05：29DoF G1 与小脑基座最小推理适配

共同约定：项目名与 conda 环境名均为 `skate-sim`；工作目录为用户的 `workspace/skate-sim`，计划位于 `plan/`。先读 [计划总览](00_overview.md) 与 [执行规范](12_execution.md)。下列命令均在项目根目录执行，先 `conda activate skate-sim`；所有窗口、交互及可视化回放只从根目录 `play.py` 启动。

前置：阶段04已 ACCEPTED；已取得用户指定或确认的测试 checkpoint 及对应配置。先地面，不上板。

## 可见成果

窗口中有完整29DoF机器人，可在安全限幅/固定底座检视模式逐关节测试；解除检视辅助后进行地面姿态控制和真实 BFM 推理。显示关节名、目标/实际位置、速度、力矩、模型观测摘要与控制模式。

不拷贝全部训练仓库。只引入运行 checkpoint 所需的网络定义、normalizer、配置解析、推理入口及许可证。配置必须与权重相符，不能只凭“G1”名称选模型。

## 必须锁定的合同

写 configs/robot.yaml 与 docs/robot.md，包含：
- checkpoint 路径/hash、来源、模型结构、命令/latent 的合法构造方式。
- 29关节的官方顺序与每后端名称映射，腕部6DoF不裁掉、不补零代替。
- 默认姿态、action scale/clip、PD gain、effort/velocity/position limits。
- physics_dt、control_dt、decimation、延迟和滤波；数值取自 checkpoint 对应正式配置。
- body顺序、四元数约定、world/body/heading frame、obs归一化与 history/reset 语义。
- 参考实现 commit、所复用的文件/函数与最小依赖清单。

原审查参考配置的尺寸为 actor 93、history 372、privileged 463；它们仅是该配置的检查线索，必须以本次选定 checkpoint 对应源码确认，禁止硬编码尺寸就算适配完成。

## 实现与对照验证

1. 建立以名称为依据的 action 映射，一关节一关节验证正方向。
2. 保持控制目标于控制周期内，**每个物理子步**按最新q/dq计算PD力矩，再执行限幅。不能重复施加控制周期开始时算出的一份旧力矩。
3. actor 重力投影和角速度来自实时刚体状态；坐标变换与官方一致，不得填固定重力/全零角速度。
4. history 按官方字段排序、字段内时间顺序、padding/reset 规则拼接。若官方为 field-major 且 newest-first，就按此实现，不能用 timestep-major 替换。
5. privileged 特征严格对照官方的 heading-normalized 位置、姿态、速度和扩展head计算；head偏移必须在正确的局部坐标下变换。
6. 建立独立参考 oracle：调用固定版本官方函数或离线生成黄金样本，与本项目输出逐元素比较。不能使用被测函数本身同时生成expected。
7. 用倾斜/旋转/速度非零的人工状态与至少100个合法状态进行测试，避免全零状态隐藏坐标错误。
8. reset 清空指定环境的history、last action、延迟队列、滤波状态并返回新obs。
9. 实际模型持续闭环至少30 s，记录有效输入、动作变化、推理时间、力矩饱和；不是最后只调用一次模型。

## 测试命令

```bash
python -m skate_sim doctor --checkpoint /absolute/path/model.pt
python play.py --backend mujoco --scene robot --controller joint --inspect --record runs/05_joints
python play.py --backend mujoco --scene robot --controller pose --record runs/05_pose
python play.py --backend mujoco --scene robot --controller policy --checkpoint /absolute/path/model.pt --record runs/05_policy
python -m skate_sim check --stage 05 --backend mujoco --checkpoint /absolute/path/model.pt --output runs/05_check
```

checkpoint 参数为示例占位；交付时须提供实际绝对路径或明确下载/配置步骤，不能将不可执行占位命令算已验证。

## 用户窗口验收

- 依次点选29个关节，小幅改变目标；显示同名关节响应，腕部也逐一测试。
- 手动改变/施力使机器人倾斜，观察重力投影和角速度确实变化。
- 切 pose 与 policy，角标、动作来源、配置/hash同步变化，不能两个选项都执行零动作。
- 暂停并单步，检查策略只在控制边界更新、物理子步照常更新力矩。
- 重置后显示初始姿态及新观测，没有上一episode残留。
- 如姿态控制不稳定，区分无辅助正常落地、固定基座检视、辅助演示，不能隐藏约束。

## 通过与返工标准

- 29关节顺序、scale、限幅全部测试；oracle误差按dtype设门槛，浮点32初始可取abs/rel 1e-5/1e-4，并记录实际误差。
- 合法四元数、history顺序、head旋转及heading处理均有针对性的非零样本。
- 子步PD测试施加变化状态，确认tau随q/dq变化。
- checkpoint实际持续运行，有完整动作与obs记录；缺checkpoint则 BLOCKED，不能用dummy完成BFM验收。
- BFM在新场景跌倒不自动证明接口错误；但若官方基准地面任务也明显异常，先复核oracle、配置与命令输入，不能进入上板阶段掩盖。

用户确认地面可视化与BFM链路后进入06。

参考：[BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero)、[官方权重](https://huggingface.co/LeCAR-Lab/BFM-Zero)。固定版本历史处理、控制和观测参考路径见 [资料清单](11_references.md)。

