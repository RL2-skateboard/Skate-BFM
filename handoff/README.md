# Codex 对话迁移交接 · 2026-11-23

按原聊天拆成五份可独立阅读的中文上下文。每份包含共同约束，单独发给对应新聊天即可；不用把五份都发给每一个聊天。

- [迁移与协作总控](00-migration/AGENTS.md)：本轮保存、分支提交、环境与后续交接。
- [data prep](01-data-prep/AGENTS.md)：HUSKY离线参考、HOI重定向、共同simdata。
- [mimic trial](02-mimic-trial/AGENTS.md)：右手Robotiq夹爪、踩尾接板与放板，最后仍失败。
- [skate](03-skate/AGENTS.md)：原生29DoF仿真、训练接口、场景与回放。
- [确认数据集使用限制](04-dataset-and-world-model/AGENTS.md)：用途说明及可微世界模型，可选。

建议在新聊天上传对应文件并说：“请把这份文件作为项目交接上下文，先核对新机器的仓库与资产状态，说明已知目标、限制和下一步；不要自动启动旧实验。”文件也可全文粘贴。不要覆盖仓库原有根AGENTS.md；这些是迁移材料，不是完整聊天导入。

来源范围：现有根AGENTS、README、代码/文档，本轮实际Git与62项定向测试，以及四个原聊天最近可读片段（8/8/8/2个turn，尚有更早历史）。历史实验和论文描述保留来源层级，没有把原答复当作本轮新验证。旧PROJECT_CONTEXT/README中的23DoF在线路径已明确标为过时。

两代码快照：BFM train `ea4de54`，Sim sim `1a72110`，仅本地提交未push。本handoff目录为随后创建，未commit/push。模型、数据、日志、视频、原始报告和聊天附件不在本目录中；迁移时另行核对。没有复制密钥或账号身份文件。
