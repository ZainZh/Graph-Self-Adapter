# 独立项目迁移

从 `/home/zheng/github/Interaction-RSI` 抽取 Graph 与 stage-local Fast Adapter，目标为 `/home/zheng/github/Graph-Self-Adapter` 和用户授权的 `ZainZh/Graph-Self-Adapter`。

[SOURCE_INVENTORY.json](SOURCE_INVENTORY.json)记录源码原始身份和有意调整的封装入口；[EVIDENCE_COPY.json](EVIDENCE_COPY.json)记录原样保留的历史报告与视频。`historical_engineering/` 是旧任务包工程验证原文，内部原路径和哈希未追改；它不是新项目的重新实验。

任务权重和冻结运行代码保持原始字节。新项目仅改变代码根目录、依赖绑定、独立说明和展示入口。数据集、公共模型及全量原始帧不在剥离范围内；历史实验驱动需要这些资料时，应明确另行取得，不回读旧项目的 prove/tools 源码。

[ACCEPTANCE.json](ACCEPTANCE.json) 是 Root 最终迁移验收，独立导入、真实模型加载及标准场景零积分编译细节见 [INDEPENDENCE_CHECK.json](INDEPENDENCE_CHECK.json)。五份任务权重与必要演示已上传 Git LFS。只做封装与代码检查，本次没有新物理 rollout。
