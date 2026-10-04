# 当前工作

Graph / stage-local Fast Adapter 已从 Interaction-RSI 剥离到本仓库。后续实现、验证和任务版本在这里维护；旧仓库只保留历史路线及既有封存材料。

当前可复用方案是 [P70 R3 book_to_caddy](../task_packages/book_to_caddy_r3/README.md)。任务图、阶段配置、五份任务权重、必要的一条演示和任务参考完整保存，执行框架在本仓库；标准模拟器和公共视觉模型作为外部依赖。历史验收没有被迁移改写。

本次迁移状态与检查证据由 [migration](../migration/)拥有，Dashboard 从 [STATUS.json](../dashboard/STATUS.json)生成。迁移验证只检查封装、模型和源码独立性，不产生新物理 rollout 或新成功率。

查看 [完整注释视频](../media/book_to_caddy_r3_full_task.mp4)、[P70 科学报告](../evaluations/outer_rsi_p70_object_graph_v01/ROOT_REPORT_ZH.md)及 [原工程验收](../task_packages/book_to_caddy_r3/ENGINEERING_ACCEPTANCE.json)。旧报告内未纳入本项目的大型原始帧／训练数据仍保存在 Interaction-RSI 历史目录，不能假设本项目是其全量备份。

下一次研究修改须先阅读 [spec](../spec.md)，围绕真实阶段入口与位置／尺寸变化提出有界实验；本次仓库整理没有自动开启这些实验。
