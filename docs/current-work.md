# 当前工作

当前以 [剩余任务验收表](../evaluations/night_tasks_v49/index.html) 和其 [STATUS](../evaluations/night_tasks_v49/STATUS.json) 为准。本轮新增套杯、开抽屉、单方块入碗、单扫刷清扫、单黄方块双臂分类均完成连续整图、最终目标和安全检查；苹果、单红杯、单黑勺保留原验收入口。挂杯、刀具仍未验收。统一重新执行入口是 tools/deployment/run_task.py，每次生成新运行范围，支持 --list；本轮真实硬件未连接，只提供真实接口离线检查。视频为封存状态的带注释离线回放，明确区别于新推理和实时性能。

以下条目保留历史版本及当时计划，不代表当前交付状态。

历史修复：[双杯V23](../evaluations/cup_manual_stage_v23/STATUS.json)。按用户要求将全部简单Code节点一次规划、连续轨迹执行，并用桌面手动SAM输入替换Root初始定位文件桥；精确小技能保持原冻结合同。先零设备GUI/接口验收，再独立连续模拟整图，通过后部署新Clover入口。

历史任务交付：以 [V22逐任务状态](../evaluations/multitask_parallel_v22/STATUS.json) 为唯一新进度来源，为10个唯一刚体任务统一一键执行入口。普通搬运／关节机构／双臂工具三组并行，服务器四卡分配见 [Server计划](../evaluations/multitask_parallel_v22/SERVER_BATCH_PLAN.json)。Clover SSH 与硬件运动仍暂停；历史资格和当前完整SAM／Gemini验证分开。

历史判定自动化：`evaluations/gemini_graph_v19/` 已验收单条完整模拟与速度比较；控制源保持 V18，首次 grounding 仍由本体处理。

当前执行优化：`evaluations/stage_trajectory_v18/` 已验收名义整图；保留 V15 成功及 V16/V17 失败，分别由各版本 ROOT_ACCEPTANCE/STATUS 拥有。

2026-10-06：当前任务转为 [SAM 共享模拟／真实执行管线](../evaluations/sam_shared_pipeline_v01/SCOPE.json)。先用双杯放箱跑通一命令启动、SAM mask、连续整图与本体节点判定，模拟／真实共用图执行器，仅换机器人和相机 IO。当前真机尝试已按用户要求停止，原控制服务保留；真实接入检查和失败日志仍在原证据目录，不作为完整任务成功。原五条失败保留；V06/V07 未解决白杯持续握持；局部长期持物验证的失败保留；V14 两条结果保留；当前直接放箱图修订与整图验收只维护 [V15 状态](../evaluations/sam_direct_place_v15/STATUS.json)。

2026-10-06：正在完成 [Clover 真机接入检查](../evaluations/xtrainer_real_check_v01/STATUS.json)。用户拔插相机后，真实 RGB-D／SAM、双臂反馈和原生 FK 的无动作 shadow 已通过；从实测关节姿态到红杯相对抓取入口的只读 IK 候选收敛。该候选使用已保存图像，不授权实时运动；正在检查 Code 接近与冻结短技能之间的衔接。原 28 阶段图和权重保持不变；后续真实尝试只完成了少量运动，尚未完成整图，当前已停止并转到上方共享模拟管线。初始机械臂姿态无需与演示一致。

2026-10-06：用户已停止剩余仿真，当前转为 [Clover 推理笔记本部署](../evaluations/xtrainer_laptop_v01/SCOPE.json)，目标 `clover@100.126.63.65`，已完成代码／模型／依赖和一键操作入口的软件验收，见 [部署报告](../evaluations/xtrainer_laptop_v01/ROOT_REPORT_ZH.md)；未执行机械臂运动，下一步由操作者启动原控制服务并做真实 shadow。[部署状态](../evaluations/xtrainer_laptop_v01/STATUS.json) 记录实际连接与验收。[双杯部分运行及用户停止记录](../evaluations/agent_stage_v07_position_v01/USER_STOP.json) 保留，十条计划未完成，后续不继续启动剩余回合。

2026-10-06：已由用户停止 [双杯位置变化 Agent 闭环回归](../evaluations/agent_stage_v07_position_v01/SCOPE.json)，复用已经打开的十个变化初态，固定原图／小模型／fixed-z 和 v06 判定规则；先冻结，再由当前本体逐次看图判断并连续执行。没有重训或硬件任务。[实际状态](../evaluations/agent_stage_v07_position_v01/STATUS.json) 是唯一进度来源，旧 8/10 结果不能直接充当新版本成绩。

2026-10-05：已完成 [v06 连续 Agent 闭环验收](../evaluations/agent_stage_v06/ROOT_REPORT_ZH.md)。当前本体实际看图、回写并从原暂停状态恢复，双杯与红刀两项开发工程案例均完成原整图、原最终目标与安全检查。逐请求判断、UNKNOWN 和等待保持由 [封存报告](../evaluations/agent_stage_v06/FINAL_REPORT.json) 拥有；[注释视频](../evaluations/agent_stage_v06/VIDEO_MANIFEST.json) 离线重建，不能作为实时性能或独立成功率。

2026-10-05：Xtrainer 迁移实现与无设备连接验收见 [Root 接入报告](../evaluations/xtrainer_v01/ROOT_REPORT_ZH.md)、[当前状态](../evaluations/xtrainer_v01/STATUS.json) 和 [任务包入口](../task_packages/xtrainer_v01/README.md)。复用冻结对象中心图和 fixed-z 几何短技能，接实际控制器反馈、标定 FK、RGB-D/SAM 与本体语义判定；文件检查、严格限速 mock、真实录制图像检查分别记录，不互相替代。下一步是有范围的真机无动作 shadow，再检查短阶段；当前未连接相机或机器人、未发硬件运动，没有真机资格。以下为保留的历史节点，其“待接入”描述按当时范围解释。

2026-10-05：已完成 [Agent 语义阶段判定原型与本体判读](../evaluations/agent_stage_v05/ROOT_REPORT_ZH.md)。当前本体实际读取一组 6 帧 RGB 和允许的机器人反馈，识别可见脱爪但尚不能确认放稳，返回 UNKNOWN；[回复校验](../evaluations/agent_stage_v05/ROOT_REVIEW_RECEIPT.json) 保存实际理由及绑定。仅离线重建，无新物理回合或整任务成功；16 项接口测试通过。实际状态由 [v05 STATUS](../evaluations/agent_stage_v05/STATUS.json) 拥有。下一项需要另立连续执行范围，接阶段/异常事件的停等、取图、Agent回写与恢复；当前未接无人值守推理或硬件。

2026-10-05：阶段判定传感迁移已完成本轮原型实现与限定验证。[Sensor Transition v04 最终报告](../evaluations/twin_sensor_v04/ROOT_REPORT_ZH.md) 记录了杯类正向开发证据、刀具整图未完成及判定误拒绝的诊断；[逐回合统计](../evaluations/twin_sensor_v04/FINAL_REPORT.json) 与 [Root 验收](../evaluations/twin_sensor_v04/ROOT_ACCEPTANCE.json) 绑定实际来源。默认 fixed-z，Align 暂定放下许可与 Release 完成分开；[v03](../evaluations/twin_sensor_v03/STATUS.json) 与 [v04](../evaluations/twin_sensor_v04/STATUS.json) 均结束，所有历史版本保持封存。当前下一项是按 [接入指南](../evaluations/twin_sensor_v04/SENSOR_INTEGRATION_GUIDE.md) 接真实采样做无动作 shadow；硬件型号、实测夹爪标定和驱动待提供，本轮没有硬件命令。以下 v02 属已结束证据。

2026-10-05：[Twin RSI v02](../evaluations/twin_rsi_v02/SCOPE.json) 的开发、选版、独立完整任务检测、验收、复用包及注释视频已经完成。它是独立新实验，只研究取刀放菜板、双杯放箱和双臂方块分类；不是 v01 的第四轮。结果是混合证据：两项小样本外层改进、分类退化，当前几何在线 z 没有额外收益。实际数字和证据边界由 [最终报告](../evaluations/twin_rsi_v02/ROOT_REPORT_ZH.md)、[逐回合统计](../evaluations/twin_rsi_v02/FINAL_REPORT.json) 和 [状态](../evaluations/twin_rsi_v02/STATUS.json) 拥有。

[Root 验收](../evaluations/twin_rsi_v02/ROOT_ACCEPTANCE.json) 核对了实验来源与绑定。独立 seed `92005` 现已打开，今后只能作为开发／回归；不追加候选，不从测试结果重新选版。默认 fixed-z 在 online 开发和独立检测前承诺。

[三任务研究复用包](../task_packages/twin_rsi_v02/selected/README.md) 保存 A2 刀、A2 杯和 A3 分类的冻结图、训练模型、参考、源码和合同，无需重新训练或 Agent。分类包明确标记独立退化，不作为已验证改善推广。校验与无执行预览见 [PACKAGE_VERIFICATION](../evaluations/twin_rsi_v02/PACKAGE_VERIFICATION.json)。[三组对比视频](../evaluations/twin_rsi_v02/VIDEO_MANIFEST.json) 从封存 qpos 轨迹离线重建，展示两项改善和分类退化，不是新的算法回合。

迁移真机仍需真实 RGB-D／角色分割跟踪、本体时序、测得的夹爪开度标定、驱动与独立安全接口。纯传感阶段 judge 在 v03 实现并验证，可靠性以其报告为准。经 [真实输入复核](../evaluations/twin_sensor_v03/REAL_INPUT_AUDIT.json) 确认，本机已有 SAM2 代码与权重，但 Graph 尚未接入或测量延迟；现有真实演示的夹爪 slots 也不能当作实际开度。历史 v02 的 mask 与 Boolean supervisor 仍按原特权合同解释。本轮没有执行硬件命令。

v01 仍是冻结历史证据。杯类低层 release 曾读取连续物体 body distance 并改变 gripper，这是未声明的控制依赖，见 [加法勘误](../evaluations/twin_rsi_v01/INPUT_PERMISSION_ERRATUM.json)；v02 三配置共用移除该依赖的后端，不把共同修正计作候选收益。其余七项 v01、[历史十任务包](../task_packages/twin_rsi_v01/README.md)、[v01 注释视频](../media/twin_rsi_v01/FINAL_TEST_VIDEO_MANIFEST.json)、LIBERO book→caddy R3、P68 核心与迁移验收均保持原冻结。
