# 三任务实时状态与采集画面

入口 `tools/deployment/three_task_v27.py` 委托 V26 原执行器，仅增加独立只读 HTTP 页面，不改图、模型、判断或物理。页面读实际公开相机 RGB 文件和节点状态；不读取 STATE_TRACE、接触真值或私有评估。没有额外相机采集。每0.5秒检查，图像只在执行器采集相机时更新，长段 Code 动作可能停留在上一帧，页面明确显示年龄；这不是连续视频。

```bash
/home/zheng/miniforge3/envs/geometry/bin/python /home/zheng/github/Graph-Self-Adapter/tools/deployment/three_task_v27.py --task collect_apple --backend sim
```

| task | 页面 |
|---|---|
| put_mug_on_box | http://127.0.0.1:18734/ |
| collect_apple | http://127.0.0.1:18735/ |
| collect_cubes | http://127.0.0.1:18736/ |

方块默认仍禁用：仅 `--check` 或明确 `--experimental` 验证可用，页面接入不赋予任务通过资格。`--check`、`--prepare-only` 和 real 离线检查不启动页面或回合。端口冲突用 `--live-port`，不重复启动服务。任务结束页面服务关闭。首次定位及 UNKNOWN 仍可能等待实际 Agent。

已在运行的任务可单独挂只读页面，不重新执行：

```bash
python tools/deployment/three_task_v27.py --task collect_apple --view-only --run-output /absolute/current/run
```

view-only 用 Ctrl+C 关闭；不停止任务。外部访问使用相同端口 SSH 转发，例如 `ssh -N -L 18735:127.0.0.1:18735 zheng@100.68.228.21`。

画面上方覆盖当前状态、累计运行时长、当前节点／阶段耗时及画面年龄。累计时间从保留的首张初始RGB开始，以封存时刻停止；没有首帧时暂从查看器启动计时。仅近期存在动作下发记录才标明正在执行节点，等待、未知、初始化与结束分别显示。指令记录不等同于确认机器人仍在运动。

自动 API 定位／阶段判断使用 `tools/deployment/three_task_v28.py`，其余 task/backend 和页面端口相同。模型 `openai/gpt-6-astra`，密钥只从本机独立文件读取。首次定位和阶段判断各请求45秒上限；实际 API 身份记录与聊天本体分别归因，失败／UNKNOWN 停止，新版整图验收状态参见 `evaluations/api_grounding_v28/STATUS.json`。

苹果任务当前已验收自动入口是 `tools/deployment/three_task_v32.py --task collect_apple --backend sim`，同一解释器及页面18735。首次定位、Lift确认和最终Complete都由Astra API，不需聊天人工回写。节点图将闭合尝试与原有界Lift作为获取持物证据的局部过程，确认前不搬运。验收来源 `evaluations/api_grounding_v32/ROOT_ACCEPTANCE.json`；仅一例名义模拟整图通过，其他任务尚未按此版本完整复验。

连续画面使用 `tools/deployment/three_task_v33.py --task collect_apple --backend sim`。页面地址不变。独立 RGB 显示渲染目标10fps，MJPEG持续传输，不再仅显示控制器保存的采样帧。任务结束保留查看服务与最后画面，画面年龄继续增加、任务时长停止；Ctrl+C关闭服务后可同端口重跑。控制和API采集相机不被此线程调用。`--check`仍无回合、无相机／设备连接。

V33复跑发现辅助table_patch误触发全角色漂移门。苹果的 V34 将背景ROI与SAM实体分开，仍严格检查苹果／碗／本体。V35入口为 `tools/deployment/three_task_v35.py --task collect_apple --backend sim`，连续画面和端口不变，失败会标明已完成节点和停止原因，程序退出不等于任务完成。其他任务不继承苹果整图资格。

当前按用户范围收敛为单红杯：`tools/deployment/three_task_v44.py --task put_red_mug_on_box --backend sim`，底层标准场景仍是 put_mug_on_box，但图只有红杯17节点，白杯仅静态背景。单红杯的权威成绩是新运行 RED_ONLY_RESULT.json 与 `evaluations/red_only_v44/ROOT_ACCEPTANCE.json`；保留原双杯 evaluator 的 RESULT.json，不用其 strict_success 判定单红杯，也不将单红杯成功计为双杯成功。页面仍使用18734。

新增已验收简化任务：`task_auto_v48.py --task put_black_spoon_on_plate --backend sim`，页面18743。只放黑勺，8节点，不宣称双勺完整分类。每次手动启动生成新的工程运行范围与目录，原已消费scope不重置。单方块入碗当前运输掉落，取刀尚未通过，不作为已验收入口。所有任务同样支持 `--backend real --check` 零设备离线接口验收，尚不执行硬件。
