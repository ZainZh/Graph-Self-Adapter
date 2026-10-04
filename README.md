# Graph Self Adapter

以物体为中心拆解长程任务：简单阶段用 Code as Policy，困难而短的交互阶段使用任务训练的小策略与 Fast Adapter。Agent 根据真实执行证据低频修订 Task Graph；冻结执行时，只在指定局部阶段更新 latent `z`。

这是从 `Interaction-RSI` 剥离出的独立开发仓库。**后续 Task Graph、阶段控制器、小模型与 Adapter 的代码和实验均以本仓库为准。** 原仓库保留历史 Interaction Template 路线和封存记录。本项目保留 `prove` / `tools` Python 名称空间，以兼容已有 checkpoint 和冻结代码。

## 已保存的任务

[book_to_caddy_r3](task_packages/book_to_caddy_r3/README.md) 保存 LIBERO-10 的书本放入收纳盒后格方案：任务图、阶段门与路由、Code 配置、五份任务训练权重、必要的一条演示及任务参考。模型通过 Git LFS 上传；标准 LIBERO 场景、共享 DINOv3 / 预训练 LiLa 和 Python 环境另行安装。

从当前机器检查并复用：

```bash
cd /home/zheng/github/Graph-Self-Adapter
bash task_packages/book_to_caddy_r3/run.sh check
# 显式重新执行一个已开放历史条件；不重新训练
bash task_packages/book_to_caddy_r3/run.sh run --cell 7
```

每次执行写入新目录，不覆盖原始实验。`check` 和 `run --dry-run` 不执行物理任务；`run` 才启动模拟器。默认 cell 7 使用原 R3 初态与 seed。十个可复用条件均已开放，重放不计作新的独立测试。

## 下载与依赖

```bash
git lfs install
git clone https://github.com/ZainZh/Graph-Self-Adapter.git
cd Graph-Self-Adapter
git lfs pull
```

新项目代码不依赖旧仓库的 Python 源码。在当前工作站，外部模拟器、视觉基座和 Python 安装仍复用既有位置；换机器时按 [依赖配置](docs/dependencies.md)设置本机路径。任务权重已随仓库保存，公共视觉基座不重复上传。

## 入口

| 内容 | 入口 |
|---|---|
| 目标、学习边界与更新规则 | [spec.md](spec.md) |
| 两层自提升与当前实现 | [docs/architecture.md](docs/architecture.md) |
| 当前工作与验证 | [docs/current-work.md](docs/current-work.md) |
| 依赖配置、检查与测试 | [docs/development.md](docs/development.md) |
| 任务图和执行库 | [prove/task_graph](prove/task_graph/) |
| 表征、模型与局部适配库 | [prove/fast_adapter](prove/fast_adapter/) |
| 数据采集、训练、评估和视频工具 | [tools/task_graph](tools/task_graph/)、[tools/fast_adapter](tools/fast_adapter/) |
| 当前任务包 | [task_packages/book_to_caddy_r3](task_packages/book_to_caddy_r3/) |
| 历史科学报告 | [P70](evaluations/outer_rsi_p70_object_graph_v01/ROOT_REPORT_ZH.md)、[P68](evaluations/fast_adapter_p68_cross_task_reference_v01/ROOT_REPORT_ZH.md) |
| 带阶段注释的完整任务视频 | [media/book_to_caddy_r3_full_task.mp4](media/book_to_caddy_r3_full_task.mp4) |
| 新项目 Dashboard | [dashboard/index.html](dashboard/index.html) |
| 剥离与验收记录 | [migration](migration/) |

运行 Dashboard：

```bash
python3 tools/evidence/build_project_dashboard.py
python3 -m http.server 59162 --bind 127.0.0.1 --directory .
```

打开 <http://127.0.0.1:59162/dashboard/>。仅在查看时启动；端口占用时先检查已有服务。本命令不启动 GPU 或机器人任务。

历史实验工具也一并保留，部分需要原实验数据；复制源码并不代表所有历史训练材料都已发布。当前一键任务包已移除旧项目源码依赖。研究结论及限制以对应封存报告为准，本次仓库剥离没有产生新的成功率样本。
