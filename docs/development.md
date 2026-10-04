# 开发与验证

后续工作目录是 `/home/zheng/github/Graph-Self-Adapter`。保留 `prove` / `tools` 名称空间；在新仓库执行，避免混用旧项目的 PYTHONPATH 或 editable 安装。

## 先检查依赖和任务包

```bash
cd /home/zheng/github/Graph-Self-Adapter
bash task_packages/book_to_caddy_r3/run.sh check
bash task_packages/book_to_caddy_r3/run.sh run --dry-run --cell 7
```

这两条命令不执行物理动作。运行依赖路径见 [dependencies.md](dependencies.md)；本机配置放 `local/DEPENDENCIES.local.json`，不会上传 Git。

## 修改后的检查

使用已配置的 Python 环境，关闭无关 pytest 插件。核心阶段／接口测试不启动机器人：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_task_graph_core.py tests/test_precision_action_policy.py \
  tests/test_p52_stage_policy.py tests/test_p53_short_insert.py \
  tests/test_p55_relation_approach.py tests/test_p68_book.py \
  tests/test_p70_object_controller.py tests/test_p70_stage_judge.py
```

完整历史 tests 中有读取旧实验资产的测试；源码保留不代表其所有 fixture 都随新项目发布。针对当前改动选择实际相关检查，缺资料必须说明，不用伪造 fixture 让历史测试通过。

```bash
python3 tools/evidence/build_project_dashboard.py
git diff --check
```

Dashboard 只读绑定图和证据并生成静态页面。修改 STATUS 或其绑定文件后重新生成；不要手改生成 HTML 或统计数字。迁移校验和实际运行结果分别记录。

## 模型与 Git

任务模型通过 Git LFS 管理。下载后执行 `git lfs pull`；不能把 LFS 指针文本当 PyTorch checkpoint。模型字节变化属于技能版本变化，不应当作为路径迁移顺手修改。阶段路由、manifest 和 release 必须与实际文件相符。

新实验先写范围、输入权限、预算和候选冻结，再启动。检查／打包／文档任务不自动启动 simulator，部署重放已开放条件不能算独立测试。原始结果封存后再生成注释视频和统计。

## 查看

在仓库根按需启动 `python3 -m http.server 59162 --bind 127.0.0.1 --directory .`，打开 `http://127.0.0.1:59162/dashboard/`。外部电脑连接当前主工作站时：

```bash
ssh -N -L 59162:127.0.0.1:59162 zheng@100.68.228.21
```

前提是工作站该端口的 HTTP 服务已经启动；先检查已有进程，不重复开服务。
