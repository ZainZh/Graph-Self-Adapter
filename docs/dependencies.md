# 外部依赖与机器配置

任务图、运行源码和任务训练权重属于本项目。标准 LIBERO 场景／资产、Python、DINOv3 和预训练 LiLa 是共享安装，不随项目重复打包；`Interaction-RSI` 的 prove/tools 不是执行依赖。

依赖合同为 [solution/DEPENDENCIES.json](../task_packages/book_to_caddy_r3/solution/DEPENDENCIES.json)。当前工作站继续使用原数据／环境目录，代码根目录自动绑定本仓库。另一台机器用忽略的 `local/DEPENDENCIES.local.json` 指定安装位置：

```json
{
  "python": "/path/to/environment/bin/python",
  "dino_model_dir": "/path/to/dinov3-vitl16-pretrain-lvd1689m",
  "lila_source": "/path/to/LiLa-WAM/models/vla_model_fm.py",
  "lila_checkpoint": "/path/to/LiLa-WAM/VISUAL_WEIGHTS.pt",
  "libero_root": "/path/to/LIBERO-PRO/libero",
  "runtime_overlay": "/path/to/optional/vision-site-packages",
  "rlinf_libero_root": "/path/to/RLinf-LIBERO-PRO"
}
```

RLinf-LIBERO-PRO 提供冻结批次使用的标准初态资源；它是运行依赖之一，不能用另一个初态文件替换而仍声称相同历史 cell。执行实际读取的 one-shot hdf5 随任务包保存于 solution/refs/demo；它是必要的任务参考，不需要外部旧演示目录。默认依赖绑定从任务包位置派生，可用 libero_one_shot_root 显式覆盖兼容材料。

也可通过任务入口的 `--dependencies /path/to/config.json` 指定配置；`--python`、`--dino-model-dir` 支持显式覆盖。`--repo-root` 只指定 Graph-Self-Adapter **源码根目录**。LIBERO 标准资产必须与合同兼容；视觉公共权重按已保存的哈希检查。

本机使用的主要版本为 Python 3.10.20、PyTorch 2.7.1、NumPy 1.26.4、MuJoCo 3.3.1、robosuite 1.4.1、transformers 4.56.2。全部测试版本及视觉基座哈希由 DEPENDENCIES.json 拥有；pyproject 的 extras 只是常见 Python 依赖入口，不替代 LIBERO、视觉权重和原生渲染环境的安装。

安装可编辑包时应在本仓库执行 `python -m pip install --no-deps -e .`，避免共享环境里混入旧源码。当前任务的 run.sh 会显式设定新源码路径，无需为迁移改动全局 Python 环境。
