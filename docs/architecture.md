# Task Graph 与阶段局部自适应

一条演示提供物体角色、阶段边界、动作目的和交接关系。Agent 将它拆成结构化图，开发时在真实前驱产生的状态上探索各阶段，再串成连续完整任务。当前已保存的解法为 LIBERO-10 book→caddy 的 P70 R3。

```text
One-shot demo → Agent 拆解物体中心图
                        ↓
                   阶段开发／验证
                        ↓
Code 接近 → 短抓持 MLP → Code 抬离／定向／运输／近入口
                                             ↓
                               P68 短 Insert + 本阶段 z 更新
                                             ↓
                                  原任务目标与保持门
                        ↓
         失败证据 → Agent 修订图／交接／控制／技能 → 冻结检测
```

图、配置与实际执行一致性由 [task_bundle.py](../task_packages/book_to_caddy_r3/task_bundle.py)校验。运行时不调用 Agent 改图；本次图就是已经开发冻结的 R3。不同阶段的状态转换依赖明确的物体关系，接口缺失或不满足入口必须保留失败／未知。

## 当前控制器

简单关系动作用 [P70ObjectController](../prove/task_graph/p70_object_controller.py)。短抓持使用训练好的 8→64→64→6 MLP；该模型训练于合成关系误差教师样本，不是从 RGB 端到端训练的抓取系统。完整任务由 [p70_stage_runner](../tools/task_graph/p70_stage_runner.py)和 [p69_full_rsi_runner](../tools/task_graph/p69_full_rsi_runner.py)衔接，末段调用 [ReferenceBookRuntime](../prove/fast_adapter/p68_book_runtime.py)。

P68 的普通输入是两路 448×448 RGB、可见角色 mask、相机内外参和本体状态。冻结 DINOv3 提特征，任务训练的 LiLa / ROI 关系编码器输出几何读出；固定参考与当前已观察轮廓约束共同决定局部修正。`z` 是本阶段估计的修正状态，不是整任务图的 Memory。网络在部署时冻结，更新只消费过去／当前允许的测量。

可概括为：

\[
U_t=E_\phi(I_t^{\mathrm{views}},M_t,K,T,q_t),\qquad
\hat g_t=\operatorname{Decode}(g_{\mathrm{ref}},z_t)
\]

\[
a_t=\pi_\theta(\hat g_t,q_t),\qquad
z_{t+1}=\operatorname{Update}(z_t,\mathcal H_{\le t},g_{\mathrm{ref}})
\]

其中 `Update` 的具体变量、轮廓投影目标和数值规则以 [p64_book_bridge.py](../prove/fast_adapter/p64_book_bridge.py)为准；这不是本次迁移重新提出的损失，也不是把早期 P52 的视觉下一帧预测方法冒充当前 P68。进入阶段重置 z，离开阶段归档并清空。

## 两层贡献分别解释

改变图和 Code 的收益由旧／新完整系统比较回答；同图同 checkpoint 的 fixed-z／online-z 比较才回答局部在线更新的增量。P70 的完整修订结果不证明梯度独有贡献。局部成功或动作减少不等于完整成功，几何误差减小也不保证成功率提高。

当前采用模拟器可见 mask、标定、固定形状／关系先验和 Boolean 监督，不称全链纯视觉或真机验证。完整实验和限制由 [P70](../evaluations/outer_rsi_p70_object_graph_v01/ROOT_REPORT_ZH.md)与 [P68](../evaluations/fast_adapter_p68_cross_task_reference_v01/ROOT_REPORT_ZH.md)拥有。
