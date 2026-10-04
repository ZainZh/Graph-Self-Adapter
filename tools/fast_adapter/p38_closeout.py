"""Root closeout: verify immutable inputs and collect bounded scientific claims."""
import hashlib
import json
import time
from pathlib import Path

R=Path('evaluations/fast_adapter_p38_causal_v01');S=R/'summary'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    physical=read(S/'final_PHYSICAL_RESULTS.json');table=physical['table']
    numeric=read(R/'online_mapping/summary/CAUSAL_RESULTS.json')['summary']
    pa=read(S/'PHYSICAL_AUDIT.json');na=read(R/'online_mapping/summary/CAUSAL_AUDIT.json')
    transfer=read(R/'TRANSFER_VERIFIED.json')
    assert pa['passed_actions_latents_physics_outcomes'] and na['passed'] and transfer['all_hashes_match']
    verified=[]
    for name in ('WAVE1_FREEZE.json','PUSH_PROTOCOL_FREEZE.json','ONLINE_MAPPING_FREEZE.json',
                 'ATTRIBUTION_FOLLOWUP_FREEZE.json','PUSH_FINAL_FREEZE.json','ONLINE_FINAL_FREEZE.json','MAPPING_FINAL_FREEZE.json'):
        j=read(R/name)
        bindings=j.get('files',j.get('hashes',{}))
        for path,sha in bindings.items():
            assert digest(Path(path))==sha,(name,path)
        verified.append({'path':str(R/name),'sha256':digest(R/name),'bound_files':len(bindings)})
    final_protocol=read(R/'ONLINE_CAUSAL_FINAL_PROTOCOL.json')
    assert digest(Path(final_protocol['file']))==final_protocol['sha256']
    online_freeze=read(R/'ONLINE_FINAL_FREEZE.json')
    assert digest(R/'mapping/controller/model.pt')==online_freeze['controller_hash']
    assert digest(Path('tools/fast_adapter/p38_online_causal_final.py'))==online_freeze['final_runner_hash']
    def rate(split,kind,arm='continuous'):return table[f'{split}/{kind}/{arm}']['success_rate']*100
    def nr(kind,split,arm='continuous'):return numeric[kind+'/'+split][arm]['success_rate']*100
    report=f'''# P38 主Agent结论：持续适配机制成立于受控范围，控制相关反馈是关键

这轮得到了比“预测损失下降”更完整的证据：**实际交互后果改变了latent，latent改变直接动作修正；使用当前物体的反馈提高成功率，替换为错误物体反馈或固定训练经验会失去收益。**

但结果支持的是有明确响应结构的局部在线适配。它没有证明任意高维视觉latent都会自我提升，也没有证明梯度方法普遍优于History或经典系统辨识。长期RSI、完整VLA和自然抓取仍需后续验证。

本轮由主Agent独立设计、实现与验收，使用已授权服务器四张GPU。重计算在服务器，本机承担分析与展示。全部预定独立批次已结束，没有继续训练或追加测试。

## 最关键的物理结果

任务是**三维MuJoCo物理中的单轴近接触推滑块**：工具施力，物块沿导轨移动约26–34mm，最终对齐图像中的标记。隐藏条件是滑块的阻力、干摩擦和质量；执行接口没有人为动作旋转或倍率扰动。不是机械臂任务，也不是官方benchmark。

最严格的对照采用32对新几何配置，共64个环境条件。每对可见几何相同，但一边阻力低、另一边高；初始位置观测和工具编码器数值完全相同。两个训练种子分别执行，不能把它们加起来当成128个独立环境。

| 对照组的方法 | 成功率 | 两个模型各自成功数 |
|---|---:|---|
| 冻结基础控制 | {rate('contrast','references','base'):.2f}% | 32/64，公共基线只执行一次 |
| 固定训练平均latent | {rate('contrast','motion','train_mean'):.2f}% | 32/64、32/64 |
| 只更新前三次，然后固定 | {rate('contrast','motion','first3'):.2f}% | 32/64、32/64 |
| 持续使用错误物体反馈 | {rate('contrast','motion','donor'):.2f}% | 7/64、7/64 |
| **持续使用当前物体的可见运动反馈** | **{rate('contrast','motion'):.2f}%** | **64/64、64/64** |
| 持续DINO特征反馈 | {rate('contrast','appearance'):.2f}% | 50/64、47/64 |
| 持续结合运动与DINO反馈 | {rate('contrast','combined'):.2f}% | 64/64、64/64 |
| 同信息History | {rate('contrast','history'):.2f}% | 32/64、32/64 |
| 同4步窗口的最小二乘 | {rate('contrast','references','least_squares'):.2f}% | 41/64 |
| 已知阻力的Oracle | {rate('contrast','references','oracle'):.2f}% | 64/64 |

运动反馈相对错误反馈高89.06个百分点，12项主比较校正后的配对簇bootstrap区间为[78.13,98.44]个百分点；相对纯特征反馈高24.22个百分点，校正区间为[11.72,36.72]。对照组中，结合表示尚未额外超过运动反馈。

这里100%是本批有限样本的观察结果。按32个几何对统计，全通过仍不能保证真实成功率为100%；全体配对差值相同时bootstrap区间可能退化，不能把这解读成没有泛化不确定性。

## 其余设置没有省略

每档32个新条件，两个固定模型在条件内平均。这里的OOD指更大物体阻力，不是换物体类别或更大形状变化。

| 设置 | 基础控制 | 运动反馈 | DINO反馈 | 结合表示 | History | 最小二乘 |
|---|---:|---:|---:|---:|---:|---:|
| 原生阻力 | {rate('native','references','base'):.2f}% | {rate('native','motion'):.2f}% | {rate('native','appearance'):.2f}% | {rate('native','combined'):.2f}% | {rate('native','history'):.2f}% | {rate('native','references','least_squares'):.2f}% |
| 范围内新条件 | {rate('id','references','base'):.2f}% | {rate('id','motion'):.2f}% | {rate('id','appearance'):.2f}% | {rate('id','combined'):.2f}% | {rate('id','history'):.2f}% | {rate('id','references','least_squares'):.2f}% |
| 更大阻力 | {rate('ood','references','base'):.2f}% | {rate('ood','motion'):.2f}% | {rate('ood','appearance'):.2f}% | {rate('ood','combined'):.2f}% | {rate('ood','history'):.2f}% | {rate('ood','references','least_squares'):.2f}% |

结合表示在范围内的均值更好，适合作为后续候选；这不是所有设置下已经确认的统一最佳编码器。更大阻力组中完整前缀最小二乘同样达到100%，因此普通系统辨识已经能解决一部分问题；对照组的LS只保留相同4步窗口，两个LS预算区别不能隐藏。

最初开发时，高阻力组内交换反馈也经常成功，说明“知道这是高阻力”可能足够，不能证明物体专属辨识。随后在模型完全不变的情况下，预先冻结了低/高阻力配对及训练平均latent对照。新独立确认支持当前反馈的作用，旧的弱归因结果完整保留。

## 本轮具体用了什么表示和损失

全部普通物理控制使用448×448 RGB。固定颜色规则定位工具、物块与参考标记，已知相机标定将像素差换成距离；不读模拟器物块坐标、接触标签或成功标签。颜色标记和固定相机属于明确的感知限制，并不是任意物体的通用Observer。工具编码器被记录，本轮最小控制核心没有依靠它识别隐藏阻力。

视觉分支仍使用冻结的真实DINOv3 ViT-L/16和既有LiLa多层融合/查询adapter。全景输出49152维，训练数据上拟合PCA16，保留约88.89%的训练方差。没有把模拟器真值框送给视觉分支。

三种梯度方法共享**2维响应latent、近静态力—位移预测形式、直接力控制器和动作预算**。预测形式为正部仿射响应：`预测位移 = max(0, estimated_gain × force + estimated_bias)`，位移按2mm归一化。共同初值严格对应固定基座。

- **运动反馈**：用RGB读出的真实物块位移监督预测位移。
- **DINO反馈**：用实际DINO/LiLa特征变化监督预测的特征变化。冻结的小网络在训练材料中学到“位移会怎样改变特征”；训练这个网络时使用RGB测量位移，不使用模拟器几何标签。
- **结合表示**：将归一化运动损失和特征变化损失各取一半。
- **History**：读取相同的特征、运动与动作序列，用GRU更新状态，没有测试时梯度。

每个已完成动作产生一次反馈，使用最近4步更新latent；动作由当前RGB目标关系和新latent直接计算。没有候选动作列表、额外人工评分或任务成功标签驱动更新。三次共同探测计入30步总预算；第30次反馈更新被记录，但不能影响已经结束的任务。

梯度分支的外层学习量很小：学习一个固定的2×2正定梯度预条件矩阵；响应形式和解析动作接口是额外结构先验。不能把这个结构化正结果写成原先自由32维视觉网络已经全面修好。比较的是反馈表示及其对应的预测目标，并非单独证明某个角色注意力编码器最优。

训练使用96条、每条30步的实际物理探索轨迹；视觉变化预测器训练4000步，四种适配方法各两个种子、1500次外层更新。外层在训练环境中使用真实阻力生成目标力标签；测试内层只接触已完成动作的观测后果。模型参数在测试期间冻结。

## 数值机制与公平的History对照

先做了共用冻结神经动作头的仿射响应测试，并保留原来的绝对/相对状态编码器对照。首轮S3/U3独立测试中，两者同分布成功率分别为28.19%和27.80%，没有支持“改成关系形式就自动变好”。显式响应结构更有效，但不能把同时改变动力学结构的收益全算成token编码器贡献。

随后按真实部署方式重新训练：没有独立校准轨迹，每回合8次动作，前7次动作之后各更新一次；外层4000次，跨大量环境通过实际数值任务误差训练。History也按同一时序和预算重新训练，不能用旧的少步History在连续执行中的失败来宣称梯度胜出。

新独立同分布测试的32维学习latent为：**冻结z {nr('effect','id','frozen'):.2f}% → 连续梯度 {nr('effect','id'):.2f}%**；错误反馈{nr('effect','id','donor'):.2f}%，固定训练平均latent {nr('effect','id','train_mean'):.2f}%。相对冻结、错配和平均经验的12项校正区间均为正，说明不是仅靠静态动作头变强。

但重新训练的History为**{nr('history','id'):.2f}%**，仍明显更好。32维梯度相对最小二乘的差值区间包含零，不能宣称一般算法优势。

更大映射偏差中，32维梯度为{nr('effect','ood'):.2f}%，History为{nr('history','ood'):.2f}%。把真实环境参数送入同一个冻结神经动作头也只有{nr('effect','ood','oracle'):.2f}%，而解析Oracle为100%；因此动作头外推也是瓶颈。真参数输入一个不完美的神经动作头并不是数学控制上界，History偶尔超过它并不构成矛盾。

本节与首轮使用不同的独立条件，不把跨批均值变化写成纯训练时序的单因素提升。全部原始数据和更新次数消融保留。

## 验收、视频和限制

独立物理测试共5600条完整执行，来自原三档96个环境条件和32对/64个阻力对照条件，再重复方法与固定模型。**5600不是5600个独立物体。** 共173600个物理状态（含初态）逐个重放，状态误差为0，动作最大复算差8.65e-6，latent最大差9.54e-7；成功标签全部一致。连续数值测试的216个方法批次也完整重放了动作、latent和带噪反馈。回传1681个文件全部哈希一致。

感知重现的例外也保留：1120次对比组初始配对检查中，一对History错配分支的RGB哈希不同，但检测位置和工具编码器完全一致；三个梯度分支的初始配对图像全部一致。两个视频重渲染的279帧中有一帧RGB哈希不同，物理位置和检测位置都完全一致。逐像素差异原因尚未定位，因此不声称所有RGB逐像素复现。没有据此修改任何动作、latent、阈值或成功率。

建议保留的结论是：**用可观测的动作后果更新与控制接口相连的latent，有可靠的局部价值；DINO可以提供外观和上下文，适配监督应优先对齐实际关系/运动变化。** 下一步若扩展，应保留当前已验证模块和冻结/均值/错配对照，逐步解除已知响应形式、颜色标记和单轴任务的限制；该下一批尚未启动。

[成功率、过程曲线与注释视频](index.html) · [全部物理数字与区间](final_PHYSICAL_RESULTS.json) · [物理复算](PHYSICAL_AUDIT.json) · [数值完整结果](../online_mapping/summary/CAUSAL_RESULTS.json) · [数值复算](../online_mapping/summary/CAUSAL_AUDIT.json)

[最初协议](../PROTOCOL.json) · [物理协议](../PUSH_PROTOCOL_FREEZE.json) · [归因补充协议](../ATTRIBUTION_FOLLOWUP_FREEZE.json) · [物理最终冻结](../PUSH_FINAL_FREEZE.json) · [连续数值最终冻结](../ONLINE_FINAL_FREEZE.json)
'''
    (S/'ROOT_REPORT_ZH.md').write_text(report)
    acceptance={'schema':'prove.fast-adapter.root-acceptance.v1','status':'completed_valid_bounded_positive_with_visual_replay_caveat',
      'time':time.time(),'full_numerical_audit_passed':True,'frozen_artifact_bindings_verified':verified,
      'causal_online_adaptation_supported':True,'control_relevant_feedback_supported':True,
      'generic_free_visual_latent_fixed':False,'gradient_universal_superiority':False,'robot_or_complete_RSI_claim':False,
      'physics':pa,'continuous_numeric_batches_replayed':len(na['replays']),'transfer':transfer,
      'claim_scope':'fixed small models,known response structures,controlled numerical mappings and single-axis MuJoCo contact with color-key RGB observations',
      'limitations':['shared analytic contact controller and known quasi-static response','colored scene/fixed calibration','two fixed physical model seeds','DINO branch uses shared RGB goal geometry','rare raster reproduction mismatch,all relevant recorded states/actions/outcomes replayed','History is stronger in numeric ID;full-prefix LS solves high resistance'],
      'report':str(S/'ROOT_REPORT_ZH.md'),'page':str(S/'index.html'),'next_batch_started':False}
    (S/'ROOT_ACCEPTANCE.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+'\n')
    artifacts=[S/'ROOT_REPORT_ZH.md',S/'ROOT_ACCEPTANCE.json',S/'final_PHYSICAL_RESULTS.json',S/'PHYSICAL_AUDIT.json',
               R/'online_mapping/summary/CAUSAL_RESULTS.json',R/'online_mapping/summary/CAUSAL_AUDIT.json',R/'TRANSFER_VERIFIED.json']
    artifacts+=list((S/'media_v03').glob('*_ANNOTATIONS.json'))
    (S/'EVIDENCE_INDEX.json').write_text(json.dumps({'schema':'prove.evidence-index.v1','id':'P38','artifacts':[
         {'path':str(p),'sha256':digest(p)} for p in artifacts]},ensure_ascii=False,indent=2)+'\n')
    print('Root closeout written; all frozen source/model bindings verified')


if __name__=='__main__':main()
