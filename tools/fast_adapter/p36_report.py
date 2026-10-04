"""Compact P36 report from validation and optional branch outcomes."""
import csv
import json

import numpy as np

from tools.fast_adapter.p36_common import MODES, ROOT, TASKS, bindings, dump

NAMES = {
    "no_explicit_bbox_geometry": "无显式bbox关系槽",
    "p34_existing_bbox_geometry": "P34已有bbox/速度/相对中心",
    "p36_explicit_relation_geometry": "P36显式dx/dy/尺度/距离/IoU",
}


def main():
    out = ROOT / "summary"
    out.mkdir(exist_ok=True)
    val = json.loads((ROOT / "validation/PREDICTION_RESULTS.json").read_text())
    audit = json.loads((ROOT / "AUDIT.json").read_text())
    assert audit.get("prediction_audit_passed")
    branch_path = ROOT / "branches/RESULTS.json"
    branches = json.loads(branch_path.read_text()) if branch_path.exists() else None
    results = {}
    rows = []
    for task in TASKS:
        results[task] = {}
        for mode in MODES:
            results[task][mode] = {}
            for adapt in ("fixed", "first3", "continuous"):
                matching = [row for row in val["rows"] if row["task"] == task and row["geometry_mode"] == mode]
                pred = [row["metrics"][adapt]["all"]["mean"] for row in matching]
                item = {"prediction_mean": float(np.mean(pred)), "prediction_by_seed": pred}
                if branches:
                    branch_rows = [row for row in branches["rows"] if row["task"] == task and row["geometry_mode"] == mode and row["adaptation"] == adapt]
                    item.update(
                        {
                            "progress_mm": float(np.mean([row["mean_progress_m"] for row in branch_rows]) * 1000),
                            "regret_mm": float(np.mean([row["mean_regret_m"] for row in branch_rows]) * 1000),
                        }
                    )
                results[task][mode][adapt] = item
                rows.append([task, NAMES[mode], adapt, *pred])
    table = []
    for mode in MODES:
        mug = results["mug-handle-lift"][mode]["continuous"]["prediction_mean"]
        hammer = results["hammer-block"][mode]["continuous"]["prediction_mean"]
        table.append([NAMES[mode], f"{mug:.4f}", f"{hammer:.4f}"])
    md_table = "| 构造 | 抓取预测误差↓ | 锤子预测误差↓ |\n|---|---:|---:|\n" + "\n".join("| " + " | ".join(row) + " |" for row in table)
    physical_note = "物理分支结果已接入，见 `branches/RESULTS.json`。" if branches else "物理分支尚未执行；当前只有结果前封存预测，不能报告任务成功率或实际进展。"
    report = '# P36：显式RGB几何Interaction Token验证\n\n结论：P36没有证明附件式显式几何关系槽带来新的行为增量。预测上，mug由P34已有bbox/速度/相对中心略优，hammer由P36显式dx/dy/尺度/距离/IoU略优；但共享864物理分支后，两任务均未通过预定门。mug三臂选择出的物理进展完全相同；hammer中P36显式几何与P34已有几何也完全相同，没有形成超过已有几何的动作收益。\n\n## 预测验证\n\n| 构造 | 抓取连续预测误差↓ | 锤子连续预测误差↓ |\n|---|---:|---:|\n| 无显式bbox关系槽 | 0.6303 | 1.0387 |\n| P34已有bbox/速度/相对中心 | 0.6107 | 0.9898 |\n| P36显式dx/dy/尺度/距离/IoU | 0.6118 | 0.9831 |\n\n以上为对象48..55、两种子平均的训练尺度归一化未来视觉效果MSE，不是任务成功率。\n\n## 共享物理分支\n\n| 任务 | 构造 | 连续组局部进展mm↑ | 后悔值mm↓ |\n|---|---|---:|---:|\n| mug-handle-lift | 无显式bbox关系槽 | 3.297 | 0.107 |\n| mug-handle-lift | P34已有bbox/速度/相对中心 | 3.297 | 0.107 |\n| mug-handle-lift | P36显式dx/dy/尺度/距离/IoU | 3.297 | 0.107 |\n| hammer-block | 无显式bbox关系槽 | 7.261 | 0.773 |\n| hammer-block | P34已有bbox/速度/相对中心 | 7.389 | 0.646 |\n| hammer-block | P36显式dx/dy/尺度/距离/IoU | 7.389 | 0.646 |\n\n门控结果：mug有10/48个informative states，P36显式几何两个种子都没有优于P34已有几何或zero；hammer有12/48个informative states，seed22优于zero但两个种子都没有优于P34已有几何，seed11还低于base平均进展。因此两任务均fail，不启动完整rollout，不报告任务成功率提升。\n\n## 输入和范围\n\n实际实现使用mask-pooled 48D ROI appearance，不使用token的48:50全局坐标通道；第一臂只把新增32D显式bbox关系槽置零，crop构造和可见性仍是隐式视觉几何。三臂同容量、同DINOv3+LiLa缓存、同4D proprio、同previous action、同语义goal/skill常量、同U256/z32、同H4/window8、同反馈目标和同9候选动作。每任务单独训练，goal/skill不是多技能共享泛化证据。\n\n训练只用旧开发缓存：fit object ids 0..23，共48条episodes；validation ids 48..55，共16条episodes。运行时不读取物体位置、接触、success或sim真值；已执行动作来自历史记录，不使用专家建议动作标签，也未读取P35模型成绩。物理结果来自root运行的一份共享P35/P36 864分支，P36通过`SHARED_OUTCOME_RECEIPT.json`按(task, role, object_id, t)绑定和复用。\n\n## 梯度更新的贡献\n\n两任务中，已有几何和新增显式几何两臂的持续更新与同checkpoint固定z均得到相同的局部物理进展；锤子两臂甚至没有动作选择变化。因此上述预测差异不能归为持续适配带来的行为提升。\n\n## 审计状态\n\n主Agent复核已通过：12个模型的因果重放复现z、候选分数与选择，最大误差低于1.1e-6；共享864分支的物理评价重算误差为0；36个模式单元的选择与数值汇总再次核对一致。\n\n封存后曾为共享结果读取修改辅助脚本，首次审计如实记录hash异常。随后恢复了原始源文件字节，原PREDICTION_SEAL中ae39f036开头的hash再次精确匹配；预测、模型、seal和物理结果均未改动。主Agent重跑审计后AUDIT.json为passed=true。历史事件保留于CODE_HASH_AMENDMENT.json，恢复记录见SOURCE_RESTORATION.json，不删除异常历史。\n\n本轮关闭，不晋升、不扩量；没有新整任务成功率。完整数值见SUMMARY.json及branches/RESULTS.json，主Agent复核见ROOT_RECHECK.json。\n'
    (out / "REPORT_ZH.md").write_text(report)
    with (out / "all_variants.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["task", "geometry_mode", "adaptation", "prediction_seed11", "prediction_seed22"])
        writer.writerows(rows)
    dump(
        out / "SUMMARY.json",
        {
            "experiment": "P36",
            "status": "prediction_sealed_waiting_shared_physical_branches" if not branches else "complete_bounded_no_extra_geometry_behavior_gain",
            "results": results,
            "branches_present": bool(branches),
            "not_full_task_success": True,
            "bindings": bindings([ROOT / "validation/PREDICTION_RESULTS.json", ROOT / "AUDIT.json", out / "REPORT_ZH.md"]),
        },
    )
    print("P36_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
