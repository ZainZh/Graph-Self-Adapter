import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_project_dashboard_under_test", ROOT / "tools/evidence/build_project_dashboard.py")
dashboard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


def _write(root: Path, relative: str, value) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value) + "\n")
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _cohort(root: Path, name: str, tasks: list[str], *, condition: str,
            successes: int, controller: str = "fixed_z") -> dict:
    rows = []
    total = 0
    for index, task in enumerate(tasks):
        success = index < successes
        run = _write(root, f"runs/{name}/{task}.json", {
            "schema": "graph-self-adapter.twin-run.v1", "task": task,
            "strict_success": success,
        })
        rows.append({"task": task, "trials": 1, "successes": int(success),
                     "rate": float(success), "reports": [run]})
        total += int(success)
    summary = _write(root, f"runs/{name}/SUMMARY.json", {
        "schema": "graph-self-adapter.twin-cohort-summary.v1",
        "candidate_id": name, "condition_set": condition, "controller": controller,
        "trials": len(tasks), "successes": total, "tasks": rows,
    })
    return {**summary, "candidate_id": name, "condition_set": condition}


def _v02_report(root: Path, tasks: list[str], *, error: bool = False,
                schema: str = "graph-self-adapter.twin-v02-development-batch-report.v1") -> dict:
    runs = []
    rows = []
    successes = 0
    for index, task in enumerate(tasks):
        success = index == 0
        run = {"arm_id": "B0", "candidate_id": "v02_b0", "controller": "fixed_z",
               "task": task, "trial": 0, "strict_success": success, "error": None}
        if error and index == 1:
            run["strict_success"] = False
            run["error"] = "RuntimeError: retained execution error"
        else:
            source = _write(root, f"runs/v02/{task}.json", {
                "schema": "graph-self-adapter.twin-run.v1", "task": task,
                "strict_success": success, "candidate_id": "v02_b0", "controller": "fixed_z",
            })
            run["report"] = source
        runs.append(run)
        rows.append({"arm_id": "B0", "task": task, "denominator": 1,
                     "strict_successes": int(success)})
        successes += int(success)
    payload = {
        "schema": schema, "runs": runs, "tasks": rows,
        "execution_errors": int(error),
        "arm_totals": {"B0": {"denominator": len(tasks),
                                 "strict_successes": successes}},
    }
    return _write(root, "v02/BATCH_REPORT.json", payload)


def _fixture(tmp_path: Path, monkeypatch, *, videos: bool = True):
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)
    tasks = [f"task_{i}" for i in range(10)]
    graph = _write(tmp_path, "graph.json", {
        "stage_ids": [{"id": "s", "entry": "e", "exit": "x"}],
        "solver_routing": {"s": "solver"},
    })
    scope = _write(tmp_path, "scope.json", {
        "schema": "graph-self-adapter.twin-scope.v1", "tasks": tasks})
    plan = _write(tmp_path, "plan.json", {
        "schema": "graph-self-adapter.twin-root-plan.v1",
        "tasks": [{"task": task, "label": f"任务 {i}", "cycles": [{}]} for i, task in enumerate(tasks)],
    })
    development = _cohort(tmp_path, "r1_negative", tasks, condition="development", successes=0)
    development["label"] = "R1 负结果"
    test = _cohort(tmp_path, "selected", tasks, condition="test", successes=10)
    test["label"] = "独立测试"
    twin = {
        "schema": "graph-self-adapter.twin-status.v1",
        "engineering_nominal_passed": True,
        "tasks": [{"task": task, "physics_qualified": True} for task in tasks],
        "report_bindings": [development, test],
        "task_videos": [],
    }
    if videos:
        video = _write(tmp_path, "videos/task0.mp4", b"video")
        source = development["path"].replace("SUMMARY.json", "task_0.json")
        source_path = tmp_path / source
        twin["task_videos"] = [{
            "task": "task_0", "label": "成功示例", **video,
            "report_path": source,
            "report_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }]
    twin_binding = _write(tmp_path, "twin.json", twin)
    bindings = {
        "task_graph": graph, "twin_scope": scope, "twin_plan": plan, "twin_status": twin_binding,
        "readme": _write(tmp_path, "README.md", b"readme"),
        "engineering_acceptance": _write(tmp_path, "acceptance.json", b"{}"),
        "science_report": _write(tmp_path, "science.json", b"{}"),
        "video": _write(tmp_path, "history.mp4", b"history"),
    }
    return {"headline": "Dashboard", "summary": "summary", "bindings": bindings}, tasks, twin


def test_dashboard_separates_test_and_development_and_keeps_metadata(tmp_path, monkeypatch):
    status, _, _ = _fixture(tmp_path, monkeypatch)
    html = dashboard.render(status)
    assert html.index("独立测试结果") < html.index("展开研发版本结果")
    assert "独立测试" in html and "R1 负结果" in html
    assert "test · fixed_z" in html and "development · fixed_z" in html
    assert "10/10（100%）" in html and "0/10（0%）" in html
    assert "max-width:1560px" in html
    assert '<video controls preload="metadata" src="../videos/task0.mp4">' in html
    assert "源运行报告" in html


def test_empty_video_binding_does_not_fabricate_video_section(tmp_path, monkeypatch):
    status, _, _ = _fixture(tmp_path, monkeypatch, videos=False)
    html = dashboard.render(status)
    assert "每任务注释视频" not in html


def test_report_label_and_task_video_bindings_remain_hash_and_task_bound(tmp_path, monkeypatch):
    status, tasks, twin = _fixture(tmp_path, monkeypatch)
    reports = dashboard.load_cohort_reports(twin, tasks)
    assert reports[0]["label"] == "R1 负结果"
    videos = dashboard.load_task_videos(twin, tasks)
    assert videos["task_0"][0]["label"] == "成功示例"
    bad_hash = json.loads(json.dumps(twin)); bad_hash["task_videos"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        dashboard.load_task_videos(bad_hash, tasks)
    bad_task = json.loads(json.dumps(twin)); bad_task["task_videos"][0]["task"] = "task_1"
    with pytest.raises(ValueError, match="task mismatch"):
        dashboard.load_task_videos(bad_task, tasks)
    bad_label = json.loads(json.dumps(twin)); bad_label["report_bindings"][0]["label"] = ""
    with pytest.raises(ValueError, match="label"):
        dashboard.load_cohort_reports(bad_label, tasks)


def test_current_report_package_and_chart_are_rendered_from_bindings(tmp_path, monkeypatch):
    status, _, _ = _fixture(tmp_path, monkeypatch, videos=False)
    status["bindings"].update({
        "twin_root_report": _write(tmp_path, "final_report.md", b"report"),
        "twin_task_package": _write(tmp_path, "package/README.md", b"package"),
        "twin_paired_report": _write(tmp_path, "paired.json", b"{}"),
        "twin_final_chart_svg": _write(tmp_path, "chart.svg", b"<svg/>") ,
        "twin_final_chart_png": _write(tmp_path, "chart.png", b"png"),
    })
    html = dashboard.render(status)
    assert "最终科学报告" in html and "可复用任务包" in html
    assert "固定／在线配对报告" in html
    assert 'src="../chart.png"' in html
    assert "统计由封存汇总报告生成" in html


def test_v02_activity_is_hash_bound_result_free_and_discloses_privileged_inputs(tmp_path, monkeypatch):
    status, _, _ = _fixture(tmp_path, monkeypatch, videos=False)
    focused = ["put_knife_board_new", "put_mug_on_box", "sorting_cube_bimanual"]
    status["bindings"].update({
        "twin_v02_scope": _write(tmp_path, "v02/SCOPE.json", {
            "schema": "graph-self-adapter.twin-v02-scope.v1",
            "focused_tasks": focused,
        }),
        "twin_v02_plan": _write(tmp_path, "v02/ROOT_PLAN_ZH.md", b"plan"),
        "twin_v02_input_audit": _write(tmp_path, "v02/INPUT_AUDIT_ZH.md", b"audit"),
        "twin_v02_status": _write(tmp_path, "v02/STATUS.json", {
            "schema": "graph-self-adapter.twin-v02-status.v1",
            "execution_authorized": False,
            "results_available": False,
            "independent_test_opened": False,
            "completed_algorithm_episodes": 0,
            "focused_tasks": [{"task": task} for task in focused],
        }),
        "twin_v01_input_erratum": _write(tmp_path, "v01/INPUT_PERMISSION_ERRATUM.json", {
            "schema": "graph-self-adapter.twin-v01-input-permission-erratum.v1",
        }),
    })
    html = dashboard.render(status)
    assert "已封存历史：Twin RSI v02（与 v01 分开统计）" in html
    assert "待 Root 执行许可" in html and "结果为空" in html
    assert "独立测试 seed 92005" in html and "未打开" in html
    assert "连续物体真值" in html and "接触、支撑和阶段完成 Boolean" in html
    assert "本机 SAM 代码和权重已存在" in html
    assert html.index("已封存历史：Twin RSI v02") < html.index("独立测试结果")


def test_v02_activity_bindings_are_all_or_none_and_hash_checked(tmp_path, monkeypatch):
    status, _, _ = _fixture(tmp_path, monkeypatch, videos=False)
    status["bindings"]["twin_v02_scope"] = _write(tmp_path, "v02/SCOPE.json", {})
    status_path = tmp_path / "dashboard-status.json"
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="v02 dashboard bindings must be all-or-none"):
        dashboard.load_status(status_path, tmp_path)


def test_v02_report_validates_sources_retains_errors_and_renders_pending(tmp_path, monkeypatch):
    status, _, _ = _fixture(tmp_path, monkeypatch, videos=False)
    focused = ["put_knife_board_new", "put_mug_on_box", "sorting_cube_bimanual"]
    v02_binding = _v02_report(tmp_path, focused, error=True)
    status["bindings"].update({
        "twin_v02_scope": _write(tmp_path, "v02/SCOPE.json", {
            "schema": "graph-self-adapter.twin-v02-scope.v1", "focused_tasks": focused}),
        "twin_v02_plan": _write(tmp_path, "v02/ROOT_PLAN_ZH.md", b"plan"),
        "twin_v02_input_audit": _write(tmp_path, "v02/INPUT_AUDIT_ZH.md", b"audit"),
        "twin_v02_status": _write(tmp_path, "v02/STATUS.json", {
            "schema": "graph-self-adapter.twin-v02-status.v1", "execution_authorized": True,
            "results_available": True, "independent_test_opened": False,
            "completed_algorithm_episodes": 3,
            "focused_tasks": [{"task": task} for task in focused],
        }),
        "twin_v01_input_erratum": _write(tmp_path, "v01/INPUT_PERMISSION_ERRATUM.json", {
            "schema": "graph-self-adapter.twin-v01-input-permission-erratum.v1"}),
        "twin_v02_development_b0_a1_report": {**v02_binding, "label": "B0 基线"},
    })
    status["bindings"].update({
        "twin_v02_root_report": _write(tmp_path, "v02/ROOT_REPORT.md", b"root report"),
        "twin_v02_task_package": _write(tmp_path, "selected/README.md", b"selected package"),
        "twin_v02_package_verification": _write(tmp_path, "v02/PACKAGE_CHECK.json", {}),
    })
    html = dashboard.render(status)
    assert '../v02/ROOT_REPORT.md' in html and 'v02 科学报告' in html
    assert '../selected/README.md' in html and '复用包检查与 dry-run' in html
    assert "B0 统一后端父图" in html and "B0 · v02_b0 · fixed_z" in html
    assert "1/1（100%）；失败 0" in html
    assert "0/1（0%）；失败 1（执行错误 1）" in html
    assert "A2 fixed-z" in html and "待测" in html
    assert "UNKNOWN、运行错误及其他非严格成功均保留在失败分母中" in html
    assert "v02 独立测试" in html


def test_v02_report_rejects_denominator_and_source_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)
    tasks = ["knife", "mugs", "sorting"]
    binding = _v02_report(tmp_path, tasks)
    payload = json.loads((tmp_path / binding["path"]).read_text())
    payload["tasks"][0]["denominator"] = 2
    binding = _write(tmp_path, binding["path"], payload)
    with pytest.raises(ValueError, match="differs from source runs"):
        dashboard.load_v02_result_report(binding, tasks)
    payload["tasks"][0]["denominator"] = 1
    payload["runs"][0]["report"]["sha256"] = "bad"
    binding = _write(tmp_path, binding["path"], payload)
    with pytest.raises(ValueError, match="source report hash mismatch"):
        dashboard.load_v02_result_report(binding, tasks)


def test_v02_mixed_selected_arm_validates_candidate_per_task_and_source(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'REPO_ROOT', tmp_path)
    tasks = ['knife', 'mugs', 'sorting']
    binding = _v02_report(tmp_path, tasks)
    payload = json.loads((tmp_path / binding['path']).read_text())
    candidates = dict(zip(tasks, ['v02_a2', 'v02_a2', 'v02_a3']))
    for run in payload['runs']:
        run.update(arm_id='selected_online', candidate_id=candidates[run['task']], controller='online_z')
        source_path = tmp_path / run['report']['path']
        source = json.loads(source_path.read_text())
        source.update(candidate_id=run['candidate_id'], controller='online_z')
        run['report'] = _write(tmp_path, run['report']['path'], source)
    for row in payload['tasks']: row['arm_id'] = 'selected_online'
    payload['arm_totals']['selected_online'] = payload['arm_totals'].pop('B0')
    binding = _write(tmp_path, binding['path'], payload)
    result = dashboard.load_v02_result_report(binding, tasks)
    assert result['arms'][0]['candidate_by_task'] == candidates
    assert result['arms'][0]['candidate_id'] == 'mixed_per_task_selection'
    assert result['arms'][0]['candidate_label'] == '逐任务混合选版'
    assert result['arms'][0]['controller'] == 'online_z'
    source_binding = payload['runs'][2]['report']
    source = json.loads((tmp_path / source_binding['path']).read_text())
    source['candidate_id'] = 'v02_a2'
    payload['runs'][2]['report'] = _write(tmp_path, source_binding['path'], source)
    binding = _write(tmp_path, binding['path'], payload)
    with pytest.raises(ValueError, match='source run candidate mismatch'):
        dashboard.load_v02_result_report(binding, tasks)


def _v02_videos(root):
    condition = {'task': 'knife', 'trial': 6}
    digest = hashlib.sha256(json.dumps(condition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    condition_binding = {'sha256': digest, 'embedded_canonical_sha256': digest}
    clips = []
    for arm in ('parent', 'selected'):
        video = _write(root, f'media/twin_rsi_v02/{arm}.mp4', arm.encode())
        report = _write(root, f'runs/twin_rsi_v02/{arm}.json', {
            'schema': 'graph-self-adapter.twin-run.v1', 'task': 'knife',
            'controller': 'fixed_z', 'condition': condition,
            'bindings': {'condition_sha256': digest}})
        inputs = {'report': report,
                  'visual_trace': _write(root, f'runs/twin_rsi_v02/{arm}.npz', b'trace'),
                  'scene': {'path': '/external/scene.xml', 'sha256': 'scene'},
                  'condition': condition_binding,
                  'initialization': {'mj_step_executed': False,
                      'candidate_freeze': _write(root, f'packages/{arm}.json', {}),
                      'physics_source': _write(root, f'snapshot/{arm}.py', b'physics')}}
        sidecar = _write(root, f'media/twin_rsi_v02/{arm}.render.json', {
            'schema': 'graph-self-adapter.twin-video.v1',
            'offline_visual_replay_only': True, 'policy_or_adapter_executed': False,
            'video': video, **inputs})
        clips.append({'video': video, 'render_sidecar': sidecar,
                      'sealed_inputs': inputs, 'offline_only': True})
    manifest = {'schema': 'graph-self-adapter.twin-v02-comparison-videos.v1',
                'offline_visual_replay_only': True,
                'controller_adapter_evaluator_or_mj_step_executed': False,
                **{key: _write(root, f'v02/{key}.json', {})
                   for key in ('render_source', 'final_report', 'final_selection', 'plan')},
                'clips': clips, 'comparisons': [{
                    'task': 'knife', 'trial': 6, 'label': '父图失败／选版成功',
                    'video': _write(root, 'media/twin_rsi_v02/comparison.mp4', b'comparison'),
                    'left_parent': clips[0]['video'], 'right_selected': clips[1]['video'],
                    'source_render_sidecars': [c['render_sidecar'] for c in clips]}]}
    return manifest, _write(root, 'v02/VIDEO_MANIFEST.json', manifest)


def test_v02_video_manifest_verified_and_rejects_corrupt_media(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'REPO_ROOT', tmp_path)
    manifest, binding = _v02_videos(tmp_path)
    with pytest.raises(ValueError, match='hash mismatch'):
        dashboard.load_v02_comparison_videos({**binding, 'sha256': 'bad'}, ['knife'])
    videos = dashboard.load_v02_comparison_videos(binding, ['knife'])
    assert videos[0]['path'] == 'media/twin_rsi_v02/comparison.mp4'
    assert videos[0]['label'] == manifest['comparisons'][0]['label']
    (tmp_path / videos[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        dashboard.load_v02_comparison_videos(binding, ['knife'])


def test_v02_video_manifest_rejects_rebound_escape_and_sidecar_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'REPO_ROOT', tmp_path)
    manifest, binding = _v02_videos(tmp_path)
    manifest['comparisons'][0]['video']['path'] = '../outside.mp4'
    binding = _write(tmp_path, binding['path'], manifest)
    with pytest.raises(ValueError, match='invalid repository evidence path'):
        dashboard.load_v02_comparison_videos(binding, ['knife'])
    manifest, binding = _v02_videos(tmp_path)
    sidecar_binding = manifest['clips'][0]['render_sidecar']
    sidecar = json.loads((tmp_path / sidecar_binding['path']).read_text())
    sidecar['report'] = manifest['clips'][1]['sealed_inputs']['report']
    manifest['clips'][0]['render_sidecar'] = _write(tmp_path, sidecar_binding['path'], sidecar)
    binding = _write(tmp_path, binding['path'], manifest)
    with pytest.raises(ValueError, match='sealed input mismatch: report'):
        dashboard.load_v02_comparison_videos(binding, ['knife'])


def test_v02_video_manifest_rejects_v01_schema_and_wrong_pairing(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'REPO_ROOT', tmp_path)
    manifest, binding = _v02_videos(tmp_path)
    manifest['schema'] = 'graph-self-adapter.twin-video.v1'
    binding = _write(tmp_path, binding['path'], manifest)
    with pytest.raises(ValueError, match='unsupported v02 video manifest schema'):
        dashboard.load_v02_comparison_videos(binding, ['knife'])
    manifest, binding = _v02_videos(tmp_path)
    manifest['comparisons'][0]['trial'] = 7
    binding = _write(tmp_path, binding['path'], manifest)
    with pytest.raises(ValueError, match='task/trial mismatch'):
        dashboard.load_v02_comparison_videos(binding, ['knife'])


def _sensor_fixture(root,monkeypatch):
    monkeypatch.setattr(dashboard,'REPO_ROOT',root)
    def report(version,tasks,initial=False,final=False):
        runs=[];rows=[]
        for task in tasks:
            modes=('privileged_gt_control','sensor_only') if initial else ('sensor_only',)
            for mode in modes:
                name=f'{version}/{task}/{mode}'
                archive=_write(root,name+'/packet.npz',b'public packet')
                seal=_write(root,name+'/seal.json',{'schema':'graph-self-adapter.sensor-prediction-seal.v1',
                    'sealed_before_finalize_evaluator':True,'execution_variant':'sensor_transition_v04',
                    'public_sensor_packet_archive':archive})
                success=task=='cup' or mode=='privileged_gt_control'
                raw=_write(root,name+'/raw.json',{'schema':'graph-self-adapter.twin-run.v1','task':task,
                    'condition':{'trial':0,'seed':51005},'controller':'fixed_z','strict_success':success,
                    'graph_completed':success,'sensor_backend':{'judge_mode':mode,'prediction_seal':seal},
                    'original_task_outcome':{'task_outcome':{'passed':True},'numerical_safety':{'passed':True}}})
                run={'task':task,'trial':0,'judge_mode':mode,'report':raw,'strict_success':success,'graph_completed':success}
                row={'task':task,'judge_mode':mode,'strict_successes':int(success),'graph_successes':int(success),'denominator':1}
                if final:
                    run.update(sensor_prediction_seal=seal,public_sensor_packet_archive=archive,
                               original_physical_goal_passed=True,original_safety_passed=True)
                    row.update(post_seal_original_physical_goal_successes=1,post_seal_original_safety_successes=1)
                else:run['prediction_seal']=seal
                runs.append(run);rows.append(row)
        schemas={'initial':'graph-self-adapter.sensor-initial-report.v1','repair':'graph-self-adapter.sensor-repair1-report.v1','final':'graph-self-adapter.sensor-transition-final-report.v1'}
        result={'schema':schemas[version],'tasks':rows,'source_reports' if final else 'runs':runs}
        if final:result.update(denominator=len(runs),errors=0,strict_successes=sum(r['strict_success'] for r in runs),graph_successes=sum(r['graph_completed'] for r in runs))
        return result
    initial=_write(root,'initial.json',report('initial',['knife','cup','sorting'],initial=True))
    repair=_write(root,'repair.json',report('repair',['knife','cup']))
    final=report('final',['knife','cup'],final=True)
    final['source_bindings']={'initial_report':initial,'repair1_report':repair}
    final['comparison_table']=[{'task':t,'initial_gt':{'successes':1,'denominator':1},
        'initial_sensor':{'successes':int(t=='cup'),'denominator':1},
        'repair1_sensor':None if t=='sorting' else {'successes':int(t=='cup'),'denominator':1},
        'v04_sensor':None if t=='sorting' else {'successes':int(t=='cup'),'denominator':1}} for t in ('knife','cup','sorting')]
    binding=_write(root,'final.json',final)
    return {'sensor_transition_final_report':binding},final


def test_sensor_counts_are_raw_bound_and_null_is_unverified(tmp_path,monkeypatch):
    bindings,_=_sensor_fixture(tmp_path,monkeypatch)
    report=dashboard.load_sensor_transition_report(bindings)
    html=dashboard.render_sensor_transition_table(report)
    assert html.count('未验证')==2 and '0%' not in html
    assert '物理目标通过但整图未完成仍不计严格成功' in html
    assert report['comparison_table'][0]['v04_sensor']=={'successes':0,'denominator':1}


@pytest.mark.parametrize('problem',['schema','statistics','null','wrong_task','raw_mismatch','hash'])
def test_sensor_invalid_sources_rejected(tmp_path,monkeypatch,problem):
    bindings,final=_sensor_fixture(tmp_path,monkeypatch)
    if problem=='schema':final['schema']='wrong'
    elif problem=='statistics':final['comparison_table'][0]['v04_sensor']['successes']=1
    elif problem=='null':final['comparison_table'][2]['v04_sensor']={'successes':0,'denominator':1}
    elif problem=='wrong_task':final['source_reports'][0]['task']='wrong'
    elif problem=='raw_mismatch':final['source_reports'][0]['strict_success']=True
    elif problem=='hash':
        (tmp_path/'initial.json').write_text('{}')
    if problem!='hash':bindings['sensor_transition_final_report']=_write(tmp_path,'final.json',final)
    with pytest.raises(ValueError):dashboard.load_sensor_transition_report(bindings)


def test_agent_live_display_reads_actual_episode_counts_and_bound_videos(monkeypatch):
    monkeypatch.setattr(dashboard,'REPO_ROOT',ROOT)
    def binding(relative):
        path=ROOT/relative
        return {'path':relative,'sha256':dashboard.sha256(path)}
    bindings={'agent_live_final_report':binding('evaluations/agent_stage_v06/FINAL_REPORT.json'),
              'agent_live_video_manifest':binding('evaluations/agent_stage_v06/VIDEO_MANIFEST.json')}
    html=dashboard.render_agent_live(bindings)
    report=json.loads((ROOT/'evaluations/agent_stage_v06/FINAL_REPORT.json').read_text())
    for episode in report['episodes']:
        assert episode['task'] in html and f"<td>{episode['request_count']}</td>" in html
    assert html.count('<video controls')==len(report['episodes'])
    assert '不作为成功率' in html and '墙钟等待已压缩' in html
    bad={**bindings,'agent_live_video_manifest':{**bindings['agent_live_video_manifest'],'sha256':'0'*64}}
    with pytest.raises(ValueError,match='hash mismatch'):dashboard.render_agent_live(bad)
