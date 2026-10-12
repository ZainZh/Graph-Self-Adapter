"""Pause-only file bridge for the current Root Agent, never an inference service.

The caller owns physics. Only an explicit bounded hold callback advances it;
waiting, panel publication and reply validation never issue robot commands.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from PIL import Image, ImageDraw
from prove.task_graph.agent_stage_judge import AgentStageJudge, REQUEST_SCHEMA, digest, validate_request

VARIANT = 'agent_live_v06'
APPROVAL_KEYS = {'schema', 'request_id', 'request_sha256', 'reply_sha256', 'actual_root_reply_identity'}

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class HandoffError(RuntimeError):
    pass


class PauseBroker:
    def __init__(self, root, *, task, graph_sha256, root_identity, wall_timeout_sec=600.,
                 max_unknown_holds=2, hold_sec=.25, max_requests=16, clock=time.monotonic, sleeper=time.sleep):
        if not root_identity or not 0 < wall_timeout_sec <= 3600:
            raise ValueError('explicit Root identity and bounded wall timeout required')
        if max_unknown_holds != 2 or hold_sec != .25:
            raise ValueError('v06 unknown contract is exactly two quarter-second holds')
        self.root = Path(root); self.task = task; self.graph_sha256 = graph_sha256
        self.root_identity = root_identity; self.wall_timeout_sec = wall_timeout_sec
        self.max_unknown_holds = max_unknown_holds; self.hold_sec = hold_sec
        if max_requests != 16: raise ValueError("v06 per-task budget is sixteen requests")
        self.max_requests = max_requests
        self.clock = clock; self.sleeper = sleeper; self.judge = AgentStageJudge()
        self.records = []; self.sequence = 0; self.unknown_holds_by_gate = {}

    def ask(self, stage, fact_ids, *, capture, state_hash, physical_time, hold, counters):
        """capture returns causal public RGB/proprio only; state_hash is opaque."""
        gate = (stage['stage_id'], tuple(fact_ids))
        gate_started = self.clock()
        for retry in range(self.max_unknown_holds + 1):
            if self.sequence >= self.max_requests:
                raise HandoffError("per-task actual Root request budget exhausted")
            before = state_hash(); now = physical_time(); before_counts = dict(counters())
            frames, proprio = capture()
            request_id = f'{self.task}_{self.sequence:05d}_{uuid.uuid4().hex}'
            self.sequence += 1
            folder = self.root / request_id
            pending = self.root / ('.' + request_id)
            pending.mkdir(parents=True, exist_ok=False)
            frame_rows = []; panel_images = []
            if not 1 <= len(frames) <= 6:
                raise HandoffError('one to six actual causal RGB frames required')
            for index, (timestamp, rgb, view) in enumerate(frames):
                name = f'frame_{index:02d}.png'
                img = Image.fromarray(rgb); img.save(pending / name)
                frame_rows.append({'frame_id': f'f{index}', 'path': name, 'sha256': digest(pending / name),
                                   'physical_timestamp_sec': timestamp, 'view': view})
                tile = img.copy().convert('RGB'); draw = ImageDraw.Draw(tile)
                draw.text((8, 8), f'f{index} {view} t={timestamp:.6f}', fill='yellow'); panel_images.append(tile)
            panel = Image.new('RGB', (max(i.width for i in panel_images), sum(i.height for i in panel_images)))
            y = 0
            for tile in panel_images:
                panel.paste(tile, (0, y)); y += tile.height
            panel.save(pending / 'PANEL.png')
            request = {'schema': REQUEST_SCHEMA, 'request_id': request_id, 'task': self.task,
                       'stage_id': stage['stage_id'], 'graph_sha256': self.graph_sha256,
                       'observed_physical_timestamp_sec': now, 'expires_physical_timestamp_sec': now + .25,
                       'stage_contract': {'skill': stage['phase'], 'actor': stage['actor'],
                           'object': stage['object'], 'target': stage['target'],
                           'goal': stage['goal_relation'], 'termination': json.dumps(stage.get('completion_predicates', []), sort_keys=True)},
                       'frames': frame_rows, 'public_proprio': proprio, 'requested_fact_ids': list(fact_ids),
                       'input_provenance': 'RGB_only_plus_optional_public_robot_feedback; simulator_proxy_not_hardware'}
            atomic_json(pending / 'REQUEST.json', request)
            validate_request(request, now_sec=now, task=self.task, stage_id=stage['stage_id'],
                             graph_sha256=self.graph_sha256, frame_root=pending)
            if state_hash() != before or physical_time() != now or counters() != before_counts:
                raise HandoffError('capture modified the paused integration state')
            os.replace(pending, folder)
            status = {'schema': 'graph-self-adapter.agent-live-status.v1', 'state': 'paused_waiting_actual_root',
                      'request_id': request_id, 'request_path': str(folder / 'REQUEST.json'),
                      'panel_path': str(folder / 'PANEL.png'), 'request_sha256': digest(folder / 'REQUEST.json'),
                      'physical_time_sec': now, 'state_sha256': before, 'stage_id': stage['stage_id'],
                      'unknown_hold_index': retry, 'counters': before_counts}
            atomic_json(self.root / 'STATUS.json', status)
            started = self.clock(); reply = folder / 'ROOT_REPLY.json'; approval = folder / 'ROOT_APPROVAL.json'
            try:
                while not (reply.exists() and approval.exists()):
                    if state_hash() != before or physical_time() != now or counters() != before_counts:
                        raise HandoffError('physics changed while waiting for Root')
                    if self.clock() - gate_started >= self.wall_timeout_sec:
                        raise HandoffError('Root handoff wall timeout')
                    self.sleeper(.1)
                approved = json.loads(approval.read_text())
                if (set(approved) != APPROVAL_KEYS or approved['schema'] != 'graph-self-adapter.root-approval.v1'
                    or approved['request_id'] != request_id or approved['request_sha256'] != digest(folder / 'REQUEST.json')
                    or approved['actual_root_reply_identity'] != self.root_identity):
                    raise HandoffError('independent Root approval identity or binding mismatch')
                result = self.judge.consume(folder / 'REQUEST.json', reply, task=self.task,
                    stage_id=stage['stage_id'], graph_sha256=self.graph_sha256, now_sec=physical_time(),
                    approved_reply_sha256=approved['reply_sha256'])
                if request_id not in self.judge.consumed:
                    raise HandoffError('malformed or unapproved Root handoff: ' + result['reason'])
                after = state_hash()
                if after != before or physical_time() != now or counters() != before_counts:
                    raise HandoffError('integration state changed across actual Root reply')
                record = {**status, 'state': 'reply_consumed', 'decision': result,
                          'pause_before_sha256': before, 'pause_after_sha256': after,
                          'resume_physical_time_sec': physical_time(), 'wall_wait_sec': self.clock() - started,
                          'reply_sha256': digest(reply), 'approval_sha256': digest(approval),
                          'actual_root_reply_identity': approved['actual_root_reply_identity']}
                self.records.append(record); atomic_json(self.root / 'STATUS.json', record)
            except (OSError, ValueError, KeyError, TypeError, HandoffError) as exc:
                failure = {**status, 'state': 'fail_closed', 'error': str(exc),
                           'pause_after_sha256': state_hash(), 'resume_physical_time_sec': physical_time()}
                self.records.append(failure); atomic_json(self.root / 'STATUS.json', failure)
                raise HandoffError(str(exc)) from exc
            if result['decision'] != 'unknown':
                return result
            used_holds = self.unknown_holds_by_gate.get(gate, 0)
            if used_holds >= self.max_unknown_holds:
                raise HandoffError('unknown exhausted two bounded fresh-image holds')
            self.unknown_holds_by_gate[gate] = used_holds + 1
            hold(self.hold_sec)
        raise AssertionError('unreachable')

    def seal(self, path, *, bindings, counters):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, {'schema': 'graph-self-adapter.agent-live-predictions.v1',
                          'sealed_before_private_evaluator': True, 'execution_variant': VARIANT,
                          'bindings': bindings, 'counters': counters, 'actual_root_reply_identity': self.root_identity,
                          'requests': self.records})
        return digest(path)
