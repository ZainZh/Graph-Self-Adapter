"""Read-only presentation of a task's recorded public camera frames and stage state."""
import io,json,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from PIL import Image

PAGE='<!doctype html><meta charset="utf-8"><title>Task live view</title>\n<style>body{background:#151515;color:#eee;font:17px sans-serif;margin:20px}.camera{position:relative;display:inline-block;min-width:580px;min-height:220px;background:#222}img{display:block;max-width:95vw;max-height:75vh}.overlay{position:absolute;top:0;left:0;right:0;background:rgba(0,0,0,.8);padding:12px 16px;line-height:1.6;pointer-events:none}.badge{font-weight:bold;font-size:24px;color:#ffd166}.executing{color:#75e6a4}.finished{color:#aaa}pre{white-space:pre-wrap}</style>\n<h2 id="title">模拟任务执行</h2><div class="camera"><img id="frame" hidden><div class="overlay"><div id="badge" class="badge">正在初始化</div><div id="timing">运行时长 — · 当前阶段 —</div><div id="node">当前节点 —</div><div id="age">等待第一帧</div><div id="msg">等待模拟器产生第一帧…</div></div></div><details><summary>详细状态</summary><pre id="state"></pre></details>\n<script>let busy=false;function duration(x){if(x==null)return \'—\';x=Math.floor(x);return String(Math.floor(x/3600)).padStart(2,\'0\')+\':\'+String(Math.floor(x/60)%60).padStart(2,\'0\')+\':\'+String(x%60).padStart(2,\'0\')}async function update(){if(busy)return;busy=true;try{let r=await fetch(\'/status.json?t=\'+Date.now());if(!r.ok)throw Error(\'状态接口 HTTP \'+r.status);let s=await r.json();document.getElementById(\'title\').textContent=s.task+\' · 模拟任务执行\';let b=document.getElementById(\'badge\');b.textContent=s.state_label;b.className=\'badge \'+s.execution_state;document.getElementById(\'timing\').textContent=\'运行时长 \'+duration(s.elapsed_wall_sec)+\' · 当前阶段 \'+duration(s.stage_elapsed_wall_sec);document.getElementById(\'node\').textContent=\'当前节点 \'+(s.node.stage_id||\'尚未进入节点\')+\' / \'+(s.node.phase||\'—\');document.getElementById(\'age\').textContent=s.frame_age_wall_sec==null?\'等待第一帧\':\'距本画面采集 \'+s.frame_age_wall_sec.toFixed(1)+\' 秒（非连续视频）\';document.getElementById(\'msg\').textContent=s.message;document.getElementById(\'state\').textContent=JSON.stringify(s,null,2);if(s.frame_available){let f=document.getElementById(\'frame\');f.src=\'/frame.jpg?t=\'+Date.now();f.hidden=false}}catch(e){document.getElementById(\'badge\').textContent=\'查看连接中断\';document.getElementById(\'msg\').textContent=String(e)}finally{busy=false}}setInterval(update,500);update()</script>'

class TaskLiveView:
    def __init__(self,task,port):
        self.task=task;self.port=port;self.owner=None;self.run=None;self.finished=False
        self.server=None;self.lock=threading.Lock();self.jpeg=None;self.frame_path=None;self.frame_mtime=None;self.last_key=None
        self.thread=None;self.stop_event=threading.Event();self.viewer_started=time.time();self.ended_at=None
    def set_owner(self,owner):self.owner=Path(owner)
    def attach(self,run):self.run=Path(run)
    def _discover(self):
        if self.run is None and self.owner:
            runs=list((self.owner/'runs').glob('*'))
            if runs:self.run=max(runs,key=lambda p:p.stat().st_mtime)
    def refresh(self):
        self._discover()
        if not self.run:return
        # Read only explicitly public RGB artifacts, never state trace/private evaluation.
        frames=list(self.run.glob('INITIAL*RGB*.png'))
        evidence=self.run/'SAM_MASK_EVIDENCE'
        if evidence.exists():frames+=list(evidence.rglob('RGB_*.png'))+list(evidence.rglob('RGB_*.jpg'))
        live=self.run/'LIVE/LATEST_RGB.jpg'
        if live.exists():frames.append(live)
        if not frames:return
        frame=max(frames,key=lambda p:p.stat().st_mtime);key=(str(frame),frame.stat().st_mtime_ns)
        if key==self.last_key:return
        try:
            with Image.open(frame) as im:
                stream=io.BytesIO();im.convert('RGB').save(stream,format='JPEG',quality=85)
            with self.lock:self.jpeg=stream.getvalue();self.frame_path=str(frame);self.frame_mtime=frame.stat().st_mtime;self.last_key=key
        except (OSError,ValueError):return
    def status(self):
        node={};waiting=None
        if self.run:
            p=self.run/'NODE_STATUS.json'
            if p.exists():
                try:node=json.loads(p.read_text())
                except (OSError,ValueError):pass
            # Only outstanding public judging requests, no verdict fabrication.
            requests=list(self.run.rglob('REQUEST.json'))
            if not node and any(self.run.glob('INITIAL*RGB*.png')) and not any(self.run.glob('INITIAL*BOXES.json')):
                waiting='等待实际 Agent 回写首次物体定位；机器人尚未开始运动'
            if (self.run/'PREDICTIONS_SEALED.json').exists():self.finished=True
            for p in sorted(requests,key=lambda p:p.stat().st_mtime,reverse=True):
                try:
                    d=json.loads(p.read_text());reply=d.get('reply_path')
                    if reply and not Path(reply).exists():waiting='等待实际 Agent 回写定位／阶段判断';break
                except (OSError,ValueError):pass
        available=self.jpeg is not None
        now=time.time();started=self.viewer_started;stage_started=None
        if self.run:
            initial=list(self.run.glob('INITIAL*RGB*.png'))
            if initial:started=min(p.stat().st_mtime for p in initial)
            np=self.run/'NODE_STATUS.json'
            if np.exists():stage_started=np.stat().st_mtime
        if self.finished:
            if self.ended_at is None:
                seal=self.run/'PREDICTIONS_SEALED.json' if self.run else None
                self.ended_at=seal.stat().st_mtime if seal and seal.exists() else now
            state='finished';label='已结束 · 不是正在执行'
        elif waiting:state='waiting';label='等待 Agent 判断 · 未继续执行'
        elif not available:state='initializing';label='正在初始化 · 尚无画面'
        elif not node:state='unknown';label='尚未进入任务节点 · 执行状态待确认'
        else:
            intent=self.run/'COMMAND_INTENT.json'
            recent=intent.exists() and now-intent.stat().st_mtime<5
            state='executing' if recent else 'unknown'
            label='正在执行节点 · 最近有动作下发' if recent else '节点处理中 · 暂无近期动作下发'
        cutoff=self.ended_at if self.finished else now
        elapsed=max(0,cutoff-started)
        stage_elapsed=None if stage_started is None else max(0,cutoff-stage_started)
        message=('本次程序已结束；显示最后采集画面' if self.finished else waiting or ('执行中；显示最近采集的公开相机帧' if available else '等待模拟器产生第一帧'))
        return {'task':self.task,'message':message,'node':node,'frame_available':available,
            'execution_state':state,'state_label':label,'elapsed_wall_sec':round(elapsed,1),
            'elapsed_origin':'first retained initial RGB frame; viewer start until first frame',
            'stage_elapsed_wall_sec':None if stage_elapsed is None else round(stage_elapsed,1),
            'frame_age_wall_sec':None if self.frame_mtime is None else round(max(0,time.time()-self.frame_mtime),1),
            'frame_source':self.frame_path,'run_output':None if self.run is None else str(self.run),
            'read_only':True,'extra_camera_captures':0,'note':'页面每0.5秒检查；图像只在执行器实际采集相机时更新。不是连续视频，也不是离线轨迹回放。'}
    def start(self):
        view=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path=self.path.split('?',1)[0]
                if path=='/':body=PAGE.encode();kind='text/html; charset=utf-8';code=200
                elif path=='/status.json':body=json.dumps(view.status(),ensure_ascii=False).encode();kind='application/json; charset=utf-8';code=200
                elif path=='/frame.jpg':
                    with view.lock:body=view.jpeg
                    if body is None:body=b'';code=204
                    else:code=200
                    kind='image/jpeg'
                else:self.send_error(404);return
                self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            def log_message(self,*_):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',self.port),Handler)
        def poll():
            while not self.stop_event.wait(.5):self.refresh()
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        return f'http://127.0.0.1:{self.server.server_address[1]}/'
    def close(self):
        self.stop_event.set()
        if self.server:self.server.shutdown();self.server.server_close()
        if self.thread:self.thread.join(timeout=2)
