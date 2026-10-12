"""MJPEG view of live simulation; original status overlay with independent frames."""
import json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from task_live_view_v27 import TaskLiveView,PAGE
ACTIVE_VIEW=None
STREAM_PAGE=PAGE.replace("f.src='/frame.jpg?t='+Date.now();f.hidden=false", "if(!f.dataset.streaming){f.src='/stream.mjpg';f.dataset.streaming='true'}f.hidden=false").replace('（非连续视频）','（实时预览）')

class StreamingTaskView(TaskLiveView):
    def __init__(self,*args,**kwargs):
        global ACTIVE_VIEW
        super().__init__(*args,**kwargs);ACTIVE_VIEW=self;self.preview=None;self.sequence=0;self.preview_metadata=None;self.condition=threading.Condition(self.lock)
    def publish(self,jpeg,metadata):
        with self.condition:
            self.jpeg=jpeg;self.frame_mtime=metadata['captured_wall_sec'];self.frame_path='live simulation main-camera RGB, independent display-only renderer'
            self.preview_metadata=metadata;self.sequence+=1;self.condition.notify_all()
    def refresh(self):
        self._discover()
        if self.preview is None:super().refresh()
    def status(self):
        s=super().status();s.update(stream_preview=True,preview_frames=self.sequence,preview_target_fps=10,
            preview_metadata=self.preview_metadata,preview_error=None if self.preview is None else self.preview.error,
            extra_camera_captures=0,extra_display_renders=self.sequence,
            note='独立RGB实时预览，目标10fps；不经SAM，不进入控制/判断输入，不写回物理状态。')
        return s
    def start(self):
        view=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path=self.path.split('?',1)[0]
                if path=='/stream.mjpg':
                    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
                    seen=-1
                    try:
                        self.connection.settimeout(3)
                        while not view.stop_event.is_set():
                            with view.condition:
                                view.condition.wait_for(lambda:view.sequence!=seen or view.stop_event.is_set(),timeout=.5)
                                if view.jpeg is None or view.sequence==seen:continue
                                jpeg=view.jpeg;seen=view.sequence
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError,TimeoutError,OSError):pass
                    return
                if path=='/':body=STREAM_PAGE.encode();kind='text/html; charset=utf-8'
                elif path=='/status.json':body=json.dumps(view.status(),ensure_ascii=False).encode();kind='application/json; charset=utf-8'
                elif path=='/frame.jpg':
                    with view.lock:body=view.jpeg
                    if body is None:self.send_response(204);self.end_headers();return
                    kind='image/jpeg'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            def log_message(self,*_):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',self.port),Handler)
        def poll():
            while not self.stop_event.wait(.5):self.refresh()
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        return f'http://127.0.0.1:{self.server.server_address[1]}/'
    def close(self):
        if self.preview:self.preview.close()
        if self.finished:
            print('任务已结束；实时页面保留最后一帧。Ctrl+C关闭查看服务。',flush=True)
            try:
                while True:time.sleep(1)
            except KeyboardInterrupt:pass
        with self.condition:self.stop_event.set();self.condition.notify_all()
        super().close()

def install():
    import task_live_view_v27
    task_live_view_v27.TaskLiveView=StreamingTaskView
    import sim_robot_io
    start=sim_robot_io.SimRobot.start;close=sim_robot_io.SimRobot.close
    def start_preview(robot):
        start(robot)
        if ACTIVE_VIEW is not None:
            from sim_preview_v33 import SimPreview
            preview=SimPreview(robot,ACTIVE_VIEW.publish);robot._display_preview=preview;ACTIVE_VIEW.preview=preview;preview.start()
    def close_preview(robot):
        preview=getattr(robot,'_display_preview',None)
        if preview:preview.close()
        return close(robot)
    sim_robot_io.SimRobot.start=start_preview;sim_robot_io.SimRobot.close=close_preview
