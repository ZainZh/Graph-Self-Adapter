"""Independent RGB-only display renderer; no data/commands sent back to execution."""
import io,threading,time
from PIL import Image

def copy_graphics_snapshot(robot,snapshot):
    if not robot.lock.acquire(timeout=.02):return None
    try:
        live=robot.world.data
        snapshot.qpos[:]=live.qpos;snapshot.qvel[:]=live.qvel
        snapshot.mocap_pos[:]=live.mocap_pos;snapshot.mocap_quat[:]=live.mocap_quat
        snapshot.time=live.time
        return {'captured_wall_sec':time.time(),'captured_monotonic_sec':time.monotonic(),'physical_time_sec':float(live.time)}
    finally:robot.lock.release()

class SimPreview:
    def __init__(self,robot,publish,fps=10):
        self.robot=robot;self.publish=publish;self.fps=fps;self.stop_event=threading.Event();self.thread=None;self.error=None
    def start(self):
        self.thread=threading.Thread(target=self._run,name='RGB-display-only',daemon=True);self.thread.start()
    def _run(self):
        renderer=None
        try:
            mj=self.robot.mj;model=self.robot.world.model
            # GL context is created, used and destroyed solely on this worker.
            renderer=mj.Renderer(model,height=360,width=640);snapshot=mj.MjData(model)
            renderer.disable_segmentation_rendering();renderer.disable_depth_rendering()
            while not self.stop_event.is_set():
                start=time.monotonic();meta=copy_graphics_snapshot(self.robot,snapshot)
                if meta is not None:
                    mj.mj_forward(model,snapshot);renderer.update_scene(snapshot,camera='main')
                    rgb=renderer.render().copy();buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=80)
                    self.publish(buffer.getvalue(),meta)
                self.stop_event.wait(max(0,1/self.fps-(time.monotonic()-start)))
        except Exception as error:self.error=type(error).__name__+': '+str(error)
        finally:
            if renderer is not None:renderer.close()
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=3)
