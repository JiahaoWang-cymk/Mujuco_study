"""把脚本自己的仿真实时显示在网页里（MuJoCo Studio 的网页查看器）。

课文页“实验台”右侧的查看器就是它：脚本照常 mj_step，每步调 sync()，浏览器里看到的是同一个仿真。
和 --view 的桌面窗口一样是“被动”模式：仿真循环在脚本里，查看器只负责显示和鼠标交互。

    from webview import open_web
    with open_web(m, d, port=8790) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync(m, d)

Studio 默认监听所有网卡，这里改成只监听本机（127.0.0.1）。
"""
import contextlib
import os
import threading

import mujoco
from mujoco.studio import launch_passive, messages, plugin, viewer_protocol, web_viewer
from mujoco.studio.web import web_server

READY = "网页查看器已就绪"
_bind = web_server.bind_public_socket


def _bind_loopback(host, port=0):
    return _bind("127.0.0.1", port)


class _Camera(plugin.ViewerPlugin):
    """载入模型时把镜头对准机器人，并跟随 track_body（被动模式下查看器不会自己设镜头）。"""

    def __init__(self, track_body, distance, azimuth, elevation):
        self.track_body, self.distance, self.azimuth, self.elevation = track_body, distance, azimuth, elevation

    @messages.handler
    def _on_post_model(self, event: messages.PostModelEvent) -> None:
        cam, m = self.viewer.camera, event.model
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.track_body)
        if body >= 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = body
        else:
            mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.azimuth, cam.elevation = self.distance, self.azimuth, self.elevation


def _exit_with_parent():
    """由资料包的本地服务（serve.py）启动时，服务一退出就跟着结束，不留后台进程。"""
    pid = int(os.environ.get("LAB_PARENT_PID", "0"))
    if not pid or os.name != "nt":
        return
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE
    if not handle:
        os._exit(0)

    def watch():
        kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)          # 等到父进程结束
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


@contextlib.contextmanager
def open_web(m, d, port, title="G1", track_body="pelvis", distance=3.0, azimuth=150.0, elevation=-12.0):
    _exit_with_parent()
    web_server.bind_public_socket = _bind_loopback
    web_viewer._print_url_banner = lambda host, p: print(f"网页查看器：http://127.0.0.1:{p}/", flush=True)
    config = viewer_protocol.ViewerConfig(gfx="web", http_port=port, title=title)
    cam = _Camera(track_body, distance, azimuth, elevation)
    with launch_passive.launch(config, plugins=[cam]) as handle:
        handle.set_model(m, data=d)
        handle.sync(m, d)
        # 端口在 set_model 之前就打开了；浏览器要等模型交给查看器之后再连，否则可能停在黑屏。
        # serve.py 看到这一行才让课文页载入画面。
        print(READY, flush=True)
        yield handle
