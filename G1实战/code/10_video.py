"""第 10 课（上）：离屏渲染，录一段 G1 站立被推的 MP4。

用法：
    python 10_video.py                  # 跟踪相机，输出 output/10_push_track.mp4
    python 10_video.py --camera free    # 固定的自由相机，输出 output/10_push_free.mp4
    python 10_video.py --view           # 不录视频，打开查看器实时看（循环播放）
"""
import argparse
import pathlib
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"

# 推力日程：(开始时刻 s, 持续 s, 世界坐标系里的力 N)。G1 面朝 +x。
PUSHES = [
    (1.0, 0.1, np.array([0.0, 120.0, 0.0])),     # 从右侧往左推 120 N：晃一下，站住
    (3.0, 0.1, np.array([-150.0, 0.0, 0.0])),    # 从正前方往后推 150 N：仰面倒下
]
T_END = 5.0


def current_force(t):
    """返回 t 时刻施加在躯干上的力（没有推力时为 0）。"""
    for t0, dur, f in PUSHES:
        if t0 <= t < t0 + dur:
            return f
    return np.zeros(3)


def add_arrow(scn, d, body_id, force, scale=0.004):
    """往场景里追加一个红色箭头，箭头指向受力点，长度与力成正比。"""
    if np.linalg.norm(force) < 1e-9 or scn.ngeom >= scn.maxgeom:
        return
    tip = d.xipos[body_id].copy()               # 躯干质心（力就作用在这里）
    tail = tip - force * scale                  # 箭尾在外面，箭头扎向身体
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.array([1, 0.1, 0.1, 1], dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, tail, tip)
    scn.ngeom += 1


def make_camera(kind, m):
    """两种相机：跟踪相机跟着骨盆走；自由相机固定在世界里某处。"""
    cam = mujoco.MjvCamera()
    if kind == "track":
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = m.body("pelvis").id    # 跟踪哪个 body（看的是它所在子树的质心）
        cam.distance = 2.2                       # 相机到注视点的距离 m
        cam.azimuth = 135.0                      # 水平方位角（度）
        cam.elevation = -15.0                    # 俯仰角（度），负数表示从上往下看
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.6]          # 注视点固定在世界坐标里
        cam.distance = 3.5
        cam.azimuth = 90.0
        cam.elevation = -10.0
    return cam


def record(args):
    m = mujoco.MjModel.from_xml_path(str(XML))
    # 离屏缓冲区默认只有 640×480，要更大的画面先把它调大（等价于 XML 里 <visual><global offwidth offheight>）
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, args.height)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    torso = m.body("torso_link").id

    cam = make_camera(args.camera, m)
    OUT.mkdir(exist_ok=True)
    tag = args.camera if args.fps == 30 else f"{args.camera}_{args.fps}fps"   # 换帧率时另存一个文件，方便比较
    path = OUT / f"10_push_{tag}.mp4"
    # 截几帧拼成一张图，方便不打开视频也能看。1.05 s、3.05 s 正在推，图上能看到箭头
    snap_times = [0.9, 1.05, 3.05, 4.5]
    snaps = []

    t_start = time.perf_counter()
    nframe = 0
    with mujoco.Renderer(m, height=args.height, width=args.width) as renderer, \
            imageio.get_writer(path, fps=args.fps, codec="libx264", quality=8) as writer:
        tilt_max = 0.0
        while d.time < T_END:
            d.xfrc_applied[torso, :3] = current_force(d.time)   # 先定好这一步的力，画面上的箭头才对得上
            # 仿真 500 Hz，视频 30 fps：仿真时间追上下一帧的时刻才渲染一帧
            if d.time >= nframe / args.fps:
                renderer.update_scene(d, camera=cam)
                add_arrow(renderer.scene, d, torso, d.xfrc_applied[torso, :3])
                frame = renderer.render()        # (H, W, 3) uint8
                writer.append_data(frame)
                if len(snaps) < len(snap_times) and d.time >= snap_times[len(snaps)]:
                    snaps.append(frame[::2, ::2])
                nframe += 1
            mujoco.mj_step(m, d)
            # 躯干 z 轴与竖直方向的夹角：xmat 第 9 个元素是躯干 z 轴在世界 z 上的分量
            tilt_max = max(tilt_max, np.degrees(np.arccos(np.clip(d.xmat[torso][8], -1, 1))))
            if round(d.time / m.opt.timestep) % 500 == 0:   # 每 1 s 打印一次
                print(f"t={d.time:.1f} s  骨盆高度 {d.qpos[2]:.3f} m  这 1 s 里躯干最大倾角 {tilt_max:5.1f}°")
                tilt_max = 0.0
    wall = time.perf_counter() - t_start

    png = OUT / f"10_push_{tag}_frames.png"
    imageio.imwrite(png, np.hstack(snaps))
    print(f"相机 {args.camera}，{args.width}×{args.height}，{args.fps} fps，共 {nframe} 帧"
          f"（仿真 {T_END:.0f} s），耗时 {wall:.1f} s")
    print(f"视频 {path}（{path.stat().st_size / 1024:.0f} KB）")
    print(f"截图 {png}")


def view(args):
    """实时查看：用 launch_passive，自己控制仿真循环。"""
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    key = m.key("stand").id
    mujoco.mj_resetDataKeyframe(m, d, key)
    torso = m.body("torso_link").id
    with mujoco.viewer.launch_passive(m, d) as viewer:
        cam = make_camera(args.camera, m)
        with viewer.lock():
            viewer.cam.type = cam.type
            viewer.cam.trackbodyid = cam.trackbodyid
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance = cam.distance
            viewer.cam.azimuth = cam.azimuth
            viewer.cam.elevation = cam.elevation
        while viewer.is_running():
            step_start = time.time()
            if d.time >= T_END:                  # 播完一遍就从头再来
                mujoco.mj_resetDataKeyframe(m, d, key)
            d.xfrc_applied[torso, :3] = current_force(d.time)
            mujoco.mj_step(m, d)
            with viewer.lock():
                viewer.user_scn.ngeom = 0        # 每帧清空自定义几何，再画这一帧的箭头
                add_arrow(viewer.user_scn, d, torso, d.xfrc_applied[torso, :3])
            viewer.sync()
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", choices=["track", "free"], default="track")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--view", action="store_true", help="打开查看器实时看，不录视频")
    a = p.parse_args()
    view(a) if a.view else record(a)
