"""第 0 课：检查环境是否装好。

逐项检查：Python 版本、各个包的版本、加载 G1 模型、跑 100 步仿真、
离屏渲染一张图、用 matplotlib 存一张曲线图、用 imageio-ffmpeg 写一段短视频、
PyTorch 是否可用（可选，第 9 课可以用它跑官方的策略文件）。每项打 √ 或 ×，最后汇总。

用法：
    python code/00_check_env.py
"""
import importlib.metadata as md
import pathlib
import platform
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
POLICY = ROOT / "models" / "unitree_rl_gym_g1" / "motion.pt"
OUT = ROOT / "output"

results = []          # (名称, 是否通过, 是否必需)


def check(name, required=True):
    """装饰器：运行一项检查，出异常就记为 ×，并打印异常的最后一行。"""
    def wrap(func):
        print(f"\n--- {name} ---")
        try:
            func()
            ok = True
        except Exception as e:  # noqa: BLE001  检查脚本要把所有错误都接住
            ok = False
            print(f"  出错：{type(e).__name__}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()
        mark = "√" if ok else "×"
        print(f"[{mark}] {name}")
        results.append((name, ok, required))
        return func
    return wrap


# ---------- 1. Python 版本 ----------
@check("Python 版本 ≥ 3.11")
def _():
    print(f"  Python {platform.python_version()}（{platform.system()} {platform.machine()}）")
    print(f"  解释器 {sys.executable}")
    if sys.version_info < (3, 11):
        raise RuntimeError("MuJoCo 4 需要 Python 3.11 及以上")


# ---------- 2. 各个包的版本 ----------
PACKAGES = [
    # (发行包名, 期望版本前缀, 说明)
    ("mujoco", "4.0.1", "元包，本身几乎没有代码"),
    ("mujoco-core", "4.0.1", "仿真核心"),
    ("mujoco-studio", "4.0.1", "Studio 查看器"),
    ("mujoco-studio-wasm", "4.0.1", "Studio 网页查看器用的前端文件"),
    ("numpy", "", "数组"),
    ("matplotlib", "", "画图"),
    ("imageio", "", "存图片、视频"),
    ("imageio-ffmpeg", "", "自带 ffmpeg，用来写 mp4"),
]


@check("各个包都已安装")
def _():
    missing = []
    for name, want, note in PACKAGES:
        try:
            ver = md.version(name)
        except md.PackageNotFoundError:
            print(f"  ×  {name:<20}{'未安装':<12}{note}")
            missing.append(name)
            continue
        bad = want and not ver.startswith(want)
        flag = "×" if bad else "√"
        tip = f"（期望 {want}）" if bad else ""
        print(f"  {flag}  {name:<20}{ver:<12}{note}{tip}")
        if bad:
            missing.append(name)
    if missing:
        raise RuntimeError("缺少或版本不对：" + ", ".join(missing))


# ---------- 3. 导入 mujoco 和 Studio ----------
@check("import mujoco / mujoco.studio")
def _():
    import mujoco
    import mujoco.studio  # noqa: F401  只是确认能导入，不打开窗口
    print(f"  mujoco.__version__ = {mujoco.__version__}")
    print(f"  mujoco 包位置 {pathlib.Path(mujoco.__file__).parent}")


# ---------- 4. 加载 G1 并跑 100 步 ----------
sim = {}              # 把模型和数据留给后面的渲染检查用


def need_sim():
    """后面几项要用第 4 项加载的模型；第 4 项没通过时给出明白的提示。"""
    if "m" not in sim:
        raise RuntimeError("上面“加载 G1 模型”那一项没通过，这一项跳过")
    return sim["m"], sim["d"]


@check("加载 G1 模型并仿真 100 步")
def _():
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    print(f"  模型 {XML.relative_to(ROOT)}")
    print(f"  nq={m.nq}  nv={m.nv}  nu={m.nu}  时间步长 {m.opt.timestep} s")

    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)   # 从站立关键帧开始
    z0 = d.qpos[2]
    heights = [z0]
    t0 = time.perf_counter()
    for _ in range(100):
        mujoco.mj_step(m, d)
        heights.append(d.qpos[2])
    wall = time.perf_counter() - t0
    print(f"  仿真时间 {d.time:.3f} s，墙钟用时 {wall * 1000:.1f} ms"
          f"（约 {d.time / wall:.0f} 倍实时）")
    print(f"  骨盆高度 {z0:.3f} m → {d.qpos[2]:.3f} m")
    if not np.all(np.isfinite(d.qpos)):
        raise RuntimeError("qpos 里出现了 NaN/Inf")
    sim.update(m=m, d=d, heights=np.array(heights))


# ---------- 5. 离屏渲染 ----------
@check("离屏渲染（mujoco.Renderer）")
def _():
    import imageio.v3 as iio
    import mujoco

    m, d = need_sim()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.body("pelvis").xpos       # 镜头对准骨盆
    cam.distance, cam.azimuth, cam.elevation = 3.0, 140, -15
    with mujoco.Renderer(m, height=480, width=640) as r:
        r.update_scene(d, camera=cam)
        img = r.render()
    OUT.mkdir(exist_ok=True)
    path = OUT / "00_check_render.png"
    iio.imwrite(path, img)
    print(f"  图像 {img.shape} {img.dtype}，像素均值 {img.mean():.1f}，已保存 {path.relative_to(ROOT)}")
    if img.std() < 1:
        raise RuntimeError("图像几乎是纯色，渲染可能没成功")


# ---------- 6. matplotlib（Agg 后端，只存文件，不弹窗） ----------
@check("matplotlib 存 PNG")
def _():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    need_sim()
    h = sim["heights"]
    t = np.arange(len(h)) * sim["m"].opt.timestep
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t, h)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pelvis z (m)")
    fig.tight_layout()
    path = OUT / "00_check_plot.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  后端 {matplotlib.get_backend()}，已保存 {path.relative_to(ROOT)}")


# ---------- 7. imageio-ffmpeg 写一段 1 秒的视频 ----------
@check("imageio-ffmpeg 写 mp4")
def _():
    import imageio.v2 as iio2
    import imageio_ffmpeg
    import mujoco

    m, _ = need_sim()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    print(f"  ffmpeg 程序 {pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()).name}")
    path = OUT / "00_check_video.mp4"
    fps = 25
    steps_per_frame = round(1 / fps / m.opt.timestep)   # 每帧之间跑几步
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 140, -15
    with mujoco.Renderer(m, height=240, width=320) as r, \
            iio2.get_writer(path, fps=fps, macro_block_size=8) as w:
        for _ in range(fps):                             # 25 帧 = 1 秒
            for _ in range(steps_per_frame):
                mujoco.mj_step(m, d)
            cam.lookat[:] = d.body("pelvis").xpos
            r.update_scene(d, camera=cam)
            w.append_data(r.render())
    print(f"  {fps} 帧，{path.stat().st_size / 1024:.0f} KB，已保存 {path.relative_to(ROOT)}")


# ---------- 8. PyTorch（可选） ----------
@check("PyTorch", required=False)
def _():
    try:
        import torch
    except ImportError:
        print("  没有安装 PyTorch。除第 9 课外都用不到它；第 9 课没有它时，")
        print("  脚本会自动改用 numpy 计算同一个策略网络。想和官方脚本一样用 torch 运行，")
        print("  按第 0 课 0.6 节离线装好 CPU 版。")
        raise
    print(f"  torch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}")
    policy = torch.jit.load(str(POLICY), map_location="cpu")   # 宇树的行走策略
    out = policy(torch.zeros(1, 47))                          # 47 维观测 → 12 维动作
    print(f"  已加载 {POLICY.name}，输入 (1, 47) → 输出 {tuple(out.shape)}")


# ---------- 汇总 ----------
print("\n========== 汇总 ==========")
for name, ok, required in results:
    tag = "" if required else "（可选）"
    print(f"[{'√' if ok else '×'}] {name}{tag}")

failed = [n for n, ok, req in results if req and not ok]
if failed:
    print(f"\n有 {len(failed)} 项必需检查没通过，对照第 0 课末尾的“如果报错”表格排查。加 -v 可看完整报错。")
    sys.exit(1)
print("\n必需项全部通过，可以开始第 1 课了。")
