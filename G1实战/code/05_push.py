"""第 5 课：G1 扛得住多大的侧向推力？

默认无界面运行，依次做 4 组实验，打印结果表，CSV 和图写到 output/：
  A. 站姿对比：直腿（stand 关键帧）与屈膝（脚本里自己算出来）两种站姿的临界推力
  B. 摩擦：改脚底小球的 friction，看临界推力和脚的滑动
  C. 接触软硬：改脚底小球的 solref / solimp，看穿透深度和临界推力
  D. 数值设置：timestep、积分器、摩擦锥会不会改变结论

推法（全课统一）：t = 1.0 s 起，在躯干 torso_link 的质心上沿 +y（机器人左侧）
施加恒定的力 F，持续 0.2 s，然后撤掉，一直看到 t = 4.0 s。
摔倒判据：这 4 s 内骨盆高度低于 0.5 m，或骨盆竖直轴倾斜超过 45°，任一条满足即算摔倒。

用法：
  python 05_push.py                           # 全部实验（无界面，约 1–1.5 分钟）
  python 05_push.py --once --force 90         # 只推一次（无界面，几秒）：打印结果，画倾角曲线和 5 张截图
  python 05_push.py --once --force 90 --critical   # 推一次，再二分求这组设置下的临界推力
  python 05_push.py --view --force 90         # 查看器里看直腿站姿被推 90 N
  python 05_push.py --web 8790 --force 90     # 同上，但显示在网页里：浏览器打开 http://127.0.0.1:8790/
  python 05_push.py --view --crouch --force 70 --mu 0.05   # 屈膝 + 脚底摩擦 0.05
--force、--crouch、--mu、--timeconst、--dur 对 --once、--view、--web 都有效。
"""
import argparse
import csv
import pathlib
import time

import matplotlib

matplotlib.use("Agg")                        # 只存图，不弹窗
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

PUSH_BODY = "torso_link"   # 推躯干
PUSH_DIR = np.array([0.0, 1.0, 0.0])   # 推力方向（世界坐标系单位向量），+y 是机器人左侧
PUSH_T0 = 1.0              # 开始推的时刻 (s)
PUSH_DUR = 0.2             # 推多久 (s)
T_END = 4.0                # 仿真到几秒
FALL_Z = 0.5               # 骨盆高度低于它算摔倒 (m)，站立时约 0.79 m
FALL_TILT = 45.0           # 骨盆倾角超过它算摔倒 (度)
F_MAX = 300.0              # 二分搜索的上界 (N)
F_TOL = 0.5                # 二分搜索停止的精度 (N)
KNEE = 1.0                 # 屈膝站姿的膝角 (rad)，约 57°

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 模型与姿势
def load_model():
    return mujoco.MjModel.from_xml_path(str(XML))


def foot_geoms(m):
    """两只脚底的 8 个小球（class="foot" 的 sphere）。"""
    feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
    return np.array([g for g in range(m.ngeom)
                     if m.geom_bodyid[g] in feet and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE])


def stand_pose(m):
    return m.key("stand").qpos.copy()


def crouch_pose(m, knee=KNEE):
    """设计一个屈膝站姿，返回 (qpos, 髋角, 踝角)。

    思路（骨盆保持竖直，只动两腿的 hip_pitch、knee、ankle_pitch）：
      1. 三个 pitch 关节的转轴都平行（都绕 y 轴），脚相对骨盆的俯仰角 = 三者之和。
         令 ankle = -(hip + knee)，脚就和骨盆一样“平”，骨盆竖直时脚掌就平放在地上。
      2. 膝角固定为 knee，髋角 hip 用二分法求：让整机质心的 x 正好落在 8 个脚底小球的
         正中间（支撑区的中心）。
      3. 最后把骨盆抬高或放低，让脚底小球的高度和 stand 关键帧里一样，刚好踩在地上。
    """
    d = mujoco.MjData(m)
    feet = foot_geoms(m)
    q_stand = stand_pose(m)
    z_foot = None

    def place(hip):
        q = q_stand.copy()
        for side in ("left", "right"):
            q[m.joint(f"{side}_hip_pitch_joint").qposadr[0]] = hip
            q[m.joint(f"{side}_knee_joint").qposadr[0]] = knee
            q[m.joint(f"{side}_ankle_pitch_joint").qposadr[0]] = -(hip + knee)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        q[2] += z_foot - d.geom_xpos[feet, 2].min()     # 第 3 步：调骨盆高度
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)                          # 算 subtree_com
        return q, d.subtree_com[0, 0] - d.geom_xpos[feet, 0].mean()

    d.qpos[:] = q_stand
    mujoco.mj_kinematics(m, d)
    z_foot = d.geom_xpos[feet, 2].min()                 # stand 关键帧里脚底小球的高度

    lo, hi = -1.5, 0.5          # hip 越大（越往后），质心相对脚越靠前
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if place(mid)[1] < 0:
            lo = mid
        else:
            hi = mid
    q, _ = place(mid)
    return q, mid, -(mid + knee)


def pose_info(m, q):
    """静态信息：质心高度、躯干质心高度、质心相对支撑区中心的 x 偏移。"""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    feet = foot_geoms(m)
    # 往 +y 推，整机会绕左脚最外侧那排小球的接地点翻倒。静力学估计：F·h = M·g·w
    # 小球和地面的接触点在球心正下方，所以支点的 y 就是球心的 y，不用加半径
    w = d.geom_xpos[feet, 1].max() - d.subtree_com[0, 1]
    M = m.body_subtreemass[0]
    return dict(com_z=d.subtree_com[0, 2],
                lever_w=w,
                F_static=M * -m.opt.gravity[2] * w / d.body(PUSH_BODY).xipos[2],
                torso_z=d.body(PUSH_BODY).xipos[2],
                com_dx=d.subtree_com[0, 0] - d.geom_xpos[feet, 0].mean(),
                pelvis_z=q[2])


# ---------------------------------------------------------------- 推一次
def tilt_deg(d, pelvis):
    """骨盆竖直轴（body 坐标系 z 轴）和世界竖直方向的夹角。"""
    up_z = d.xmat[pelvis][8]                 # 旋转矩阵第 3 列的 z 分量
    return np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))


def simulate(m, q, F, record=False):
    """从姿势 q 出发（ctrl = 关节目标 = q[7:]），推一次，返回结果字典。"""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = q[7:]                        # 位置伺服：目标就是初始关节角
    mujoco.mj_forward(m, d)
    torso = m.body(PUSH_BODY).id
    pelvis = m.body("pelvis").id
    foot_sites = [m.site("left_foot").id, m.site("right_foot").id]
    foot0 = d.site_xpos[foot_sites, :2].copy()

    fell, min_z, max_tilt, slide = False, 9.0, 0.0, 0.0
    ts, tilts = [], []
    while d.time < T_END - 1e-9:
        pushing = PUSH_T0 <= d.time < PUSH_T0 + PUSH_DUR
        d.xfrc_applied[torso, :] = 0.0       # 不会自动清零，每步自己设
        if pushing:
            d.xfrc_applied[torso, :3] = F * PUSH_DIR   # 世界坐标系的力，作用在躯干质心
        mujoco.mj_step(m, d)

        tilt = tilt_deg(d, pelvis)
        min_z = min(min_z, d.qpos[2])
        max_tilt = max(max_tilt, tilt)
        drift = np.linalg.norm(d.site_xpos[foot_sites, :2] - foot0, axis=1).max()
        slide = max(slide, drift)
        if record:
            ts.append(d.time)
            tilts.append(tilt)
        if min_z < FALL_Z or max_tilt > FALL_TILT:
            fell = True
            if not record:
                break                        # 已经倒了，不用再算
    return dict(fell=fell, min_z=min_z, max_tilt=max_tilt, slide=slide,
                t=np.array(ts), tilt=np.array(tilts))


def critical_force(m, q):
    """二分搜索：不摔倒的最大推力。返回 (F, 用这个力推的结果)。

    前提：推力越大越容易倒（单调）。课文里用逐点扫描验证过这一点。
    """
    lo, hi = 0.0, F_MAX
    if simulate(m, q, lo)["fell"]:
        return 0.0, None                     # 不推都站不住
    if not simulate(m, q, hi)["fell"]:
        return float("inf"), None            # 上界都推不倒
    while hi - lo > F_TOL:
        mid = 0.5 * (lo + hi)
        if simulate(m, q, mid)["fell"]:
            hi = mid
        else:
            lo = mid
    return lo, simulate(m, q, lo)


def penetration_mm(m, q):
    """不推，站 1 s 后脚底小球压进地面的最大深度 (mm)。"""
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = q[7:]
    while d.time < 1.0:
        mujoco.mj_step(m, d)
    return -1000 * d.contact.dist[:d.ncon].min()


# ---------------------------------------------------------------- 实验
ROWS = []


def report(group, label, m, q, extra=""):
    t0 = time.time()
    F, res = critical_force(m, q)
    slide = res["slide"] * 100 if res else float("nan")
    pen = penetration_mm(m, q)
    print(f"  {label:<34}临界推力 {F:6.1f} N   脚最大位移 {slide:5.1f} cm   "
          f"静止穿透 {pen:4.2f} mm   ({time.time() - t0:.1f} s){extra}")
    ROWS.append(dict(group=group, label=label, F_crit=round(F, 1),
                     foot_slide_cm=round(slide, 1), penetration_mm=round(pen, 2)))
    return F


def part_a():
    print("\n=== A. 直腿 vs 屈膝 ===")
    m = load_model()
    qs = stand_pose(m)
    qc, hip, ankle = crouch_pose(m)
    print(f"  屈膝站姿：knee = {KNEE:.3f}  hip_pitch = {hip:.3f}  ankle_pitch = {ankle:.3f} rad")
    for name, q in (("直腿", qs), ("屈膝", qc)):
        info = pose_info(m, q)
        r0 = simulate(m, q, 0.0)
        print(f"  {name}：骨盆高 {info['pelvis_z']:.3f} m  质心高 {info['com_z']:.3f} m  "
              f"躯干质心高 {info['torso_z']:.3f} m  质心比支撑区中心 x 偏 {info['com_dx'] * 100:+.1f} cm  "
              f"不推时最大倾角 {r0['max_tilt']:.1f}°")
        print(f"        质心到左脚外侧小球接地点 w = {info['lever_w'] * 100:.2f} cm，"
              f"静力学翻倒推力 M·g·w/h = {info['F_static']:.1f} N")

    # 单调性检查：每 4 N 推一次，'.' 站住，'X' 摔倒
    print("  单调性检查（40–200 N，每 4 N 一格，. 站住 X 摔倒）：")
    for name, q in (("直腿", qs), ("屈膝", qc)):
        s = "".join("X" if simulate(m, q, F)["fell"] else "." for F in range(40, 201, 4))
        print(f"    {name} {s}")

    Fs = report("A", "直腿 stand（默认）", m, qs)
    Fc = report("A", f"屈膝 knee={KNEE}", m, qc)

    # 对照：把 29 个执行器的 kp 都乘 10，kv 乘 sqrt(10)，保持临界阻尼
    m10 = load_model()
    m10.actuator_gainprm[:, 0] *= 10
    m10.actuator_biasprm[:, 1] *= 10         # biasprm[1] = -kp
    m10.actuator_biasprm[:, 2] *= np.sqrt(10)  # biasprm[2] = -kv，kv ∝ sqrt(kp)
    Fs10 = report("A", "直腿，kp×10", m10, qs)
    Fc10 = report("A", "屈膝，kp×10", m10, qc)

    # 画倾角曲线：临界推力的 95% 和 105%
    curves = []
    for name, q, F in (("直腿", qs, Fs), ("屈膝", qc, Fc)):
        for k in (0.95, 1.05):
            r = simulate(m, q, k * F, record=True)
            curves.append((name, k, k * F, r))
    return qs, qc, curves


def part_b(qs):
    print("\n=== B. 摩擦（只改脚底 8 个小球的 friction[0]）===")
    out = []
    for mu in (0.05, 0.1, 0.3, 0.6, 1.0):
        m = load_model()
        m.geom_friction[foot_geoms(m), 0] = mu
        tag = "（默认）" if mu == 0.6 else ""
        out.append((mu, report("B", f"脚底 μ = {mu}{tag}", m, qs)))
    m = load_model()
    m.geom_friction[m.geom("floor").id, 0] = 0.1   # 只改地面
    report("B", "只把地面 μ 改成 0.1", m, qs)
    return out


def part_c(qs):
    print("\n=== C. 接触软硬（只改脚底小球）===")
    out = []
    for tc in (0.004, 0.01, 0.02, 0.05, 0.1):
        m = load_model()
        m.geom_solref[foot_geoms(m), 0] = tc
        tag = "（默认）" if tc == 0.02 else ""
        out.append((tc, report("C", f"solref timeconst = {tc}{tag}", m, qs)))
    for si in ((0.99, 0.99, 0.001), (0.5, 0.95, 0.001), (0.9, 0.95, 0.01), (0.5, 0.6, 0.01)):
        m = load_model()
        m.geom_solimp[foot_geoms(m), :3] = si
        report("C", f"solimp = {si[0]} {si[1]} {si[2]}", m, qs)
    return out


def part_d(qs, qc):
    print("\n=== D. 数值设置（直腿 / 屈膝）===")
    settings = [("timestep = 0.001", lambda m: setattr(m.opt, "timestep", 0.001)),
                ("integrator = Euler", lambda m: setattr(m.opt, "integrator", mujoco.mjtIntegrator.mjINT_EULER)),
                ("cone = elliptic", lambda m: setattr(m.opt, "cone", mujoco.mjtCone.mjCONE_ELLIPTIC))]
    for label, change in settings:
        m = load_model()
        change(m)
        report("D", f"直腿，{label}", m, qs)
        report("D", f"屈膝，{label}", m, qc)


def plot(curves, fric, soft):
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for name, k, F, r in curves:
        ax.plot(r["t"], r["tilt"], color=blue if name == "直腿" else orange,
                ls="-" if k < 1 else "--", lw=2, label=f"{name} {F:.0f} N（临界的 {k:.0%}）")
    ax.axvspan(PUSH_T0, PUSH_T0 + PUSH_DUR, color=grid, zorder=0)
    ax.axhline(FALL_TILT, color=ink, lw=1, ls=":")
    ax.text(T_END, FALL_TILT + 2, "摔倒线 45°", ha="right", color=ink)
    ax.set_ylim(0, 60)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("骨盆倾角 / °")
    ax.set_title("临界推力上下的骨盆倾角（灰带：推的 0.2 s）")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    x, y = zip(*fric)
    ax.plot(x, y, color=blue, lw=2, marker="o", ms=8)
    for xi, yi in fric:
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(7, 4), ha="left", color=ink)
    ax.set_xscale("log")
    ax.set_xticks(x, [str(v) for v in x])
    ax.margins(x=0.12, y=0.12)
    ax.set_xlabel("脚底摩擦系数 μ（对数刻度）")
    ax.set_ylabel("临界推力 / N")
    ax.set_title("直腿：临界推力 vs 摩擦")

    ax = axes[2]
    x, y = zip(*soft)
    ax.plot(x, y, color=blue, lw=2, marker="o", ms=8)
    for xi, yi in soft:
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(7, 4), ha="left", color=ink)
    ax.set_xscale("log")
    ax.set_xticks(x, [str(v) for v in x])
    ax.margins(x=0.12, y=0.12)
    ax.set_xlabel("脚底 solref timeconst / s（对数刻度）")
    ax.set_ylabel("临界推力 / N")
    ax.set_title("直腿：临界推力 vs 接触软硬")

    for ax in axes:
        ax.grid(color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    path = OUT / "05_push.png"
    fig.savefig(path, dpi=120)
    print(f"\n图已保存：{path}")



# ---------------------------------------------------------------- 推一次（无界面）
def setup(crouch, mu, timeconst, dur):
    """按命令行参数准备模型和站姿。dur 改的是全局 PUSH_DUR，simulate() 和 view() 都用它。"""
    global PUSH_DUR
    if dur is not None:
        PUSH_DUR = dur
    m = load_model()
    if mu is not None:
        m.geom_friction[foot_geoms(m), 0] = mu
    if timeconst is not None:
        m.geom_solref[foot_geoms(m), 0] = timeconst
    q = crouch_pose(m)[0] if crouch else stand_pose(m)
    return m, q


def once(force, crouch, mu, timeconst, dur, critical):
    m, q = setup(crouch, mu, timeconst, dur)
    feet = foot_geoms(m)
    print(f"站姿：{'屈膝 knee=' + str(KNEE) if crouch else '直腿 stand'}   "
          f"脚底 μ = {m.geom_friction[feet[0], 0]:g}   脚底 solref timeconst = {m.geom_solref[feet[0], 0]:g} s")
    print(f"推法：t = {PUSH_T0} s 起，侧推 {force:g} N，持续 {PUSH_DUR:g} s（冲量 {force * PUSH_DUR:.1f} N·s）")

    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = q[7:]
    mujoco.mj_forward(m, d)
    torso, pelvis = m.body(PUSH_BODY).id, m.body("pelvis").id
    foot_sites = [m.site("left_foot").id, m.site("right_foot").id]
    foot0 = d.site_xpos[foot_sites, :2].copy()

    # 从正前方看（相机在 +x，朝 -x 看），机器人左侧 +y 在画面右边
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.45, 0.55]           # 往推的方向偏一点，倒地后也在画面里
    cam.distance, cam.azimuth, cam.elevation = 3.6, 180.0, -10.0
    shots = [PUSH_T0 - 0.05, PUSH_T0 + PUSH_DUR, 2.0, 3.0, T_END - 0.01]
    frames, ts, tilts, zs = [], [], [], []
    fell_at, slide = None, 0.0
    with mujoco.Renderer(m, height=320, width=360) as r:
        while d.time < T_END - 1e-9:
            d.xfrc_applied[torso, :] = 0.0
            if PUSH_T0 <= d.time < PUSH_T0 + PUSH_DUR:
                d.xfrc_applied[torso, :3] = force * PUSH_DIR
            mujoco.mj_step(m, d)
            ts.append(d.time)
            tilts.append(tilt_deg(d, pelvis))
            zs.append(d.qpos[2])
            slide = max(slide, np.linalg.norm(d.site_xpos[foot_sites, :2] - foot0, axis=1).max())
            if fell_at is None and (zs[-1] < FALL_Z or tilts[-1] > FALL_TILT):
                fell_at = d.time
            if len(frames) < len(shots) and d.time >= shots[len(frames)]:
                r.update_scene(d, cam)
                frames.append((d.time, r.render().copy()))

    if fell_at is None:
        print(f"结果：站住了。骨盆最大倾角 {max(tilts):.1f}°，最低 {min(zs):.3f} m，两脚最大水平位移 {slide * 100:.1f} cm")
    else:
        print(f"结果：摔倒了（t = {fell_at:.2f} s 触发判据）。两脚最大水平位移 {slide * 100:.1f} cm")

    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, len(shots), height_ratios=[1.25, 1])
    for k, (t, img) in enumerate(frames):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"t = {t:.2f} s" + ("（推之前）" if k == 0 else "（推完）" if k == 1 else ""), fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(ts, tilts, color=orange if fell_at else blue, lw=2, label="骨盆倾角 / °")
    ax.axvspan(PUSH_T0, PUSH_T0 + PUSH_DUR, color=grid, zorder=0)
    ax.axhline(FALL_TILT, color=ink, lw=1, ls=":")
    ax.text(T_END, FALL_TILT + 2, "摔倒线 45°", ha="right", color=ink)
    ax.set_ylim(0, max(60, min(100, max(tilts) + 8)))
    ax.set_xlim(0, T_END)
    ax.set_xlabel("时间 / s（灰带：推的这段时间）")
    ax.set_ylabel("骨盆倾角 / °")
    verdict = f"摔倒（t = {fell_at:.2f} s）" if fell_at else f"站住了，最大倾角 {max(tilts):.1f}°"
    ax.set_title(f"{'屈膝' if crouch else '直腿'}，侧推 {force:g} N × {PUSH_DUR:g} s：{verdict}")
    ax.grid(color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = OUT / "05_once.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"图已保存：{path}")

    if critical:
        t0 = time.time()
        F, _ = critical_force(m, q)
        print(f"这组设置下的临界推力：{F:.1f} N（二分到 {F_TOL} N，用时 {time.time() - t0:.1f} s）")

# ---------------------------------------------------------------- 查看器
def view(force, crouch, mu, timeconst, dur, web_port=None):
    """实时观看。web_port 为 None 时开桌面查看器；给了端口就用网页查看器（课文页实验台右侧那个）。"""
    m, q = setup(crouch, mu, timeconst, dur)
    d = mujoco.MjData(m)
    torso = m.body(PUSH_BODY).id
    print(f"{'屈膝' if crouch else '直腿'}站姿，每 {T_END:.0f} s 重来一次，"
          f"第 {PUSH_T0} s 起侧推 {force} N、持续 {PUSH_DUR} s。" + ("" if web_port else "关掉窗口结束。"), flush=True)

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.ctrl[:] = q[7:]
        mujoco.mj_forward(m, d)

    reset()
    pushing = False
    if web_port:
        from webview import open_web                # 同目录的 webview.py
        viewer_cm = open_web(m, d, web_port, title="第 5 课：侧推 G1")
    else:
        from mujoco import viewer as mj_viewer
        viewer_cm = mj_viewer.launch_passive(m, d)
    with viewer_cm as viewer:
        while viewer.is_running():
            step_start = time.time()
            if d.time >= T_END:
                reset()
            now = PUSH_T0 <= d.time < PUSH_T0 + PUSH_DUR
            if now:
                d.xfrc_applied[torso, :3] = force * PUSH_DIR
            elif pushing:                    # 推完清零一次，之后不碰，鼠标拖拽仍可用
                d.xfrc_applied[torso, :3] = 0.0
            pushing = now
            mujoco.mj_step(m, d)
            if web_port:
                viewer.sync(m, d)
            else:
                viewer.sync()
            rest = m.opt.timestep - (time.time() - step_start)   # 按真实时间播放
            if rest > 0:
                time.sleep(rest)


def main():
    ap = argparse.ArgumentParser(description="第 5 课：侧推 G1，找临界推力")
    ap.add_argument("--view", action="store_true", help="打开查看器实时观看")
    ap.add_argument("--web", type=int, metavar="PORT", help="像 --view 一样实时观看，但显示在网页里（http://127.0.0.1:PORT/）")
    ap.add_argument("--once", action="store_true", help="只推一次（无界面），图写到 output/05_once.png")
    ap.add_argument("--critical", action="store_true", help="--once 时再二分求临界推力")
    ap.add_argument("--force", type=float, default=90.0, help="推力 (N)")
    ap.add_argument("--crouch", action="store_true", help="用屈膝站姿")
    ap.add_argument("--mu", type=float, default=None, help="脚底摩擦系数（默认 0.6）")
    ap.add_argument("--timeconst", type=float, default=None, help="脚底 solref 的 timeconst（默认 0.02 s）")
    ap.add_argument("--dur", type=float, default=None, help=f"推多久 (s)，默认 {PUSH_DUR}")
    args = ap.parse_args()
    if args.view or args.web:
        view(args.force, args.crouch, args.mu, args.timeconst, args.dur, web_port=args.web)
        return
    if args.once:
        once(args.force, args.crouch, args.mu, args.timeconst, args.dur, args.critical)
        return

    t0 = time.time()
    print(f"MuJoCo {mujoco.__version__}；推 {PUSH_BODY}，方向 {PUSH_DIR}，t={PUSH_T0}–{PUSH_T0 + PUSH_DUR} s；"
          f"判据：骨盆 < {FALL_Z} m 或倾角 > {FALL_TILT}°")
    qs, qc, curves = part_a()
    fric = part_b(qs)
    soft = part_c(qs)
    part_d(qs, qc)
    plot(curves, fric, soft)

    path = OUT / "05_push_results.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    print(f"结果表已保存：{path}")
    print(f"总用时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
