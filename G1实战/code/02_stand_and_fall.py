"""第 2 课：G1 为什么站着、为什么会倒。

默认无界面运行，依次做 6 组实验并打印结果，图和 CSV 写到 output/：
  1. 读出 29 个 position 执行器的 gainprm / biasprm / forcerange
  2. 用公式手算执行器力矩，和 MuJoCo 算的 actuator_force 对比；演示 forcerange 截断
  3. 三种起始方式（关键帧 / 不加载关键帧 / ctrl 清零）各站 10 秒，再长跑 60 秒
  4. 把 kp 逐步调小（kv 不变），记录骨盆高度曲线和摔倒时间
  5. 侧推躯干 0.2 秒：推倒以后会不会自己爬起来？
  6. 时间步长 timestep × 积分器 integrator 的稳定性对比

用法：
  python 02_stand_and_fall.py                 # 全部实验（无界面）
  python 02_stand_and_fall.py --kv-follow     # 第 4 组改用 MjSpec 重新编译，kv 随 kp 按 dampratio 重算
  python 02_stand_and_fall.py --view --kp 100 # 打开查看器实时看：kp=100 时它怎么倒
  python 02_stand_and_fall.py --view --push 100   # kp 保持 500，第 1 秒侧推 100 N
  python 02_stand_and_fall.py --web 8790 --kp 100 # 同 --view，但显示在网页里：浏览器打开 http://127.0.0.1:8790/
  python 02_stand_and_fall.py --once --kp 150     # 只跑一次（无界面，几秒）：打印结果，画骨盆高度曲线和截图
  python 02_stand_and_fall.py --once --dt 0.01 --integrator euler   # 换步长和积分器
--kp、--kv-follow、--push、--start、--dt、--integrator、--time 对 --once、--view、--web 都有效。
"""
import argparse
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

FALL_Z = 0.5          # 骨盆高度低于 0.5 m 就算摔倒（站立时约 0.79 m）

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(kp=None, kv_follow=False, timestep=None, integrator=None):
    """加载 G1。kp 不为 None 时把 29 个执行器的 kp 都改成这个值。

    kv_follow=False：直接改编译好的 MjModel，只动 kp，kv 保持原值。
    kv_follow=True ：用 MjSpec 改 kp 再重新编译，dampratio=1 会按新的 kp 重算 kv。
    """
    if kp is not None and kv_follow:
        spec = mujoco.MjSpec.from_file(str(XML))
        for act in spec.actuators:
            act.gainprm[0] = kp
            act.biasprm[1] = -kp             # biasprm[2] 仍是 +1，表示 dampratio=1
        m = spec.compile()
    else:
        m = mujoco.MjModel.from_xml_path(str(XML))
        if kp is not None:
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp   # biasprm[2]（即 -kv）不动
    if timestep is not None:
        m.opt.timestep = timestep
    if integrator is not None:
        m.opt.integrator = integrator
    return m


def simulate(m, T, start="key", push=None, kv_zero=False):
    """仿真 T 秒，返回 (时间数组, 骨盆高度数组, 摔倒时刻或 None, 最终的 MjData)。

    start: "key"      加载关键帧 stand（qpos 和 ctrl 都来自关键帧）
           "key_ctrl0" 加载关键帧，但把 ctrl 清零
           "qpos0"     不加载关键帧（mj_resetData），ctrl 默认为 0
    push : (力 N, 开始时刻 s, 持续 s)，沿 +y 方向推躯干
    """
    if kv_zero:
        m.actuator_biasprm[:, 2] = 0.0
    d = mujoco.MjData(m)
    if start.startswith("key"):
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        if start == "key_ctrl0":
            d.ctrl[:] = 0.0
    else:
        mujoco.mj_resetData(m, d)
    torso = m.body("torso_link").id

    n = int(round(T / m.opt.timestep))
    ts, zs = np.zeros(n), np.zeros(n)
    t_fall = None
    for i in range(n):
        if push is not None:
            F, t0, dur = push
            d.xfrc_applied[torso, 1] = F if t0 <= d.time < t0 + dur else 0.0
        mujoco.mj_step(m, d)
        ts[i], zs[i] = d.time, d.qpos[2]
        if t_fall is None and d.qpos[2] < FALL_Z:
            t_fall = d.time
    return ts, zs, t_fall, d


def fmt_fall(t):
    return "没倒" if t is None else f"{t:5.2f} s"


# ---------------------------------------------------------------- 1
def part1_params():
    m = load_model()
    print("=" * 70)
    print("1. 执行器参数（编译后的 MjModel 里真正用的数）")
    print(f"   timestep = {m.opt.timestep} s，integrator = {mujoco.mjtIntegrator(m.opt.integrator).name}")
    print(f"   gaintype = {mujoco.mjtGain(m.actuator_gaintype[0]).name}，"
          f"biastype = {mujoco.mjtBias(m.actuator_biastype[0]).name}")
    print(f"   {'执行器':<28}{'gainprm[0]':>11}{'biasprm[1]':>11}{'biasprm[2]':>11}{'forcerange':>13}")
    for a in range(m.nu):
        g = m.actuator_gainprm[a, 0]
        b1, b2 = m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2]
        print(f"   {m.actuator(a).name:<28}{g:>11.1f}{b1:>11.1f}{b2:>11.2f}"
              f"{m.actuator_forcerange[a, 1]:>10.0f} N·m")
    kv = -m.actuator_biasprm[:, 2]
    print(f"   kv 最小 {kv.min():.2f}（{m.actuator(int(kv.argmin())).name}），"
          f"最大 {kv.max():.2f}（{m.actuator(int(kv.argmax())).name}）")


# ---------------------------------------------------------------- 2
def part2_formula():
    print("=" * 70)
    print("2. 手算力矩 vs MuJoCo 的 actuator_force")
    m = load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    for _ in range(250):                     # 先站 0.5 秒，让各关节有点偏差和速度
        mujoco.mj_step(m, d)

    def by_formula():
        # 每个执行器只驱动一个铰链关节，用 trnid 找到它在 qpos / qvel 里的位置
        jid = m.actuator_trnid[:, 0]
        q = d.qpos[m.jnt_qposadr[jid]]
        qd = d.qvel[m.jnt_dofadr[jid]]
        kp = m.actuator_gainprm[:, 0]
        f = kp * d.ctrl + m.actuator_biasprm[:, 0] + m.actuator_biasprm[:, 1] * q \
            + m.actuator_biasprm[:, 2] * qd  # = kp*(ctrl-q) - kv*qd
        lo, hi = m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]
        return f, np.clip(f, lo, hi)

    mujoco.mj_forward(m, d)
    raw, clipped = by_formula()
    print(f"   站立 0.5 s 后：手算与 actuator_force 最大差 {np.abs(clipped - d.actuator_force).max():.2e} N·m")
    k = m.actuator("left_knee_joint").id
    print(f"   左膝：ctrl={d.ctrl[k]:.3f}  q={d.joint('left_knee_joint').qpos[0]:.5f}  "
          f"力矩={d.actuator_force[k]:.3f} N·m")

    # 把左膝目标突然改成 1.0 rad：kp*误差 = 500 N·m，远超 139 N·m 的上限
    d.ctrl[k] = 1.0
    mujoco.mj_forward(m, d)
    raw, clipped = by_formula()
    print(f"   左膝目标改成 1.0 rad：公式给出 {raw[k]:.1f} N·m，截断后 {clipped[k]:.1f}，"
          f"MuJoCo 给出 {d.actuator_force[k]:.1f}")


# ---------------------------------------------------------------- 3
def part3_start():
    print("=" * 70)
    print("3. 三种起始方式，各站 10 秒（kp=500）")
    for start, desc in [("key", "加载 stand 关键帧（qpos+ctrl）"),
                        ("qpos0", "不加载关键帧，mj_resetData"),
                        ("key_ctrl0", "加载关键帧后把 ctrl 清零")]:
        m = load_model()
        d0 = mujoco.MjData(m)
        if start == "qpos0":
            mujoco.mj_resetData(m, d0)
        else:
            mujoco.mj_resetDataKeyframe(m, d0, m.key("stand").id)
        z0 = d0.qpos[2]
        ts, zs, tf, d = simulate(m, 10.0, start=start)
        elbow = d.joint("left_elbow_joint").qpos[0]
        print(f"   {desc:<26} 初始 {z0:.3f} m → 10 s 后 {zs[-1]:.4f} m，"
              f"最低 {zs.min():.4f}，左肘 {elbow:.2f} rad，摔倒：{fmt_fall(tf)}")

    # 长跑 60 秒，看会不会慢慢漂走
    m = load_model()
    t0 = time.perf_counter()
    ts, zs, tf, d = simulate(m, 60.0, start="key")
    wall = time.perf_counter() - t0
    print(f"   关键帧 + ctrl 保持，60 s 后：骨盆 x={d.qpos[0]*1000:+.3f} mm, "
          f"y={d.qpos[1]*1000:+.3f} mm, z={d.qpos[2]:.4f} m；摔倒：{fmt_fall(tf)}")
    print(f"   （仿真 60 s 用了 {wall:.2f} s 真实时间，约 {60/wall:.0f} 倍实时）")

    # 位置伺服的“稳态误差”：要输出力矩 τ，就必须有误差 τ/kp
    jid = m.actuator_trnid[:, 0]
    err = d.ctrl - d.qpos[m.jnt_qposadr[jid]]
    order = np.argsort(-np.abs(d.actuator_force))[:5]
    print("   站稳后力矩最大的 5 个执行器（误差 = 力矩 / kp）：")
    for a in order:
        print(f"     {m.actuator(a).name:<26} 力矩 {d.actuator_force[a]:+7.2f} N·m   "
              f"ctrl-q = {err[a]:+.5f} rad ({np.degrees(err[a]):+.3f}°)")


# ---------------------------------------------------------------- 4
def part4_kp(kv_follow):
    tag = "MjSpec 重编译，kv 随 kp 重算" if kv_follow else "只改 kp，kv 保持原值"
    print("=" * 70)
    print(f"4. 调小 kp，各仿真 10 秒（{tag}）")
    cases = [(500, False), (250, False), (200, False), (150, False),
             (100, False), (50, False), (0, False)]
    if not kv_follow:
        cases.append((0, True))               # kp=0 且 kv=0：彻底断电
    # 粗估：把整机看成绕踝关节转动的倒立摆，重力“掀翻”它的刚度是 M*g*h（N·m/rad）
    m = load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    M = m.body_subtreemass[0]
    h = d.subtree_com[0, 2] - d.xanchor[m.joint("left_ankle_pitch_joint").id, 2]
    print(f"   倒立摆估算：M={M:.2f} kg，质心比踝轴高 h={h:.3f} m，"
          f"M*g*h={M * 9.81 * h:.0f} N·m/rad → 两个踝 2*kp 要大于它，kp > {M * 9.81 * h / 2:.0f}")
    rows, curves = [], []
    for kp, kv_zero in cases:
        m = load_model(kp=kp, kv_follow=kv_follow)
        knee_kv = -m.actuator_biasprm[m.actuator("left_knee_joint").id, 2]
        ts, zs, tf, d = simulate(m, 10.0, start="key", kv_zero=kv_zero)
        if kv_zero:
            knee_kv = 0.0
        label = f"kp={kp}" + ("，kv=0" if kv_zero else "")
        print(f"   {label:<12} 左膝 kv={knee_kv:5.2f}  摔倒：{fmt_fall(tf)}  "
              f"10 s 后骨盆 {zs[-1]:.3f} m")
        rows.append((kp, knee_kv, -1 if tf is None else tf, zs[-1]))
        curves.append((label, ts, zs))

    suffix = "_kvfollow" if kv_follow else ""
    np.savetxt(OUT / f"02_kp_sweep{suffix}.csv", np.array(rows), delimiter=",",
               header="kp,left_knee_kv,t_fall(-1=no fall),z_final", comments="", fmt="%.4f")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, ts, zs in curves:
        ax.plot(ts, zs, label=label)
    ax.axhline(FALL_Z, color="gray", ls="--", lw=1)
    ax.text(0.1, FALL_Z + 0.01, "摔倒判据 0.5 m", ha="left", va="bottom", color="gray")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("骨盆高度 (m)")
    ax.set_title(f"kp 越小越站不住（{tag}）")
    ax.legend(ncol=2, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.3))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / f"02_kp_sweep{suffix}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"   曲线已存 {path}")


# ---------------------------------------------------------------- 5
def part5_push():
    print("=" * 70)
    print("5. 第 1 秒起侧推躯干 0.2 秒（+y 方向），共仿真 10 秒")
    fig, ax = plt.subplots(figsize=(8, 4))
    for F in [50, 80, 90, 100, 200]:
        m = load_model()
        ts, zs, tf, d = simulate(m, 10.0, start="key", push=(F, 1.0, 0.2))
        up = "站着" if zs[-1] > FALL_Z else "躺在地上"
        # 躺下以后，各关节离 ctrl 目标还差多少？
        jid = m.actuator_trnid[:, 0]
        err = np.abs(d.ctrl - d.qpos[m.jnt_qposadr[jid]])
        print(f"   {F:>4} N（冲量 {F*0.2:4.0f} N·s）：摔倒 {fmt_fall(tf)}，"
              f"10 s 后骨盆 {zs[-1]:.3f} m（{up}），最后 1 s 骨盆最高 {zs[ts > 9].max():.3f} m，"
              f"关节离目标最多差 {np.degrees(err.max()):.1f}°（{m.actuator(int(err.argmax())).name}）")
        ax.plot(ts, zs, label=f"{F} N")
    ax.axvspan(1.0, 1.2, color="orange", alpha=0.2)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("骨盆高度 (m)")
    ax.set_title("侧推 0.2 秒后的骨盆高度（橙色为推力作用时段）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "02_push.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"   曲线已存 {path}")


# ---------------------------------------------------------------- 6
def part6_integrator():
    print("=" * 70)
    print("6. timestep × integrator，站立 5 秒（kp=500）")
    # dampratio=1 时 kv = 2*sqrt(kp*m)，反推等效惯量 m，再算伺服的固有周期 2π/sqrt(kp/m)
    m = load_model()
    kp = m.actuator_gainprm[:, 0]
    kv = -m.actuator_biasprm[:, 2]
    m_eff = kv ** 2 / (4 * kp)
    period = 2 * np.pi / np.sqrt(kp / m_eff)
    a = int(period.argmin())
    print(f"   最“快”的伺服：{m.actuator(a).name}，等效惯量 {m_eff[a]:.4f} kg·m²，"
          f"固有周期 {period[a] * 1000:.1f} ms（最慢的 {period.max() * 1000:.1f} ms）")
    # 把它当成孤立的“质量-弹簧-阻尼”，kp、kv 都显式积分时，步长要满足 ω·dt < 2√2-2 ≈ 0.83
    omega = 2 * np.pi / period[a]
    print(f"   粗估：kv 显式处理时，这个伺服要求 dt < 0.83/ω = {0.83 / omega * 1000:.1f} ms")
    I = mujoco.mjtIntegrator
    BADQACC = int(mujoco.mjtWarning.mjWARN_BADQACC)
    for integ in [I.mjINT_IMPLICITFAST, I.mjINT_EULER]:
        for dt in [0.002, 0.005, 0.01, 0.02]:
            m = load_model(timestep=dt, integrator=integ)
            t0 = time.perf_counter()
            ts, zs, tf, d = simulate(m, 5.0, start="key")
            wall = time.perf_counter() - t0
            nbad = d.warning[BADQACC].number
            # 发散时 MuJoCo 会调用 mj_resetData，d.time 回到 0。时间倒退一次就是重置了一次
            nreset = int(np.sum(np.diff(ts) < 0))
            print(f"   {integ.name:<20} dt={dt:<6} 摔倒：{fmt_fall(tf):<8} 最低 {zs.min():.3f} m  "
                  f"结束时 d.time={d.time:4.2f} s 骨盆 {zs[-1]:.3f} m  "
                  f"BADQACC 计数 {nbad}  实际重置 {nreset:2d} 次  用时 {wall:.2f} s")
    # 对照：去掉关节的干摩擦 frictionloss，Euler 在小步长下也会发散
    m = load_model(timestep=0.004, integrator=I.mjINT_EULER)
    m.dof_frictionloss[6:] = 0.0              # 前 6 个是 freejoint，本来就是 0
    ts, zs, tf, d = simulate(m, 5.0, start="key")
    print(f"   对照：mjINT_EULER dt=0.004，把 frictionloss 改成 0 → 摔倒：{fmt_fall(tf)}")


# ---------------------------------------------------------------- 跑一次（无界面）
INTEGRATORS = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
               "euler": mujoco.mjtIntegrator.mjINT_EULER,
               "rk4": mujoco.mjtIntegrator.mjINT_RK4}
START_NAME = {"key": "加载 stand 关键帧", "qpos0": "不加载关键帧（mj_resetData）", "key_ctrl0": "加载关键帧后把 ctrl 清零"}


def setup(kp, kv_follow, dt, integrator):
    """按命令行参数准备模型。kp=500 且不加 --kv-follow 时，模型和 XML 里的一样。"""
    integ = INTEGRATORS[integrator] if integrator else None
    return load_model(kp=kp, kv_follow=kv_follow, timestep=dt, integrator=integ)


def reset_to(m, d, start):
    """按 start（同 simulate() 的 start 参数）重置。"""
    if start.startswith("key"):
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        if start == "key_ctrl0":
            d.ctrl[:] = 0.0
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


def describe(m, kp, kv_follow, push, start, T):
    knee_kv = -m.actuator_biasprm[m.actuator("left_knee_joint").id, 2]
    print(f"kp={kp:g}（左膝 kv={knee_kv:.2f}{'，随 kp 重算' if kv_follow else ''}），起始：{START_NAME[start]}，"
          f"timestep={m.opt.timestep:g} s，integrator={mujoco.mjtIntegrator(m.opt.integrator).name}")
    print(f"仿真 {T:g} s" + (f"，第 1 秒起侧推躯干 {push:g} N、持续 0.2 s（冲量 {push * 0.2:.0f} N·s）" if push else "，不推"),
          flush=True)


def once(kp, kv_follow, push, start, dt, integrator, T):
    m = setup(kp, kv_follow, dt, integrator)
    describe(m, kp, kv_follow, push, start, T)
    d = mujoco.MjData(m)
    reset_to(m, d, start)
    torso = m.body("torso_link").id

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.25, 0.4]            # 往前（+x）、往左（+y，推的方向）偏一点，倒地后也在画面里
    cam.distance, cam.azimuth, cam.elevation = 3.2, 150.0, -18.0
    n = int(round(T / m.opt.timestep))
    shot_steps = {int(round(k * n / 4)) - 1 for k in (1, 2, 3, 4)}
    frames = []
    ts, zs = np.zeros(n), np.zeros(n)
    t_fall, nreset = None, 0
    with mujoco.Renderer(m, height=300, width=300) as r:
        r.update_scene(d, cam)
        frames.append((0.0, r.render().copy()))
        for i in range(n):
            if push:
                d.xfrc_applied[torso, 1] = push if 1.0 <= d.time < 1.2 else 0.0
            t_before = d.time
            mujoco.mj_step(m, d)
            if d.time < t_before:            # 发散后 MuJoCo 自动 mj_resetData，d.time 回到 0
                nreset += 1
            ts[i], zs[i] = (i + 1) * m.opt.timestep, d.qpos[2]   # 横轴用步数×timestep，重置了也连续
            if t_fall is None and d.qpos[2] < FALL_Z:
                t_fall = ts[i]
            if i in shot_steps:
                r.update_scene(d, cam)
                frames.append((ts[i], r.render().copy()))

    jid = m.actuator_trnid[:, 0]
    err = np.abs(d.ctrl - d.qpos[m.jnt_qposadr[jid]])
    up = "站着" if zs[-1] > FALL_Z else "躺在地上"
    print(f"结果：摔倒 {fmt_fall(t_fall).strip()}，最低 {zs.min():.3f} m，{T:g} s 后骨盆 {zs[-1]:.3f} m（{up}），"
          f"关节离目标最多差 {np.degrees(err.max()):.1f}°（{m.actuator(int(err.argmax())).name}）")
    if nreset:
        print(f"注意：仿真发散，MuJoCo 自动重置了 {nreset} 次，结束时 d.time 只有 {d.time:.2f} s。"
              "上面的时间按“步数 × timestep”累计，不随重置回到 0")

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, len(frames), height_ratios=[1.2, 1])
    for k, (t, img) in enumerate(frames):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"t = {t:.2f} s", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(ts, zs, color="#eb6834" if t_fall else "#2a78d6", lw=2)
    ax.axhline(FALL_Z, color="gray", ls="--", lw=1)
    ax.text(0.05, FALL_Z + 0.01, "摔倒判据 0.5 m", ha="left", va="bottom", color="gray")
    if push:
        ax.axvspan(1.0, 1.2, color="orange", alpha=0.2)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("时间 (s)" + ("（橙色：推力作用时段）" if push else ""))
    ax.set_ylabel("骨盆高度 (m)")
    integ = mujoco.mjtIntegrator(m.opt.integrator).name.replace("mjINT_", "")
    ax.set_title(f"kp={kp:g}，{START_NAME[start]}，" + (f"推 {push:g} N，" if push else "") + f"dt={m.opt.timestep:g} s {integ}："
                 f"摔倒 {fmt_fall(t_fall).strip()}" + (f"（自动重置 {nreset} 次）" if nreset else ""))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "02_once.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"图已保存：{path}")


# ---------------------------------------------------------------- 查看器
def view(kp, push, kv_follow=False, start="key", dt=None, integrator=None, T=10.0, web_port=None):
    """实时观看，每 T 秒重来一次。web_port 为 None 时开桌面查看器；给了端口就用网页查看器（课文页实验台右侧那个）。"""
    m = setup(kp, kv_follow, dt, integrator)
    d = mujoco.MjData(m)
    reset_to(m, d, start)
    torso = m.body("torso_link").id
    pushing = False
    describe(m, kp, kv_follow, push, start, T)
    print(f"每 {T:g} s 重来一次。" + ("" if web_port else "关掉窗口结束。"), flush=True)
    if web_port:
        from webview import open_web                # 同目录的 webview.py
        viewer_cm = open_web(m, d, web_port, title="第 2 课：站着与倒下")
    else:
        from mujoco import viewer as mj_viewer
        viewer_cm = mj_viewer.launch_passive(m, d)
    with viewer_cm as viewer:
        t_run = 0.0                          # 这一轮跑了多久：自己累加，发散自动重置时 d.time 会回到 0
        while viewer.is_running():
            step_start = time.time()
            if t_run >= T:
                reset_to(m, d, start)
                d.xfrc_applied[:] = 0.0
                pushing, t_run = False, 0.0
            if push:
                now = 1.0 <= d.time < 1.2
                if now:
                    d.xfrc_applied[torso, 1] = push
                elif pushing:                # 推完清零一次，之后不再碰 xfrc_applied
                    d.xfrc_applied[torso, 1] = 0.0
                pushing = now
            mujoco.mj_step(m, d)
            t_run += m.opt.timestep
            if web_port:
                viewer.sync(m, d)
            else:
                viewer.sync()
            # 按真实时间播放
            rest = m.opt.timestep - (time.time() - step_start)
            if rest > 0:
                time.sleep(rest)


def main():
    ap = argparse.ArgumentParser(description="第 2 课：站着与倒下")
    ap.add_argument("--view", action="store_true", help="打开查看器实时观看（不跑批量实验）")
    ap.add_argument("--web", type=int, metavar="PORT", help="像 --view 一样实时观看，但显示在网页里（http://127.0.0.1:PORT/）")
    ap.add_argument("--once", action="store_true", help="只跑一次（无界面），图写到 output/02_once.png")
    ap.add_argument("--kp", type=float, default=500.0, help="--once/--view/--web 时所有执行器的 kp")
    ap.add_argument("--push", type=float, default=0.0, help="--once/--view/--web 时第 1 秒侧推的力 (N)")
    ap.add_argument("--start", choices=list(START_NAME), default="key", help="起始方式（同实验一），默认 key")
    ap.add_argument("--dt", type=float, default=None, help="timestep (s)，默认 0.002")
    ap.add_argument("--integrator", choices=list(INTEGRATORS), default=None, help="积分器，默认 implicitfast")
    ap.add_argument("--time", type=float, default=10.0, help="--once 仿真多久 / --view、--web 多久重来一次 (s)")
    ap.add_argument("--kv-follow", action="store_true",
                    help="第 4 组用 MjSpec 重编译，kv 随 kp 重算（和 --once/--view/--web 一起用时作用于那次仿真）")
    args = ap.parse_args()

    print(f"MuJoCo {mujoco.__version__}")
    if args.view or args.web:
        view(args.kp, args.push, args.kv_follow, args.start, args.dt, args.integrator, args.time,
             web_port=args.web)
        return
    if args.once:
        once(args.kp, args.kv_follow, args.push, args.start, args.dt, args.integrator, args.time)
        return
    if args.kv_follow:                       # 只跑第 4 组
        part4_kp(kv_follow=True)
        return
    part1_params()
    part2_formula()
    part3_start()
    part4_kp(kv_follow=False)
    part5_push()
    part6_integrator()


if __name__ == "__main__":
    main()
