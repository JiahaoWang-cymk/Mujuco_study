"""第 3 课：把 G1 吊起来挥手。

做四件事：
1. 用 MjSpec 删掉骨盆的 freejoint，把骨盆固定在半空（再加一根只用来看的“吊绳”）。
2. 给右臂设计挥手轨迹：五次多项式抬手 → 正弦挥手 → 五次多项式放下。
3. 用三种方式跟踪同一条轨迹，记录目标角与实际角，画跟踪误差图：
     servo     纯位置伺服（模型自带的 position 执行器）
     ff        位置伺服 + 前馈 qfrc_bias（写进 qfrc_applied）
     gravcomp  位置伺服 + MuJoCo 内置重力补偿（body gravcomp + joint actuatorgravcomp）
4. 用 spec.to_xml() 把改好的模型存成 XML，再重新加载核对。

用法：
    python 03_wave.py                 # 无界面，跑三种方式，输出图、CSV 和 XML
    python 03_wave.py --view          # 实时观看（默认 servo 方式）
    python 03_wave.py --view --mode ff
"""
import argparse
import pathlib
import time

import matplotlib

matplotlib.use("Agg")                       # 只存图，不弹窗
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"

HANG_Z = 1.2                                # 骨盆吊在离地 1.2 m 处（站立时约 0.79 m）

# 右臂 7 个关节，顺序和模型里一致
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
       "right_wrist_yaw_joint"]
# 抬手姿势：大臂向外侧抬起，大臂再绕自身转 90°，让小臂朝上
Q_RAISE = np.array([0.0, -1.3, -1.57, 0.0, 0.0, 0.0, 0.0])
WAVE_JOINT = 3                              # 挥手靠肘关节（ARM 里的第 3 个）
WAVE_AMP = 0.5                              # 挥手幅度，rad
WAVE_FREQ = 1.5                             # 挥手频率，Hz

# 时间表（秒）：静止 → 抬手 → 挥手 → 放下 → 静止
T_UP0, T_UP1 = 0.5, 2.0
T_WAVE1 = 6.0                               # 挥 4 s，正好 6 个周期
T_DOWN1 = 7.5
T_END = 8.0

MODES = ["servo", "ff", "gravcomp"]
MODE_NAME = {"servo": "纯位置伺服", "ff": "伺服 + qfrc_bias 前馈", "gravcomp": "伺服 + 内置 gravcomp"}


# ---------------------------------------------------------------- 1. 改模型
def build_spec(mode="servo"):
    """读入 Menagerie 的 G1，改成“吊在空中”的版本。"""
    spec = mujoco.MjSpec.from_file(str(XML))

    # 删掉 freejoint：pelvis 没有关节，就和父 body（world）焊死了
    spec.delete(spec.joint("floating_base_joint"))
    spec.body("pelvis").pos = [0, 0, HANG_Z]

    # 关键帧里的 qpos 原来有 7 + 29 = 36 个数，去掉 freejoint 的 7 个
    key = spec.key("stand")
    key.qpos = list(key.qpos)[7:]

    # 加一根“吊绳”：只用来看，不参与碰撞，挂在 world 上也不算质量
    spec.worldbody.add_geom(
        name="rope", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, HANG_Z + 0.45, 0, 0, 3.0], size=[0.006, 0, 0],
        contype=0, conaffinity=0, rgba=[0.9, 0.8, 0.5, 1])

    if mode == "gravcomp":
        # 每个 body 加一个等于自身重量的向上的力
        for body in spec.bodies:
            body.gravcomp = 1.0
        # 这个力算到执行器头上（qfrc_actuator），而不是被动力 qfrc_passive
        for joint in spec.joints:
            joint.actgravcomp = True
    return spec


# ---------------------------------------------------------------- 2. 轨迹
def quintic(t, t0, t1):
    """五次多项式插值：s 从 0 平滑地走到 1，起止点速度、加速度都是 0。"""
    tau = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def arm_target(t, q_start):
    """t 时刻右臂 7 个关节的目标角。"""
    if t < T_UP1:                           # 静止 + 抬手
        s = quintic(t, T_UP0, T_UP1)
        return q_start + s * (Q_RAISE - q_start)
    if t < T_WAVE1:                         # 挥手：肘关节做正弦
        q = Q_RAISE.copy()
        q[WAVE_JOINT] += WAVE_AMP * np.sin(2 * np.pi * WAVE_FREQ * (t - T_UP1))
        return q
    s = quintic(t, T_WAVE1, T_DOWN1)        # 放下 + 静止
    return Q_RAISE + s * (q_start - Q_RAISE)


# ---------------------------------------------------------------- 3. 仿真
class Waver:
    """一个模型 + 一套控制方式。"""

    def __init__(self, mode):
        self.mode = mode
        self.spec = build_spec(mode)
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.arm_q = [self.m.joint(n).qposadr[0] for n in ARM]     # 在 qpos 里的下标
        self.arm_v = [self.m.joint(n).dofadr[0] for n in ARM]      # 在 qvel 里的下标
        self.arm_u = [self.m.actuator(n).id for n in ARM]          # 在 ctrl 里的下标
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("stand").id)
        self.q_start = self.d.ctrl[self.arm_u].copy()   # 右臂起始目标 = 关键帧里的 ctrl

    def step(self):
        """推进一步。用 mj_step1/mj_step2 拆开，中间拿到“这一步”的 qfrc_bias。"""
        m, d = self.m, self.d
        mujoco.mj_step1(m, d)                           # 算到速度阶段，qfrc_bias 已更新
        self.q_des = arm_target(d.time, self.q_start)   # 这一步的目标角
        d.ctrl[self.arm_u] = self.q_des
        if self.mode == "ff":
            d.qfrc_applied[:] = d.qfrc_bias             # 重力 + 科氏力/离心力，全部前馈
        mujoco.mj_step2(m, d)                           # 用新的 ctrl、qfrc_applied 完成这一步

    def run(self):
        """从头跑到 T_END，返回记录。"""
        self.reset()
        n = int(round(T_END / self.m.opt.timestep))
        log = {"t": np.zeros(n), "target": np.zeros((n, 7)), "q": np.zeros((n, 7)),
               "servo": np.zeros((n, 7)), "total": np.zeros((n, 7))}
        for i in range(n):
            self.step()
            # 记录同一时刻的目标角和实际角：这一步结束后，时间是 d.time，
            # 实际角是 d.qpos，目标角也要按 d.time 重新算（不能用这一步开头的 q_des，
            # 否则会差一个时间步，运动中的关节误差会被少算 timestep·q̇）
            log["t"][i] = self.d.time
            log["target"][i] = arm_target(self.d.time, self.q_start)
            log["q"][i] = self.d.qpos[self.arm_q]
            log["servo"][i] = self.d.actuator_force[self.arm_u]     # 位置伺服本身出的力矩
            log["total"][i] = (self.d.qfrc_actuator[self.arm_v]     # 关节实际得到的驱动力矩
                               + self.d.qfrc_applied[self.arm_v])
        return log


def save_model(mode, path):
    """用 to_xml() 存模型。

    to_xml() 会先编译，编译时网格要按原来的 meshdir="assets" 找，所以先导出，
    再把文本里的 meshdir 换成相对 output/ 的路径，存出来的文件才能直接打开。
    """
    spec = build_spec(mode)
    xml = spec.to_xml()
    old = 'meshdir="assets"'
    assert old in xml
    xml = xml.replace(old, 'meshdir="../models/unitree_g1/assets"')
    path.write_text(xml, encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(path))         # 重新加载核对
    return xml, m


# ---------------------------------------------------------------- 4. 画图与输出
def report(logs):
    t = logs["servo"]["t"]
    wave = (t >= T_UP1 + 1.0) & (t < T_WAVE1)            # 挥手段（跳过刚开始的 1 s 过渡）
    hold = t >= T_END - 0.2                              # 最后 0.2 s：静止保持
    short = ["肩pitch", "肩roll", "肩yaw", "肘", "腕roll", "腕pitch", "腕yaw"]

    print("\n=== 挥手段跟踪误差（度）：RMS / 最大 ===")
    print(f"{'方式':<22}" + "".join(f"{s:>14}" for s in short[:4]))
    for mode in MODES:
        e = np.degrees(logs[mode]["target"] - logs[mode]["q"])
        cells = "".join(f"{np.sqrt(np.mean(e[wave, j]**2)):>7.2f}/{np.abs(e[wave, j]).max():<6.2f}"
                        for j in range(4))
        print(f"{MODE_NAME[mode]:<20}{cells}")

    print("\n=== 放下后静止保持时的误差（度）===")
    for mode in MODES:
        e = np.degrees(logs[mode]["target"] - logs[mode]["q"])[hold].mean(axis=0)
        print(f"{MODE_NAME[mode]:<20}" + "  ".join(f"{s}{v:+.3f}" for s, v in zip(short[:4], e[:4])))

    print("\n=== 挥手段右肩 roll 的力矩（N·m）：伺服出力最大值 / 关节总驱动最大值 ===")
    for mode in MODES:
        lg = logs[mode]
        print(f"{MODE_NAME[mode]:<20}{np.abs(lg['servo'][wave, 1]).max():6.2f} / "
              f"{np.abs(lg['total'][wave, 1]).max():6.2f}")

    # ---- 图 1：纯位置伺服的目标与实际
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    lg = logs["servo"]
    for ax, j in zip(axes, [1, 3]):
        ax.plot(t, np.degrees(lg["target"][:, j]), color="#999999", lw=2.5, label="目标")
        ax.plot(t, np.degrees(lg["q"][:, j]), color="#1f6fb4", lw=1.2, label="实际")
        ax.set_ylabel(f"{ARM[j]}（度）")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_title("纯位置伺服：右臂目标角与实际角")
    axes[-1].set_xlabel("时间（s）")
    fig.tight_layout()
    fig.savefig(OUT / "03_wave_tracking.png", dpi=120)
    plt.close(fig)

    # ---- 图 2：三种方式的跟踪误差
    colors = {"servo": "#d62728", "ff": "#1f6fb4", "gravcomp": "#2ca02c"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, j in zip(axes, [1, 3]):
        for mode in MODES:
            e = np.degrees(logs[mode]["target"][:, j] - logs[mode]["q"][:, j])
            ax.plot(t, e, color=colors[mode], lw=1.0, label=MODE_NAME[mode])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(f"{short[j]} 误差（度）")
        ax.grid(alpha=0.3)
        for x in (T_UP0, T_UP1, T_WAVE1, T_DOWN1):
            ax.axvline(x, color="#bbbbbb", ls="--", lw=0.8)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("跟踪误差 = 目标 − 实际（虚线：抬手 / 挥手 / 放下的分界）")
    axes[-1].set_xlabel("时间（s）")
    fig.tight_layout()
    fig.savefig(OUT / "03_wave_error.png", dpi=120)
    plt.close(fig)

    # ---- CSV：每种方式一份
    for mode in MODES:
        lg = logs[mode]
        header = "t," + ",".join(f"target_{n},q_{n}" for n in ARM[:4])
        cols = [lg["t"]]
        for j in range(4):
            cols += [lg["target"][:, j], lg["q"][:, j]]
        np.savetxt(OUT / f"03_wave_{mode}.csv", np.column_stack(cols),
                   delimiter=",", header=header, comments="", fmt="%.6f")


def snapshot(waver, path, t_shot=2.0 + 1 / (4 * WAVE_FREQ)):
    """离屏渲染一张挥手的截图（第 10 课会细讲 Renderer）。"""
    waver.reset()
    while waver.d.time < t_shot:
        waver.step()
    mujoco.mj_forward(waver.m, waver.d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, -0.1, 1.2]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 160, -10
    with mujoco.Renderer(waver.m, 480, 640) as r:
        r.update_scene(waver.d, cam)
        import imageio.v3 as iio
        iio.imwrite(path, r.render())


# ---------------------------------------------------------------- 实时观看
def view(mode):
    import mujoco.viewer
    w = Waver(mode)
    print(f"实时观看：{MODE_NAME[mode]}。轨迹每 {T_END} s 重来一遍，关掉窗口结束。")
    with mujoco.viewer.launch_passive(w.m, w.d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if w.d.time >= T_END:
                w.reset()
            w.step()
            viewer.sync()
            dt = w.m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="用 launch_passive 实时观看")
    ap.add_argument("--mode", choices=MODES, default="servo", help="--view 时用哪种控制方式")
    args = ap.parse_args()
    if args.view:
        view(args.mode)
        return

    OUT.mkdir(exist_ok=True)
    m0 = mujoco.MjModel.from_xml_path(str(XML))
    w = Waver("servo")
    print(f"原模型：nq={m0.nq} nv={m0.nv}  →  吊起来后：nq={w.m.nq} nv={w.m.nv} nu={w.m.nu}")
    print(f"骨盆位置 {w.m.body('pelvis').pos}，父 body 是 world（id={w.m.body('pelvis').parentid[0]}）")

    logs = {}
    for mode in MODES:
        t0 = time.perf_counter()
        logs[mode] = Waver(mode).run()
        print(f"{MODE_NAME[mode]:<20} 仿真 {T_END} s，用时 {time.perf_counter() - t0:.2f} s")
    report(logs)

    for mode, name in [("servo", "g1_hanging.xml"), ("gravcomp", "g1_hanging_gravcomp.xml")]:
        xml, m = save_model(mode, OUT / name)
        print(f"\n已保存 output/{name}：{len(xml.splitlines())} 行，重新加载 nq={m.nq} nu={m.nu}，"
              f"含 gravcomp 属性的行 {sum('gravcomp=' in l for l in xml.splitlines())} 行")

    snapshot(w, OUT / "03_wave_snapshot.png")
    print("\n图：output/03_wave_tracking.png、output/03_wave_error.png、output/03_wave_snapshot.png")
    print("数据：output/03_wave_servo.csv、03_wave_ff.csv、03_wave_gravcomp.csv")


if __name__ == "__main__":
    main()
