"""第 4 课：G1 “感觉”到了什么——IMU、关节传感器、脚底接触力、质心与支撑区。

运行：
    python code/04_senses.py               # 无界面，打印结果并把图、CSV 存到 output/
    python code/04_senses.py --force 60    # 换一个推力（N）
    python code/04_senses.py --dir backward   # 换一个方向：从正面往后推
    python code/04_senses.py --view        # 打开查看器实时观看推的过程
    python code/04_senses.py --web 8790 --force 60    # 网页里实时看（http://127.0.0.1:8790/），每 4 s 重推一次
    python code/04_senses.py --force 60 --out 04_lab  # 图和 CSV 改存成 output/04_lab_*，不覆盖默认的 04_*
--force、--dir 对 --view、--web 都有效。
"""
import argparse
import pathlib
import time

import matplotlib
matplotlib.use("Agg")                      # 只存图，不弹窗
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
np.set_printoptions(precision=3, suppress=True)
# 推力方向（世界坐标）。机器人面朝 +x，左手边是 +y
DIRS = {"left": (0, 1, 0), "right": (0, -1, 0), "forward": (1, 0, 0), "backward": (-1, 0, 0)}


# ---------- 1. 用 MjSpec 给模型补几个传感器（不改原文件） ----------
def build_model():
    spec = mujoco.MjSpec.from_file(str(XML))
    S, O = mujoco.mjtSensor, mujoco.mjtObj
    spec.add_sensor(name="knee_L_pos", type=S.mjSENS_JOINTPOS,
                    objtype=O.mjOBJ_JOINT, objname="left_knee_joint")
    spec.add_sensor(name="knee_L_vel", type=S.mjSENS_JOINTVEL,
                    objtype=O.mjOBJ_JOINT, objname="left_knee_joint")
    spec.add_sensor(name="knee_L_frc", type=S.mjSENS_ACTUATORFRC,
                    objtype=O.mjOBJ_ACTUATOR, objname="left_knee_joint")
    spec.add_sensor(name="com", type=S.mjSENS_SUBTREECOM,
                    objtype=O.mjOBJ_BODY, objname="pelvis")
    return spec.compile()


# ---------- 2. 脚底接触力：遍历 d.contact，换到世界坐标，按左右脚求和 ----------
def foot_contacts(m, d):
    """返回 {脚: 世界坐标下的合力 (3,)}，以及所有脚底接触点的 xy 坐标。"""
    foot_ids = {m.body(name).id: name for name in FEET}
    total = {name: np.zeros(3) for name in FEET}
    points = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b2 in foot_ids:            # 力作用在 geom2 上，方向不变
            name, sign = foot_ids[b2], 1.0
        elif b1 in foot_ids:          # 脚是 geom1，作用在脚上的力要反号
            name, sign = foot_ids[b1], -1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, i, f6)       # 接触坐标系：[法向, 切向1, 切向2]
        frame = c.frame.reshape(3, 3)             # 每一行是接触坐标系的一根轴
        f_world = frame.T @ f6[:3]                # 换到世界坐标
        total[name] += sign * f_world
        points.append(c.pos[:2].copy())
    return total, np.array(points)


# ---------- 3. 支撑多边形：接触点的凸包；质心投影到每条边的有符号距离 ----------
def convex_hull(pts):
    """单调链算法，返回逆时针排列的凸包顶点。"""
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def seg_dist(p, a, b):
    """点 p 到线段 ab 的距离。"""
    e = b - a
    L2 = e @ e
    t = 0.0 if L2 == 0 else np.clip((p - a) @ e / L2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * e))


def margin(hull, p):
    """点 p 到支撑区边界的有符号距离（m）：正 = 在里面，负 = 在外面。"""
    if len(hull) == 0:
        return np.nan                          # 双脚都离地，没有支撑区
    if len(hull) < 3:                          # 只剩一个点或一条线（比如脚掌侧棱着地）
        return -min(seg_dist(p, a, b) for a, b in zip(hull, np.roll(hull, -1, axis=0)))
    edges = list(zip(hull, np.roll(hull, -1, axis=0)))
    # 逆时针凸包：点在每条边的左侧 ⇔ 在里面
    inside = all((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0] >= 0 for a, b in edges)
    dist = min(seg_dist(p, a, b) for a, b in edges)
    return dist if inside else -dist


def web(force, direction, port):
    """网页查看器（课文页实验台右侧那个）：站 1 s → 推 0.2 s → 看到 4 s，然后重来，一直循环。"""
    from webview import open_web                # 同目录的 webview.py
    m = build_model()
    d = mujoco.MjData(m)
    key = m.key("stand").id
    torso = m.body("torso_link").id
    t_push, dur, t_end = 1.0, 0.2, 4.0
    push = force * np.array(DIRS[direction], dtype=float)
    print(f"每 {t_end:.0f} s 重来一次：第 {t_push} s 起往 {direction} 推躯干 {force:g} N，持续 {dur} s。", flush=True)

    def reset():
        mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)

    reset()
    pushing = False
    with open_web(m, d, port, title="第 4 课：推一下，看传感器") as viewer:
        while viewer.is_running():
            step_start = time.time()
            if d.time >= t_end:
                reset()
            now = t_push <= d.time < t_push + dur
            if now:
                d.xfrc_applied[torso, :3] = push
            elif pushing:                       # 推完清零一次，之后不碰，鼠标拖拽仍可用
                d.xfrc_applied[torso, :3] = 0.0
            pushing = now
            mujoco.mj_step(m, d)
            viewer.sync(m, d)
            rest = m.opt.timestep - (time.time() - step_start)   # 按真实时间播放
            if rest > 0:
                time.sleep(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", type=float, default=30.0, help="推力大小 N")
    ap.add_argument("--dir", default="left", choices=list(DIRS),
                    help="把机器人往哪边推：left(+y) right(-y) forward(+x) backward(-x)，默认 left")
    ap.add_argument("--view", action="store_true", help="打开查看器实时观看")
    ap.add_argument("--web", type=int, metavar="PORT", help="在网页里循环播放 C 段（http://127.0.0.1:PORT/）")
    ap.add_argument("--out", default="04", metavar="PREFIX",
                    help="输出文件名前缀，默认 04（04_senses.png 等）；换成 04_lab 就不会覆盖默认结果")
    args = ap.parse_args()
    if args.web:
        web(args.force, args.dir, args.web)
        return
    OUT.mkdir(exist_ok=True)

    m = build_model()
    d = mujoco.MjData(m)
    key = m.key("stand").id
    mujoco.mj_resetDataKeyframe(m, d, key)      # 同时设好 qpos 和 ctrl
    mass = m.body_subtreemass[0]
    g = -m.opt.gravity[2]
    print(f"总质量 {mass:.2f} kg，体重 m*g = {mass * g:.1f} N，时间步长 {m.opt.timestep} s")
    print(f"传感器 {m.nsensor} 个，nsensordata = {m.nsensordata}")

    # ===== A. 静止站立 1 s 后读数 =====
    while d.time < 1.0 - 1e-9:
        mujoco.mj_step(m, d)

    print("\n=== A. 静止站立 t = %.2f s ===" % d.time)
    for site in ("torso", "pelvis"):
        acc = d.sensor(f"imu-{site}-linear-acceleration").data
        gyr = d.sensor(f"imu-{site}-angular-velocity").data
        print(f"{site:6s} 加速度计 {np.round(acc, 3)}  |a| = {np.linalg.norm(acc):.3f}   "
              f"陀螺仪 {np.round(gyr, 5)}")
    R = d.site("imu_in_torso").xmat.reshape(3, 3)
    print("把 -gravity 转到躯干 IMU 坐标系：", np.round(R.T @ -m.opt.gravity, 3))
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    print(f"躯干 IMU 的 z 轴偏离竖直 {tilt:.2f}°")

    # 关节传感器 vs 直接读 MjData
    knee_j, knee_a = d.joint("left_knee_joint"), d.actuator("left_knee_joint")
    a_id = m.actuator("left_knee_joint").id
    kp, kv = m.actuator_gainprm[a_id, 0], -m.actuator_biasprm[a_id, 2]
    print(f"\n左膝 jointpos 传感器 {d.sensor('knee_L_pos').data[0]:.6f}   qpos {knee_j.qpos[0]:.6f}")
    print(f"左膝 jointvel 传感器 {d.sensor('knee_L_vel').data[0]:.6f}   qvel {knee_j.qvel[0]:.6f}")
    print(f"左膝 actuatorfrc 传感器 {d.sensor('knee_L_frc').data[0]:.4f}   "
          f"actuator_force {knee_a.force[0]:.4f}")
    print(f"ctrl - qpos = {knee_a.ctrl[0] - knee_j.qpos[0]:.5f} rad，kp = {kp:.0f}，kv = {kv:.2f}")

    def gaps():
        pd = kp * (knee_a.ctrl[0] - knee_j.qpos[0]) - kv * knee_j.qvel[0]
        return (d.sensor("knee_L_pos").data[0] - knee_j.qpos[0],
                d.sensor("knee_L_vel").data[0] - knee_j.qvel[0],
                d.sensor("knee_L_frc").data[0] - pd)

    print("mj_step 之后  传感器 - 状态：pos %.1e  vel %.1e  frc - kp(ctrl-q)+kv*qd %.1e" % gaps())
    mujoco.mj_forward(m, d)                     # 用当前状态重算一遍（不推进时间）
    print("mj_forward 后 传感器 - 状态：pos %.1e  vel %.1e  frc - kp(ctrl-q)+kv*qd %.1e" % gaps())

    # 接触力
    F, pts = foot_contacts(m, d)
    fz_l, fz_r = F[FEET[0]][2], F[FEET[1]][2]
    print(f"\n接触点 {d.ncon} 个，其中脚底 {len(pts)} 个")
    print(f"左脚合力 {np.round(F[FEET[0]], 2)} N")
    print(f"右脚合力 {np.round(F[FEET[1]], 2)} N")
    # 前掌和后跟各分多少：按接触点 x 坐标分成两组
    f6, front, heel = np.zeros(6), 0.0, 0.0
    x_mid = pts[:, 0].mean()
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        if d.contact[i].pos[0] > x_mid:
            front += f6[0]
        else:
            heel += f6[0]
    print(f"前掌 4 个球合计 {front:.1f} N，后跟 4 个球合计 {heel:.1f} N")
    print(f"竖直合力 {fz_l + fz_r:.1f} N   体重 {mass * g:.1f} N   比值 {(fz_l + fz_r) / (mass * g):.4f}")

    # 质心与支撑区
    com = d.subtree_com[0]
    print(f"\n整机质心 subtree_com[0] = {np.round(com, 4)}")
    print(f"subtreecom 传感器(pelvis) = {np.round(d.sensor('com').data, 4)}")
    hull = convex_hull(pts)
    span = hull.max(axis=0) - hull.min(axis=0)
    print(f"支撑多边形 {len(hull)} 个顶点，x 方向 {span[0]*100:.1f} cm，y 方向 {span[1]*100:.1f} cm")
    print("顶点 (x, y) cm：", "  ".join(f"({x*100:.1f}, {y*100:.1f})" for x, y in hull))
    print(f"质心投影离支撑区边界 {margin(hull, com[:2])*100:.2f} cm（正 = 在里面）")
    foot_y = [d.xpos[m.body(n).id][1] for n in FEET]   # 两只脚踝 roll 连杆的横向位置
    print(f"两脚中心横向间距 {(foot_y[0] - foot_y[1])*100:.1f} cm，"
          f"躯干质心离地 {d.xipos[m.body('torso_link').id][2]:.2f} m（推力就加在这里）")
    hull_static, pts_static = hull.copy(), pts.copy()

    # ===== B. 自由下落时的加速度计 =====
    d2 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d2, key)
    d2.qpos[2] += 1.0                          # 抬高 1 m，脚离地
    for _ in range(50):                        # 下落 0.1 s
        mujoco.mj_step(m, d2)
    acc_fall = d2.sensor("imu-torso-linear-acceleration").data
    print("\n=== B. 自由下落 0.1 s ===")
    print(f"躯干加速度计 {np.round(acc_fall, 3)}  |a| = {np.linalg.norm(acc_fall):.3f}  接触数 {d2.ncon}")

    # ===== C. 侧推 0.2 s，记录 3 s =====
    torso = m.body("torso_link").id
    t_push, dur, t_end = 1.0, 0.2, 4.0
    push = args.force * np.array(DIRS[args.dir], dtype=float)
    rows = []
    viewer = None
    if args.view:
        viewer = mujoco.viewer.launch_passive(m, d)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True

    while d.time < t_end - 1e-9:
        step_start = time.time()
        pushing = t_push <= d.time < t_push + dur
        d.xfrc_applied[torso, :3] = push if pushing else 0
        mujoco.mj_step(m, d)
        F, pts = foot_contacts(m, d)
        com = d.subtree_com[0]
        hull = convex_hull(pts) if len(pts) else np.zeros((0, 2))
        rows.append([d.time,
                     *d.sensor("imu-torso-angular-velocity").data,
                     *d.sensor("imu-torso-linear-acceleration").data,
                     F[FEET[0]][2], F[FEET[1]][2],
                     com[0], com[1], margin(hull, com[:2]),
                     d.sensor("knee_L_frc").data[0], d.qpos[2], len(pts)])
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            left = m.opt.timestep - (time.time() - step_start)
            if left > 0:
                time.sleep(left)

    data = np.array(rows)
    cols = ["t", "gx", "gy", "gz", "ax", "ay", "az", "Fz_left", "Fz_right",
            "com_x", "com_y", "margin", "knee_L_frc", "pelvis_z", "n_foot_contacts"]
    csv = OUT / f"{args.out}_push_log.csv"
    np.savetxt(csv, data, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")

    T = data[:, 0]
    after = T >= t_push
    i_mg = np.nanargmin(data[:, 11])
    print(f"\n=== C. t = {t_push} s 起往 {args.dir} 推躯干 {args.force:.0f} N，持续 {dur} s ===")
    print(f"陀螺仪 x 轴（侧倾）峰值 {np.abs(data[after, 1]).max():.3f} rad/s")
    print(f"加速度计 y 分量范围 [{data[after, 5].min():.2f}, {data[after, 5].max():.2f}] m/s²")
    print(f"左脚竖直力 {data[after, 7].min():.1f} ~ {data[after, 7].max():.1f} N，"
          f"右脚 {data[after, 8].min():.1f} ~ {data[after, 8].max():.1f} N")
    dt = m.opt.timestep
    print(f"左脚离地 {np.sum(data[after, 7] == 0) * dt:.3f} s，右脚离地 {np.sum(data[after, 8] == 0) * dt:.3f} s")
    print(f"质心水平最大偏移 {np.hypot(data[after, 9] - data[0, 9], data[after, 10] - data[0, 10]).max()*100:.2f} cm")
    print(f"质心离边界最近 {data[i_mg, 11]*100:.2f} cm（t = {T[i_mg]:.3f} s，"
          f"此刻脚底接触点 {data[i_mg, 14]:.0f} 个）")
    print(f"质心 x 范围 [{data[after, 9].min()*100:.1f}, {data[after, 9].max()*100:.1f}] cm，"
          f"y 范围 [{data[after, 10].min()*100:.1f}, {data[after, 10].max()*100:.1f}] cm")
    print(f"左膝力矩 {data[after, 12].min():.1f} ~ {data[after, 12].max():.1f} N·m")
    fallen = data[-1, 13] < 0.5
    print(f"结束时骨盆高度 {data[-1, 13]:.3f} m → {'倒了' if fallen else '站住了'}")
    print(f"结束时左右脚竖直力 {data[-1, 7]:.1f} / {data[-1, 8]:.1f} N，质心离边界 {data[-1, 11]*100:.2f} cm，"
          f"质心 (x, y) = ({data[-1, 9]*100:.1f}, {data[-1, 10]*100:.1f}) cm")

    # ---------- 画图 ----------
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(T, data[:, 1], label="gx（侧倾）")
    ax[0].plot(T, data[:, 2], label="gy（俯仰）")
    ax[0].plot(T, data[:, 3], label="gz（偏航）")
    ax[0].set_ylabel("陀螺仪 rad/s")
    ax[1].plot(T, data[:, 4], label="ax")
    ax[1].plot(T, data[:, 5], label="ay")
    ax[1].plot(T, data[:, 6], label="az")
    ax[1].set_ylabel("加速度计 m/s²")
    ax[2].plot(T, data[:, 7], label="左脚")
    ax[2].plot(T, data[:, 8], label="右脚")
    ax[2].plot(T, data[:, 7] + data[:, 8], "k", lw=1, label="合计")
    ax[2].axhline(mass * g, color="gray", ls="--", label="m·g")
    ax[2].set_ylabel("竖直接触力 N")
    ax[3].plot(T, data[:, 11] * 100, label="质心离支撑区边界")
    ax[3].axhline(0, color="r", ls="--")
    ax[3].set_ylabel("cm（正=在里面）")
    ax[3].set_xlabel("时间 s")
    for a in ax:
        a.axvspan(t_push, t_push + dur, color="orange", alpha=0.2)
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(f"往 {args.dir} 推 {args.force:.0f} N × {dur} s（橙色区间）")
    fig.tight_layout()
    fig.savefig(OUT / f"{args.out}_senses.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 6))
    h = np.vstack([hull_static, hull_static[:1]])
    ax.fill(h[:, 0], h[:, 1], alpha=0.2, label="支撑多边形（静止时）")
    ax.plot(pts_static[:, 0], pts_static[:, 1], "b.", label="脚底接触点")
    ax.plot(data[:, 9], data[:, 10], "r-", lw=1, label="质心投影轨迹")
    ax.plot(data[0, 9], data[0, 10], "ko", label="起点")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("俯视：质心投影与支撑区")
    fig.tight_layout()
    fig.savefig(OUT / f"{args.out}_support.png", dpi=120)
    print(f"\n已保存 {OUT / (args.out + '_senses.png')}、{OUT / (args.out + '_support.png')}、{csv}")

    if viewer is not None:                     # 仿真结束后窗口保持打开，手动关掉再退出
        print("仿真已结束，关闭查看器窗口即可退出。")
        while viewer.is_running():
            time.sleep(0.1)
        viewer.close()


if __name__ == "__main__":
    main()
