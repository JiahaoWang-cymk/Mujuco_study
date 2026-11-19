"""第 6 课：省力地站——用逆动力学算出“刚好站住”需要的力矩。

运行：
    python code/06_inverse.py                 # 无界面，打印结果并在 output/ 下保存图和 CSV
    python code/06_inverse.py --view          # 打开查看器，看屈膝站 + 前馈 PD（默认 kp=200）
    python code/06_inverse.py --view --kp 300 --noff   # 换增益、去掉前馈再看
    python code/06_inverse.py --once --kp 200 --noff   # 只跑一组（无界面，几秒）：打印结果，图存到 output/06_once.png
    python code/06_inverse.py --web 8790 --kp 200      # 像 --view，但显示在网页里（http://127.0.0.1:8790/），每 8 s 重来
--kp、--noff、--bend 对 --once、--view、--web 都有效。
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE = "#2a78d6", "#eb6834"
np.set_printoptions(precision=3, suppress=True, linewidth=120)

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
KEY = m.key("stand").id
NAMES = [m.actuator(i).name for i in range(m.nu)]
SHORT = [n.replace("_joint", "") for n in NAMES]
LIMIT = m.actuator_forcerange[:, 1]          # 各电机的力矩上限（N·m）
G = -m.opt.gravity[2]
MASS = m.body_subtreemass[0]
# 脚底的 8 个小球（第 1 课 1.3 ④）：半径 5 mm 的 sphere
FOOT = [g for g in range(m.ngeom)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_size[g, 0] == 0.005]
KNEE = NAMES.index("left_knee_joint")        # 执行器序号 = qpos[7:] 里的序号（第 1 课 1.5）


# ---------------- 工具函数 ----------------
def pose(bend):
    """stand 关键帧的姿势，再把两腿屈膝 bend 弧度：髋、踝各分担一半，小腿和大腿对称。"""
    q = m.key_qpos[KEY].copy()
    for s in ("left", "right"):
        q[7 + NAMES.index(f"{s}_hip_pitch_joint")] = -bend / 2
        q[7 + NAMES.index(f"{s}_knee_joint")] = bend
        q[7 + NAMES.index(f"{s}_ankle_pitch_joint")] = -bend / 2
    return q


def place(mm, dd, q):
    """摆好姿势，并把骨盆抬到“最低的脚底球刚好碰地”的高度。"""
    mujoco.mj_resetData(mm, dd)
    dd.qpos[:] = q
    mujoco.mj_forward(mm, dd)
    dd.qpos[2] -= min(dd.geom_xpos[g, 2] for g in FOOT) - 0.005
    mujoco.mj_forward(mm, dd)


def make_model(kp):
    """用 MjSpec 把 29 个 position 执行器的 kp 统一改掉，kv 仍按 dampratio=1 自动算。"""
    spec = mujoco.MjSpec.from_file(str(XML))
    for a in spec.actuators:
        a.set_to_position(kp=kp, dampratio=1.0)   # forcerange 保持不变
        a.ctrllimited = 0                         # 前馈要折算成目标偏移，不能被关节范围截断
    return spec.compile()


def run(mm, q, tff, T, start_qpos=None, log=False):
    """position 伺服 + 前馈：ctrl = q_des + tff/kp，等价于 τ = tff + kp(q_des−q) − kv·q̇。"""
    dd = mujoco.MjData(mm)
    if start_qpos is None:
        place(mm, dd, q)
    else:
        dd.qpos[:] = start_qpos
    kp = mm.actuator_gainprm[0, 0]
    dd.ctrl[:] = q[7:] + tff / kp
    rows = []
    for i in range(int(round(T / mm.opt.timestep))):
        mujoco.mj_step(mm, dd)
        if log and i % 10 == 0:
            rows.append((dd.time, dd.qpos[7 + KNEE] - q[7 + KNEE], pitch_deg(dd)))
        if dd.qpos[2] < 0.45:                     # 骨盆低于 0.45 m 就算倒了
            return dd, dd.time, rows
    return dd, None, rows


def inverse_static(mm, qpos):
    """在 qpos 处、速度和加速度都为 0 时做逆动力学，返回 qfrc_inverse。"""
    dd = mujoco.MjData(mm)
    dd.qpos[:] = qpos                         # qvel、qacc 新建时就是 0
    mujoco.mj_inverse(mm, dd)
    return dd.qfrc_inverse.copy(), dd


def pitch_deg(dd):
    """骨盆前倾角（度），正值 = 向前倾。"""
    R = dd.xmat[m.body("pelvis").id].reshape(3, 3)
    return np.degrees(np.arctan2(R[0, 2], R[2, 2]))


def settle_with_ff(mm, q, rounds=3, T=3.0):
    """站稳 → 逆动力学 → 把结果当前馈再站稳 → 再算……返回最后一轮的力矩和状态。"""
    tff = np.zeros(m.nu)
    for r in range(rounds):
        dd, fell, _ = run(mm, q, tff, T)
        tau, _ = inverse_static(mm, dd.qpos)
        err = np.abs(dd.qpos[7:] - q[7:]).max()
        print(f"  第 {r + 1} 轮：前馈={'无' if r == 0 else '有'}，最大关节误差 {err:.4f} rad，"
              f"左膝力矩 {tau[6 + KNEE]:+.2f} N·m，骨盆残差 |前6维| 最大 {np.abs(tau[:6]).max():.3f}")
        tff = tau[6:]
    return tff, dd


def main_report():
    t_start = time.time()
    print(f"MuJoCo {mujoco.__version__}，总质量 {MASS:.2f} kg，重力 m·g = {MASS * G:.2f} N")

    # ======== 1. 浮动基：qfrc_inverse 的前 6 维 ========
    print("\n=== 1. 在 stand 姿势、qvel=qacc=0 时做逆动力学：三种情况 ===")
    mujoco.mj_resetDataKeyframe(m, d, KEY)
    mujoco.mj_inverse(m, d)
    print(f"(a) 直接用关键帧（骨盆 z={d.qpos[2]:.3f}，接触 {d.ncon} 个）：前 6 维 {d.qfrc_inverse[:6]}")

    flags = m.opt.disableflags
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT      # 关掉接触 = 悬在空中
    mujoco.mj_inverse(m, d)
    print(f"(b) 关掉接触（悬空）：前 6 维 {d.qfrc_inverse[:6]}")
    print(f"    左腿 6 个关节 {d.qfrc_inverse[6:12]}")
    print(f"    与 qfrc_bias 的最大差 {np.abs(d.qfrc_inverse - d.qfrc_bias).max():.2e}")
    m.opt.disableflags = flags

    mujoco.mj_resetDataKeyframe(m, d, KEY)
    for _ in range(1500):                                         # 用原模型站 3 秒
        mujoco.mj_step(m, d)
    q_settled = d.qpos.copy()
    act_force = d.actuator_force.copy()
    f6, fsum = np.zeros(6), np.zeros(3)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        fsum += f6[:3]
    print(f"(c) 先仿真 3 s 站稳（骨盆 z={d.qpos[2]:.4f}，最大速度 {np.abs(d.qvel).max():.1e}，"
          f"接触法向力合计 {fsum[0]:.2f} N）")
    tau_stand, di = inverse_static(m, q_settled)
    print(f"    前 6 维 {tau_stand[:6]}")
    print(f"    左腿 6 个关节 {tau_stand[6:12]}")
    resid = tau_stand - (di.qfrc_bias - di.qfrc_passive - di.qfrc_constraint)
    print(f"    验证 qfrc_inverse = qfrc_bias − qfrc_passive − qfrc_constraint：最大差 {np.abs(resid).max():.2e}")
    print(f"    逆动力学 vs 伺服实际输出 actuator_force：最大差 {np.abs(tau_stand[6:] - act_force).max():.3f} N·m"
          f"（在 {SHORT[np.abs(tau_stand[6:] - act_force).argmax()]}）")

    # 差别从哪来？看左脚接触力的侧向（y）分量：两只脚可以互相“较劲”而不改变合力
    def left_foot_fy(dd):
        fy, f6 = 0.0, np.zeros(6)
        for i in range(dd.ncon):
            c = dd.contact[i]
            if m.body(m.geom_bodyid[c.geom2]).name.startswith("left"):
                mujoco.mj_contactForce(m, dd, i, f6)
                fy += (c.frame.reshape(3, 3).T @ f6[:3])[1]   # 接触坐标系 → 世界坐标系
        return fy
    print(f"    左脚受到的侧向接触力：仿真中 {left_foot_fy(d):+.3f} N，逆动力学中 {left_foot_fy(di):+.3f} N")

    # 把逆动力学结果当外力加回去、关掉执行器，正向动力学的加速度应该是 0
    d2 = mujoco.MjData(m)
    d2.qpos[:] = q_settled
    d2.qfrc_applied[:] = tau_stand
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    mujoco.mj_forward(m, d2)
    m.opt.disableflags = flags
    print(f"    把它作为 qfrc_applied 加回去、关掉执行器做 mj_forward：|qacc| 最大 {np.abs(d2.qacc).max():.2e}")

    # ======== 2. 力矩与 forcerange：直腿 vs 屈膝 ========
    print("\n=== 2. 屈膝 1.2 rad 的站姿：先站稳，再迭代算前馈 ===")
    m500 = make_model(500)
    q_bent = pose(1.2)
    tau_bent_j, d_bent = settle_with_ff(m500, q_bent)
    tau_stand_j = tau_stand[6:]

    print("\n=== 3. 各关节所需力矩与上限（直腿 stand / 屈膝 1.2 rad） ===")
    print(f"{'关节':<22}{'上限':>6}{'直腿τ':>9}{'余量':>8}{'屈膝τ':>9}{'余量':>8}")
    rows_csv = []
    for i in range(m.nu):
        r1 = 100 * (1 - abs(tau_stand_j[i]) / LIMIT[i])
        r2 = 100 * (1 - abs(tau_bent_j[i]) / LIMIT[i])
        print(f"{SHORT[i]:<22}{LIMIT[i]:>6.0f}{tau_stand_j[i]:>+9.2f}{r1:>7.1f}%{tau_bent_j[i]:>+9.2f}{r2:>7.1f}%")
        rows_csv.append([NAMES[i], LIMIT[i], tau_stand_j[i], r1, tau_bent_j[i], r2])
    for tag, t in (("直腿", tau_stand_j), ("屈膝", tau_bent_j)):
        u = np.abs(t) / LIMIT
        print(f"{tag}：最吃力的关节 {SHORT[u.argmax()]}（占上限 {100 * u.max():.1f}%），"
              f"29 个关节力矩平方和 {np.sum(t ** 2):.1f}")
    with open(OUT / "06_torque_margin.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["joint", "limit", "tau_stand", "margin_stand_%", "tau_bent1.2", "margin_bent_%"])
        w.writerows(rows_csv)

    # ======== 4. 屈膝扫描 ========
    print("\n=== 4. 屈膝角度扫描（kp=500 伺服站稳 3 s 后做逆动力学） ===")
    print(f"{'膝角':>6}{'骨盆z':>8}{'髋pitch':>9}{'膝':>9}{'踝pitch':>9}{'膝占上限':>9}")
    bends = np.round(np.arange(0.0, 1.51, 0.15), 2)
    sweep = []
    for b in bends:
        dd, fell, _ = run(m500, pose(b), np.zeros(m.nu), 3.0)
        t, _ = inverse_static(m500, dd.qpos)
        hip, knee, ank = t[6 + 0], t[6 + 3], t[6 + 4]
        sweep.append((b, dd.qpos[2], hip, knee, ank))
        print(f"{b:>6.2f}{dd.qpos[2]:>8.3f}{hip:>+9.2f}{knee:>+9.2f}{ank:>+9.2f}{100 * abs(knee) / LIMIT[KNEE]:>8.1f}%")

    # ======== 5. 质量矩阵 ========
    print("\n=== 5. 质量矩阵 mj_fullM（直腿站稳的姿势） ===")
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, di, M)
    eig = np.linalg.eigvalsh(M)
    print(f"形状 {M.shape}，非零元素 {np.count_nonzero(M)} / {M.size}，对称误差 {np.abs(M - M.T).max():.1e}，"
          f"特征值范围 [{eig[0]:.4f}, {eig[-1]:.2f}]")
    print("左上 3×3（平移块）:\n", M[:3, :3])
    wy = m.joint("left_wrist_yaw_joint").dofadr[0]
    print(f"左膝 M_ii = {M[6 + KNEE, 6 + KNEE]:.4f}，左腕 yaw M_ii = {M[wy, wy]:.4f}"
          f"（其中转子惯量 armature = {m.dof_armature[wy]}）")
    # dampratio 用的“有效惯量”：在 qpos0 处取 1 / (M⁻¹)_ii
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)                  # 新建的 MjData 就是 qpos0
    M0 = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d0, M0)
    Minv = np.linalg.inv(M0)
    kv_model = -m.actuator_biasprm[:, 2]
    kv_diag = 2 * np.sqrt(500 * np.diag(M0)[6:])
    kv_eff = 2 * np.sqrt(500 / np.diag(Minv)[6:])
    print(f"{'关节':<18}{'模型kv':>8}{'2√(kp·M_ii)':>13}{'2√(kp/(M⁻¹)_ii)':>17}")
    for i in (0, 1, 3, 4, 14, 17):
        print(f"{SHORT[i]:<18}{kv_model[i]:>8.3f}{kv_diag[i]:>13.3f}{kv_eff[i]:>17.3f}")
    print(f"29 个执行器里 |模型kv − 2√(kp/(M⁻¹)_ii)| 最大 {np.abs(kv_model - kv_eff).max():.1e}")

    # ======== 6. 前馈 + 小增益 PD ========
    print("\n=== 6. 增益扫描：从站稳状态出发跑 5 s，有/无前馈 ===")
    tff_stand, d_stand_ff = settle_with_ff(m500, m.key_qpos[KEY], rounds=2)
    # 粗估：把整机看成绕踝关节转动的倒立摆，重力带来的“负刚度”是 m·g·h
    mujoco.mj_forward(m500, d_stand_ff)
    h = d_stand_ff.subtree_com[0, 2] - d_stand_ff.xanchor[m.joint("left_ankle_pitch_joint").id, 2]
    print(f"倒立摆粗估：整机质心比踝 pitch 轴高 h = {h:.3f} m，负刚度 m·g·h = {MASS * G * h:.0f} N·m/rad；"
          f"两个踝 pitch 并联，kp 至少要 {MASS * G * h / 2:.0f}")
    cases = {"直腿": (m.key_qpos[KEY], tff_stand, d_stand_ff.qpos.copy()),
             "屈膝1.2": (q_bent, tau_bent_j, d_bent.qpos.copy())}
    kps = [500, 400, 300, 200, 150, 100, 70, 50]
    models = {kp: make_model(kp) for kp in kps}
    gain_rows = []
    for name, (q, tff, q0) in cases.items():
        print(f"--- {name}（前馈里左膝 {tff[KNEE]:+.2f} N·m）")
        for kp in kps:
            line = f"kp={kp:>3}"
            for ff in (0, 1):
                dd, fell, _ = run(models[kp], q, tff * ff, 5.0, start_qpos=q0)
                e = dd.qpos[7:] - q[7:]
                if fell:
                    txt = f"{fell:.2f} s 时倒下"
                    gain_rows.append([name, kp, ff, "fell", fell, "", "", ""])
                else:
                    txt = (f"膝误差 {e[KNEE]:+.4f}，最大 {np.abs(e).max():.4f}，"
                           f"骨盆前倾 {pitch_deg(dd):+.1f}°")
                    gain_rows.append([name, kp, ff, "ok", "", e[KNEE], np.abs(e).max(), pitch_deg(dd)])
                line += f" | {'有' if ff else '无'}前馈：{txt:<36}"
            print(line)
    with open(OUT / "06_gain_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pose", "kp", "feedforward", "result", "fall_time_s",
                    "knee_err_rad", "max_err_rad", "pelvis_pitch_deg"])
        w.writerows(gain_rows)

    # 屈膝站、kp=300 时有/无前馈的过程曲线
    logs = {}
    for ff in (0, 1):
        _, fell, rows = run(models[300], q_bent, tau_bent_j * ff, 5.0,
                            start_qpos=d_bent.qpos.copy(), log=True)
        logs[ff] = np.array(rows)

    # ======== 画图 ========
    x = np.arange(m.nu)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, 100 * np.abs(tau_stand_j) / LIMIT, 0.38, color=BLUE, label="直腿 stand")
    ax.bar(x + 0.2, 100 * np.abs(tau_bent_j) / LIMIT, 0.38, color=ORANGE, label="屈膝 1.2 rad")
    ax.set_xticks(x, SHORT, rotation=75, fontsize=8)
    ax.set_ylabel("所需力矩 / forcerange（%）")
    ax.set_title("静止站立时各关节用掉了多少电机能力")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "06_torque_margin.png", dpi=130)
    plt.close(fig)

    s = np.array(sweep)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s[:, 0], np.abs(s[:, 3]), "o-", color=ORANGE, lw=2, label="膝")
    ax.plot(s[:, 0], np.abs(s[:, 2]), "s-", color=BLUE, lw=2, label="髋 pitch")
    ax.set_xlabel("膝关节角（rad）")
    ax.set_ylabel("所需力矩大小（N·m，单腿）")
    ax.set_title("蹲得越低，膝盖越吃力")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "06_knee_sweep.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for ff, color, lab in ((0, BLUE, "无前馈"), (1, ORANGE, "有前馈")):
        L = logs[ff]
        axs[0].plot(L[:, 0], L[:, 1], color=color, lw=2, label=lab)
        axs[1].plot(L[:, 0], L[:, 2], color=color, lw=2, label=lab)
    axs[0].set_ylabel("左膝误差 q − q_des（rad）")
    axs[1].set_ylabel("骨盆前倾（°）")
    axs[1].set_xlabel("时间（s）")
    axs[0].set_title("屈膝站，kp=300：有无前馈")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "06_ff_vs_pd.png", dpi=130)
    plt.close(fig)

    print(f"\n图和 CSV 已保存到 {OUT}（06_torque_margin.png / 06_knee_sweep.png / 06_ff_vs_pd.png 等）")
    print(f"总用时 {time.time() - t_start:.1f} s")


def main_once(kp, use_ff, bend, T=5.0):
    """只跑一组：kp=500 加前馈站稳 → 换成 kp 跑 T 秒（有/无前馈各跑一次，选中的那组画粗线）。"""
    t_start = time.time()
    q = pose(bend)
    print(f"屈膝 {bend:g} rad，先用 kp=500 站稳并迭代算前馈：")
    tff, d_ok = settle_with_ff(make_model(500), q)
    print(f"前馈里左膝 {tff[KNEE]:+.2f} N·m（上限 {LIMIT[KNEE]:.0f}，占 {100 * abs(tff[KNEE]) / LIMIT[KNEE]:.1f}%），"
          f"最吃力的关节 {SHORT[np.argmax(np.abs(tff) / LIMIT)]}")
    mm = make_model(kp)
    logs = {}
    for ff in (use_ff, not use_ff):
        dd, fell, rows = run(mm, q, tff * ff, T, start_qpos=d_ok.qpos.copy(), log=True)
        logs[ff] = (np.array(rows), fell)
        e = dd.qpos[7:] - q[7:]
        tag = f"kp={kp:g}，{'有' if ff else '无'}前馈"
        if fell:
            res = f"{fell:.2f} s 时倒下"
        else:
            res = f"站住了：膝误差 {e[KNEE]:+.4f} rad，最大 {np.abs(e).max():.4f} rad，骨盆前倾 {pitch_deg(dd):+.1f}°"
        print(("结果　" if ff == use_ff else "对照　") + f"{tag}：{res}")

    fig, axs = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for ff in (not use_ff, use_ff):
        L, fell = logs[ff]
        style = dict(color=ORANGE if ff else BLUE, lw=2.2 if ff == use_ff else 1.0,
                     ls="-" if ff == use_ff else "--",
                     label=f"{'有' if ff else '无'}前馈" + ("" if ff == use_ff else "（对照）")
                     + (f"，{fell:.2f} s 倒下" if fell else ""))
        axs[0].plot(L[:, 0], L[:, 1], **style)
        axs[1].plot(L[:, 0], L[:, 2], **style)
    axs[0].set_ylabel("左膝误差 q − q_des（rad）")
    axs[1].set_ylabel("骨盆前倾（°）")
    axs[1].set_xlabel("时间（s）")
    axs[0].set_title(f"屈膝 {bend:g} rad，kp={kp:g}")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "06_once.png", dpi=110)
    plt.close(fig)
    print(f"图已保存：{OUT / '06_once.png'}（用时 {time.time() - t_start:.1f} s）")


def main_view(kp, use_ff, bend, web_port=None, T_LOOP=8.0):
    """在查看器里看：屈膝站 + （可选）前馈。web_port 给了端口就用网页查看器，并且每 T_LOOP 秒重来一次。"""
    m500 = make_model(500)
    q = pose(bend)
    print("先离线算前馈……", flush=True)
    tff, d_ok = settle_with_ff(m500, q)
    mm = make_model(kp)
    dd = mujoco.MjData(mm)

    def reset():
        mujoco.mj_resetData(mm, dd)
        dd.qpos[:] = d_ok.qpos
        dd.ctrl[:] = q[7:] + (tff if use_ff else 0) / kp
        mujoco.mj_forward(mm, dd)

    reset()
    print(f"kp={kp:g}，{'有' if use_ff else '无'}前馈。"
          + (f"每 {T_LOOP:.0f} s 重来一次。" if web_port else "关掉窗口结束。"), flush=True)
    if web_port:
        from webview import open_web                # 同目录的 webview.py
        viewer_cm = open_web(mm, dd, web_port, title="第 6 课：前馈 + 小增益 PD")
    else:
        from mujoco import viewer as mj_viewer
        viewer_cm = mj_viewer.launch_passive(mm, dd)
    with viewer_cm as viewer:
        while viewer.is_running():
            t0 = time.time()
            if web_port and dd.time >= T_LOOP:
                reset()
            mujoco.mj_step(mm, dd)
            if web_port:
                viewer.sync(mm, dd)
            else:
                viewer.sync()
            dt = mm.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="打开查看器实时观看")
    ap.add_argument("--web", type=int, metavar="PORT", help="像 --view 一样实时观看，但显示在网页里（http://127.0.0.1:PORT/）")
    ap.add_argument("--once", action="store_true", help="只跑一组（无界面），图写到 output/06_once.png")
    ap.add_argument("--kp", type=float, default=200, help="--once/--view/--web 时使用的 kp")
    ap.add_argument("--noff", action="store_true", help="--once/--view/--web 时不加前馈")
    ap.add_argument("--bend", type=float, default=1.2, help="--once/--view/--web 时的膝关节角（rad）")
    args = ap.parse_args()
    if args.view or args.web:
        main_view(args.kp, not args.noff, args.bend, web_port=args.web)
    elif args.once:
        main_once(args.kp, not args.noff, args.bend)
    else:
        main_report()
