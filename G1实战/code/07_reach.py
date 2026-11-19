"""第 7 课：站着的 G1 伸出右手去碰一个球。

流程：
1. 用 MjSpec 给右手加一个末端 site，在场景里加一个 mocap 小球当目标；
2. 用 mj_jacSite 求雅可比，做阻尼最小二乘（DLS）逆运动学，只动右臂 7 个关节；
3. 把 IK 结果当作位置伺服的目标 ctrl，平滑过渡过去，看手是不是真的碰到了球；
4. 对照实验：闭环修正、阻尼大小、够不着的点、加上腰、让球参与碰撞。

用法：
    python 07_reach.py           # 无界面运行，打印结果并保存图
    python 07_reach.py --view    # 打开查看器，依次去碰 A、B、C、D 四个球
"""
import argparse
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

BALL_R = 0.04          # 目标球半径 (m)
PALM_R = 0.02          # 实验 5 里给手掌补的碰撞球半径 (m)
PALM_POS = [0.13, 0.0, 0.0]   # 末端点在 right_wrist_yaw_link 坐标系里的位置

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
       "right_wrist_yaw_joint"]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# 目标点（世界坐标，m）。G1 面朝 +x，右手在 -y 一侧
TARGETS = {
    "A": [0.30, -0.20, 1.00],   # 胸前
    "B": [0.25, -0.30, 0.75],   # 右前下方
    "C": [0.05, -0.45, 1.25],   # 右侧高处
    "D": [0.80, -0.20, 1.00],   # 正前方 0.8 m，太远了
}
TARGET_E = [0.55, -0.20, 0.95]  # 胳膊差一点、弯腰能够到


# ================================================================ 1. 改模型
def build_model(ball_collides=False):
    """在 G1 场景上加：右手末端 site、目标球；可选让球和手掌参与碰撞。"""
    spec = mujoco.MjSpec.from_file(str(XML))
    hand = spec.body("right_wrist_yaw_link")
    # 橡胶手的外形在手腕坐标系 x≈0.04~0.17 m，末端点放在手掌中部偏前
    hand.add_site(name="right_palm", pos=PALM_POS, size=[0.01, 0, 0],
                  rgba=[0, 1, 0, 1], group=0)

    # 目标球放在 mocap body 上：不受力、不会掉，用 d.mocap_pos 随时摆放
    target = spec.worldbody.add_body(name="target", mocap=True, pos=TARGETS["A"])
    if ball_collides:
        # 只用第 2 位（值 2）：只和同样带第 2 位的 geom 碰撞
        target.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[BALL_R, 0, 0], rgba=[1, 0.5, 0, 1],
                        contype=2, conaffinity=2)
        # 橡胶手只是外观（contype=0），这里补一个碰撞球当“手掌”，同样只用第 2 位
        hand.add_geom(name="palm_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      pos=PALM_POS, size=[PALM_R, 0, 0], rgba=[0, 1, 0, 0.3],
                      contype=2, conaffinity=2, density=0, group=3)
    else:
        # 纯标记：contype=conaffinity=0，和谁都不碰
        target.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[BALL_R, 0, 0], rgba=[1, 0.5, 0, 0.6],
                        contype=0, conaffinity=0)
    return spec.compile()


# ================================================================ 2. 逆运动学
def solve_ik(m, d0, target, joints, damping=0.05, max_iter=200, tol=1e-4, max_step=0.2):
    """阻尼最小二乘 IK。只改 joints 里的关节，其余（包括浮动基）保持 d0 当前的值。

    返回 (关节角, 每次迭代的误差, 每次迭代的最大关节步长)。
    """
    d = mujoco.MjData(m)                       # 单独一份数据，只做运动学，不碰 d0
    d.qpos[:] = d0.qpos
    sid = m.site("right_palm").id
    jids = [m.joint(n).id for n in joints]
    qadr = m.jnt_qposadr[jids]
    vadr = m.jnt_dofadr[jids]
    lo, hi = m.jnt_range[jids].T
    jacp = np.zeros((3, m.nv))
    errs, steps = [], []
    for it in range(max_iter + 1):             # 最多走 max_iter 步；多一轮只为量出最后一步之后的误差
        mujoco.mj_kinematics(m, d)             # 更新各 body / site 的位姿
        mujoco.mj_comPos(m, d)                 # 雅可比要用到的中间量
        err = target - d.site_xpos[sid]
        errs.append(np.linalg.norm(err))
        if errs[-1] < tol or it == max_iter:   # 够准了，或步数用完了
            break
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:, vadr]                      # 3×k：只取要动的那几列
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        s = np.abs(dq).max()                   # 一步太大就整体缩小，防止跳过头
        if s > max_step:
            dq *= max_step / s
        steps.append(np.abs(dq).max())
        # 这里全是铰链关节，qpos 与 qvel 一一对应，可以直接加；有球关节时要用 mj_integratePos
        d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
    return d.qpos[qadr].copy(), errs, steps


# ================================================================ 3. 仿真
def settle(m, d, seconds=0.5):
    """从 stand 关键帧出发，先站稳一会儿。"""
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def reach(m, d, joints, q_goal, ramp=1.0, hold=2.0, viewer=None):
    """把 joints 的伺服目标从当前 ctrl 平滑移到 q_goal，再保持 hold 秒。

    返回 (时间, 手掌 site 到球心的距离) 两个数组。
    """
    aids = [m.actuator(n).id for n in joints]  # G1 的执行器和关节同名
    c0 = d.ctrl[aids].copy()
    sid = m.site("right_palm").id
    mid = m.body_mocapid[m.body("target").id]
    t0, wall0 = d.time, time.time()
    ts, dist = [], []
    for k in range(int(round((ramp + hold) / m.opt.timestep))):
        s = min((d.time - t0) / ramp, 1.0)
        s = 3 * s**2 - 2 * s**3                # smoothstep：起点和终点速度都是 0
        d.ctrl[aids] = c0 + s * (q_goal - c0)
        mujoco.mj_step(m, d)
        ts.append(d.time - t0)
        dist.append(np.linalg.norm(d.site_xpos[sid] - d.mocap_pos[mid]))
        if viewer is not None and k % 8 == 0:  # 约 60 Hz 刷新画面
            if not viewer.is_running():
                break
            viewer.sync()
            lag = (d.time - t0) - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)                # 让仿真和真实时间同速
    return np.array(ts), np.array(dist)


def correct(m, d, target, joints, rounds=3, viewer=None):
    """闭环修正：从“实际站着的状态”再做一次 IK，把差值加到 ctrl 上。返回每轮末的距离。"""
    aids = [m.actuator(n).id for n in joints]
    qadr = m.jnt_qposadr[[m.joint(n).id for n in joints]]
    out = []
    for _ in range(rounds):
        q_need, _, _ = solve_ik(m, d, target, joints)
        goal = d.ctrl[aids] + (q_need - d.qpos[qadr])
        _, dist = reach(m, d, joints, goal, ramp=0.5, hold=1.0, viewer=viewer)
        out.append(dist[-1])
    return out


def run_view():
    """--view：在查看器里依次去碰 A、B、C、D 四个球，关掉窗口结束。"""
    import mujoco.viewer

    m = build_model()
    d = mujoco.MjData(m)
    mid = m.body_mocapid[m.body("target").id]
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for name, tgt in TARGETS.items():
                tgt = np.array(tgt)
                # 和无界面模式一样，每个点都从站姿出发；否则 IK 从上一个点的姿态起步，
                # 可能卡在局部极小值里（实测连续去碰时 B 点会停在 120 mm 左右）
                settle(m, d)
                d.mocap_pos[mid] = tgt
                viewer.sync()
                q, errs, _ = solve_ik(m, d, tgt, ARM)
                print(f"{name}: IK 误差 {errs[-1]*1000:.2f} mm，伸手……")
                _, dist = reach(m, d, ARM, q, viewer=viewer)
                if not viewer.is_running():
                    return
                print(f"   手到球心 {dist[-1]*1000:.1f} mm")


# ================================================================ 主程序
def main():
    np.set_printoptions(precision=3, suppress=True)   # 数组打印成小数，不用科学计数法
    m = build_model()
    d = mujoco.MjData(m)
    sid = m.site("right_palm").id
    mid = m.body_mocapid[m.body("target").id]
    print(f"加了 site 和球之后：nsite={m.nsite}  ngeom={m.ngeom}  nmocap={m.nmocap}")
    print(f"机器人质量 {m.body_subtreemass[m.body('pelvis').id]:.2f} kg（和原模型一样）")

    settle(m, d)
    sh = d.body("right_shoulder_pitch_link").xpos
    print(f"站稳后：手掌 site {d.site_xpos[sid].round(3)}，右肩 {sh.round(3)}")

    # ---------- 实验 1：四个目标点，只动右臂 7 个关节，开环执行 ----------
    print("\n=== 实验 1：右臂 7 关节 DLS（阻尼 0.05），IK 结果直接当 ctrl ===")
    all_errs, rows = {}, []
    for name, tgt in TARGETS.items():
        tgt = np.array(tgt)
        settle(m, d)                           # 每次都从站姿出发
        q, errs, _ = solve_ik(m, d, tgt, ARM)
        all_errs[name] = errs
        print(f"\n[{name}] 目标 {tgt}，离右肩 {np.linalg.norm(tgt - sh):.3f} m")
        show = sorted({0, 1, 2, 3, 5, 10, 20, 50, len(errs) - 1})
        print("  迭代:误差(mm)  " + "  ".join(
            f"{i}:{errs[i]*1000:.2f}" for i in show if i < len(errs)))
        print(f"  共 {len(errs)-1} 次迭代，IK 末误差 {errs[-1]*1000:.2f} mm")
        print(f"  关节角(rad) {q.round(3)}")
        d.mocap_pos[mid] = tgt
        ts, dist = reach(m, d, ARM, q)
        hit = dist.min() < BALL_R
        print(f"  仿真：最近 {dist.min()*1000:.1f} mm，3 s 末 {dist[-1]*1000:.1f} mm，"
              f"{'碰到了' if hit else '没碰到'}；骨盆高 {d.body('pelvis').xpos[2]:.3f} m")
        rows.append((name, ts, dist))

    # ---------- 实验 2：闭环修正 ----------
    print("\n=== 实验 2：A 点，按实际状态再做 IK 修正 3 轮 ===")
    settle(m, d)
    tgt = np.array(TARGETS["A"])
    d.mocap_pos[mid] = tgt
    q_stand = d.qpos.copy()                    # IK 时假设的身体姿态
    q, _, _ = solve_ik(m, d, tgt, ARM)
    _, dist = reach(m, d, ARM, q)
    print(f"  开环：{dist[-1]*1000:.2f} mm")

    # 误差从哪来？把“手臂没跟上”和“身体动了”分开算（只做运动学）
    qadr = m.jnt_qposadr[[m.joint(n).id for n in ARM]]
    def palm_err(qpos):
        dk = mujoco.MjData(m)
        dk.qpos[:] = qpos
        mujoco.mj_kinematics(m, dk)
        return np.linalg.norm(dk.site_xpos[sid] - tgt) * 1000
    only_arm = q_stand.copy()
    only_arm[qadr] = d.qpos[qadr]              # 身体照旧，手臂用实际角度
    only_body = d.qpos.copy()
    only_body[qadr] = q                        # 身体用实际姿态，手臂用 IK 角度
    print(f"  手臂关节跟踪误差 {np.abs(q - d.qpos[qadr]).max():.4f} rad → 单独造成 {palm_err(only_arm):.1f} mm")
    print(f"  骨盆前移 {(d.qpos[0] - q_stand[0])*1000:.1f} mm、腿和腰最多转了 "
          f"{np.abs(d.qpos[7:22] - q_stand[7:22]).max():.4f} rad → 单独造成 {palm_err(only_body):.1f} mm")
    for k, e in enumerate(correct(m, d, tgt, ARM), 1):
        print(f"  修正第 {k} 轮后：{e*1000:.2f} mm")

    # ---------- 实验 3：阻尼大小 ----------
    print("\n=== 实验 3：阻尼大小（A 能够到，D 够不着） ===")
    settle(m, d)
    print(f"  {'阻尼':>6} | A 迭代次数 | D 末误差(mm) | D 最后 20 步最大步长(rad)")
    for lam in [0.001, 0.05, 0.3]:
        _, ea, _ = solve_ik(m, d, np.array(TARGETS["A"]), ARM, damping=lam)
        qd, ed, sd = solve_ik(m, d, np.array(TARGETS["D"]), ARM, damping=lam)
        print(f"  {lam:>6} | {len(ea)-1:>10} | {ed[-1]*1000:>12.1f} | {max(sd[-20:]):.4f}"
              f"   （D 的肘角 {qd[3]:+.3f}）")

    # 手臂最长能伸多远？在站姿上扫一遍肘角，量肩到手掌的直线距离
    dk = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, dk, m.key("stand").id)
    elbow = m.jnt_qposadr[m.joint("right_elbow_joint").id]
    best = (0.0, 0.0)
    for e in np.arange(0.0, 2.05, 0.05):
        dk.qpos[elbow] = e
        mujoco.mj_kinematics(m, dk)
        L = np.linalg.norm(dk.site_xpos[sid] - dk.body("right_shoulder_pitch_link").xpos)
        best = max(best, (L, e))
    print(f"  参考：肘角 {best[1]:.2f} rad 时肩到手掌最远，{best[0]:.3f} m")

    # ---------- 实验 4：加上腰 ----------
    print("\n=== 实验 4：右臂 7 + 腰 3 = 10 个关节 ===")
    tgt = np.array(TARGET_E)
    settle(m, d)
    print(f"  E 点 {tgt}，离右肩 {np.linalg.norm(tgt - sh):.3f} m")
    _, ea, _ = solve_ik(m, d, tgt, ARM)
    q, eb, _ = solve_ik(m, d, tgt, ARM + WAIST)
    print(f"  只用手臂：{len(ea)-1} 次迭代，末误差 {ea[-1]*1000:.1f} mm")
    print(f"  加上腰  ：{len(eb)-1} 次迭代，末误差 {eb[-1]*1000:.2f} mm，"
          f"腰 yaw/roll/pitch = {q[7:].round(3)}")
    d.mocap_pos[mid] = tgt
    _, dist = reach(m, d, ARM + WAIST, q)
    print(f"  仿真开环：{dist[-1]*1000:.1f} mm；", end="")
    fix = correct(m, d, tgt, ARM + WAIST)
    print("修正 3 轮：" + " → ".join(f"{e*1000:.1f}" for e in fix) +
          f" mm；骨盆高 {d.body('pelvis').xpos[2]:.3f} m")

    settle(m, d)
    tgt = np.array(TARGETS["D"])
    q, ed, _ = solve_ik(m, d, tgt, ARM + WAIST)
    d.mocap_pos[mid] = tgt
    _, dist = reach(m, d, ARM + WAIST, q)
    print(f"  D 点加腰：IK 末误差 {ed[-1]*1000:.1f} mm，腰 = {q[7:].round(3)}；"
          f"仿真 3 s 后骨盆高 {d.body('pelvis').xpos[2]:.3f} m")

    # ---------- 实验 5：让球参与碰撞 ----------
    print("\n=== 实验 5：球和手掌碰撞球都设 contype=conaffinity=2 ===")
    mc = build_model(ball_collides=True)
    dc = mujoco.MjData(mc)
    wrist = mc.body("right_wrist_yaw_link").id
    wrist_gid = [i for i in range(mc.ngeom)          # 手腕那个碰撞 mesh 没有名字，按 body 找
                 if mc.geom_bodyid[i] == wrist and mc.geom_contype[i]][0]
    for label, gid in [("ball", mc.geom("ball").id), ("palm_collision", mc.geom("palm_collision").id),
                       ("floor", mc.geom("floor").id), ("手腕碰撞 mesh", wrist_gid)]:
        print(f"  {label:<15} contype={mc.geom_contype[gid]}  conaffinity={mc.geom_conaffinity[gid]}")
    settle(mc, dc)
    tgt = np.array(TARGETS["A"])
    dc.mocap_pos[mc.body_mocapid[mc.body("target").id]] = tgt
    q, errs, _ = solve_ik(mc, dc, tgt, ARM)      # IK 仍然瞄准球心
    _, dist = reach(mc, dc, ARM, q)
    ball_gid = mc.geom("ball").id
    fn, pairs = 0.0, []
    for i in range(dc.ncon):
        c = dc.contact[i]
        if ball_gid in (c.geom1, c.geom2):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(mc, dc, i, f6)
            fn += f6[0]
            pairs.append(f"{mc.geom(c.geom1).name}-{mc.geom(c.geom2).name}")
    print(f"  IK 仍瞄准球心，末误差 {errs[-1]*1000:.2f} mm（IK 不知道有碰撞）")
    print(f"  仿真 3 s 末：手掌 site 到球心 {dist[-1]*1000:.1f} mm"
          f"（两球半径和 {(BALL_R+PALM_R)*1000:.0f} mm），接触 {pairs}，法向力 {fn:.2f} N")

    # ---------- 画图 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, errs in all_errs.items():
        ax[0].semilogy(np.array(errs) * 1000, label=name)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("IK error (mm)")
    ax[0].set_title("DLS IK convergence (right arm, 7 joints)")
    ax[0].set_xlim(0, 60)                      # D 后面 140 步都是平的，不画了
    ax[0].legend()
    for name, ts, dist in rows:
        ax[1].plot(ts, dist * 1000, label=name)
    ax[1].axhline(BALL_R * 1000, color="gray", ls="--", label="ball radius")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("palm-to-ball distance (mm)")
    ax[1].set_title("Simulation: position servos track IK result")
    ax[1].set_ylim(0, 1000)
    ax[1].legend()
    fig.tight_layout()
    png = OUT / "07_reach.png"
    fig.savefig(png, dpi=120)
    print(f"\n图已保存：{png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="打开查看器观看")
    args = ap.parse_args()
    if args.view:
        run_view()
    else:
        main()
