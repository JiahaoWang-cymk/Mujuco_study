"""第 8 课：单腿平衡——把官方 LQR notebook 搬到 G1 上。

流程：
  1. 用逆运动学摆出“左脚独立”的姿态（质心投影落在左脚里）
  2. 高度扫描 + mj_inverse（照搬 notebook），再把倾角也调准，得到真正的平衡点
  3. 把平衡力矩换算成 position 执行器的目标角度 ctrl0
  4. 不加反馈的对照实验
  5. mjd_transitionFD 线性化
  6. 只用 numpy 解离散 Riccati 方程，得到 LQR 增益 K
  7. 闭环站 10 s
  8. 推力测试：不加反馈 vs LQR

用法：
  python 08_lqr_balance.py            # 无界面，打印结果，图和 CSV 写到 output/（约 10 秒）
  python 08_lqr_balance.py --view     # 算完后打开查看器实时观看（每 3 秒推它一下）
  python 08_lqr_balance.py --once --dir left --force 40     # 只推一次（几秒）：不加反馈、LQR 各推一次，图写到 output/08_once.png
  python 08_lqr_balance.py --once --dir right --search      # 推一次，再逐级加力求这个方向的极限
  python 08_lqr_balance.py --view --dir left --force 40     # 跳过对照实验和推力表直接看：每 4 s 重来，第 1 s 推 0.1 s
  python 08_lqr_balance.py --web 8790 --dir left --force 40 # 同上，但显示在网页里：浏览器打开 http://127.0.0.1:8790/
--r-scale（R 乘几倍，练习 1）和 --no-clip（取消 ctrl 截断，练习 2）对所有模式都有效；
--ctrl open 让 --view/--web 只送 ctrl0、不加反馈。
"""
import argparse
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

parser = argparse.ArgumentParser(description="第 8 课：G1 单腿 LQR 平衡")
parser.add_argument("--view", action="store_true", help="算完后打开查看器实时观看")
parser.add_argument("--web", type=int, metavar="PORT", help="像 --view --force 一样实时观看，但显示在网页里（http://127.0.0.1:PORT/）")
parser.add_argument("--once", action="store_true", help="只推一次（无界面），图写到 output/08_once.png")
parser.add_argument("--search", action="store_true", help="--once 时再逐级加力（10 N 一档）求这个方向的极限")
parser.add_argument("--dir", choices=["front", "back", "left", "right"], default="left", help="推的方向，默认 left（+y）")
parser.add_argument("--force", type=float, default=None, help="推力 (N)，推 0.1 s；--once/--web 默认 40")
parser.add_argument("--ctrl", choices=["lqr", "open"], default="lqr", help="--view/--web 的控制：lqr，或 open（只送 ctrl0）")
parser.add_argument("--r-scale", type=float, default=1.0, help="R = r_scale × 单位阵（练习 1），默认 1")
parser.add_argument("--no-clip", action="store_true", help="取消所有执行器的 ctrl 截断（练习 2）")
args = parser.parse_args()
# 快速模式：只算到 LQR 增益 K 为止，跳过对照实验、闭环 10 s、推力表和画图
FAST = args.once or args.web is not None or (args.view and args.force is not None)
if FAST and args.force is None:
    args.force = 40.0
T_START = time.perf_counter()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=3, suppress=True, linewidth=120)

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
nv, nu = m.nv, m.nu
PELVIS = m.body("pelvis").id
TORSO = m.body("torso_link").id
LFOOT = m.body("left_ankle_roll_link").id
FLOOR = m.geom("floor").id


def dof(name):
    return m.joint(name).dofadr[0]


def set_joint(name, value):
    d.qpos[m.joint(name).qposadr[0]] = value


# =====================================================================
# 1. 摆姿势：左脚独立
# =====================================================================
def rot_err(xmat):
    """当前姿态 xmat 到“水平、朝前”（单位阵）的转角误差，世界坐标系，3 维。"""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, xmat)
    e = np.zeros(3)
    mujoco.mju_subQuat(e, np.array([1.0, 0, 0, 0]), q)   # 结果在当前坐标系下
    return xmat.reshape(3, 3) @ e                         # 转到世界坐标系


def find_one_leg_pose():
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    foot_target = d.site("left_foot").xpos.copy()        # 左脚原地不动
    # 左脚 4 个脚底球的 x 在 -0.05～0.12 之间，中点比脚踝 site 靠前 3.5 cm
    com_target = foot_target[:2] + np.array([0.035, 0.0])

    # 手工给定：右腿抬起（屈髋、屈膝），左腿微屈，两臂稍微张开
    set_joint("right_hip_pitch_joint", -0.5)
    set_joint("right_knee_joint", 1.0)
    set_joint("right_ankle_pitch_joint", -0.4)
    set_joint("left_hip_pitch_joint", -0.2)
    set_joint("left_knee_joint", 0.4)
    set_joint("left_ankle_pitch_joint", -0.2)
    set_joint("left_shoulder_roll_joint", 0.4)
    set_joint("right_shoulder_roll_joint", -0.4)
    mujoco.mj_forward(m, d)
    qref = d.qpos.copy()

    # 只让骨盆 6 个自由度 + 左腿 6 个关节 + 腰 roll/pitch 参与求解，其余保持手工值
    free = list(range(6)) + [dof(n) for n in (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "waist_roll_joint", "waist_pitch_joint")]
    mask = np.zeros(nv)
    mask[free] = 1
    reg = 0.01 * mask
    reg[:6] = 0                                           # 骨盆位置不往参考值拉

    jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
    jc, jt = np.zeros((3, nv)), np.zeros((3, nv))
    sL = m.site("left_foot").id
    for _ in range(200):                                  # 阻尼最小二乘，反复迭代
        mujoco.mj_forward(m, d)
        mujoco.mj_jacSite(m, d, jp, jr, sL)
        mujoco.mj_jacSubtreeCom(m, d, jc, PELVIS)         # 整机质心的雅可比
        mujoco.mj_jacBody(m, d, None, jt, TORSO)
        J = np.vstack([jp, jr, 3 * jc[:2], 0.3 * jt[:2]]) * mask
        e = np.concatenate([
            foot_target - d.site_xpos[sL],                # 左脚位置不变
            rot_err(d.site_xmat[sL]),                     # 左脚放平
            3 * (com_target - d.subtree_com[PELVIS][:2]), # 质心投影到左脚中点（权重大）
            0.3 * rot_err(d.xmat[TORSO])[:2],             # 躯干尽量竖直（权重小）
        ])
        back = np.zeros(nv)
        mujoco.mj_differentiatePos(m, back, 1, d.qpos, qref)
        dq = np.linalg.solve(J.T @ J + np.diag(reg) + 1e-6 * np.eye(nv),
                             J.T @ e + reg * back)
        mujoco.mj_integratePos(m, d.qpos, dq, 1)
    mujoco.mj_forward(m, d)
    return d.qpos.copy()


def foot_polygon():
    """左脚 4 个脚底球的世界 xy（按逆时针排好），即单脚支撑多边形。"""
    pts = np.array([d.geom_xpos[g][:2] for g in range(m.ngeom)
                    if m.geom_bodyid[g] == LFOOT and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE])
    c = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return pts[order]


def margin_inside(poly, p):
    """点 p 到凸多边形各边的最小距离；正数表示在里面。"""
    dist = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)   # 逆时针时指向内侧
        dist.append(normal @ (p - a))
    return min(dist)


def room_along(poly, p, u):
    """从点 p 沿水平方向 u 走多远会碰到多边形的边（p 在里面时）。"""
    best = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        M2 = np.column_stack([u, a - b])                  # 解 p + t*u = a + s*(b - a)
        if abs(np.linalg.det(M2)) < 1e-12:
            continue
        t, s = np.linalg.solve(M2, a - p)
        if t > 0 and 0 <= s <= 1:
            best = min(best, t)
    return best


print("=== 1. 摆姿势 ===")
q_pose = find_one_leg_pose()
poly = foot_polygon()
com = d.subtree_com[PELVIS].copy()
print(f"质心 xy = ({com[0]:.3f}, {com[1]:.3f})，高度 {com[2]:.3f} m")
off_mm = (com[:2] - poly.mean(0)) * 1000
print(f"质心相对左脚中点偏 ({off_mm[0]:+.1f}, {off_mm[1]:+.1f}) mm（正则把姿势往手工值拉，任务没有完全满足）")
print(f"质心投影到左脚支撑多边形边缘的最小距离 {margin_inside(poly, com[:2]) * 100:.1f} cm（正 = 在里面）")
print(f"右脚抬高：右脚 site 离地 {d.site('right_foot').xpos[2]:.3f} m（站立时 0.033 m）")
for n in ("left_hip_roll_joint", "left_ankle_roll_joint", "waist_roll_joint"):
    print(f"  {n:<24} = {d.joint(n).qpos[0]:+.3f} rad")


# =====================================================================
# 2. 找平衡点：高度扫描（notebook 的做法）+ 倾角微调
# =====================================================================
def inverse(qpos):
    """在 qpos 处要求 qvel = 0、qacc = 0，返回需要的广义力 qfrc_inverse。"""
    d.qpos[:] = qpos
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse.copy()


print("\n=== 2. 高度扫描 + mj_inverse ===")
offsets = np.linspace(-0.003, 0.003, 3001)                 # ±3 mm，步长 2 μm
fz = np.zeros_like(offsets)
for i, off in enumerate(offsets):
    q = q_pose.copy()
    q[2] += off
    fz[i] = inverse(q)[2]                                  # 需要的“魔法”竖直力
best = np.argmin(np.abs(fz))
weight = m.body_subtreemass[PELVIS] * np.linalg.norm(m.opt.gravity)
print(f"体重 {weight:.1f} N；最佳高度偏移 {offsets[best] * 1000:+.3f} mm，此时竖直魔法力 {fz[best]:.3f} N")
q_height = q_pose.copy()
q_height[2] += offsets[best]
print("只调高度后，骨盆 6 维魔法力 [fx fy fz mx my mz] =", inverse(q_height)[:6].round(3))

# notebook 的人形是双脚之一着地、脚掌宽；G1 单脚只有 4 个小球，
# 剩下的魔法力矩 mx、my 要靠整机微微侧倾 / 前倾来消掉：
# 对 (高度, 绕 x 转, 绕 y 转) 三个量做牛顿迭代，让 (fz, mx, my) 全部归零
ROOT_DOFS = [2, 3, 4]
qpos0 = q_height.copy()
for it in range(6):
    r = inverse(qpos0)[ROOT_DOFS]
    J = np.zeros((3, 3))
    for j, k in enumerate(ROOT_DOFS):                     # 有限差分求雅可比
        step = np.zeros(nv)
        step[k] = 1e-6
        q = qpos0.copy()
        mujoco.mj_integratePos(m, q, step, 1)
        J[:, j] = (inverse(q)[ROOT_DOFS] - r) / 1e-6
    step = np.zeros(nv)
    step[ROOT_DOFS] = np.linalg.solve(J, -r)
    mujoco.mj_integratePos(m, qpos0, step, 1)
qfrc0 = inverse(qpos0)
print("再调倾角后，骨盆 6 维魔法力 =", qfrc0[:6].round(3))
dpos = np.zeros(nv)
mujoco.mj_differentiatePos(m, dpos, 1, q_height, qpos0)
print(f"  调整量：高度 {dpos[2] * 1000:+.3f} mm，侧倾 {np.degrees(dpos[3]):+.3f}°，前倾 {np.degrees(dpos[4]):+.3f}°")

d.qpos[:] = qpos0
mujoco.mj_forward(m, d)
com = d.subtree_com[PELVIS].copy()                         # 以后都以平衡点的质心、脚掌为准
poly = foot_polygon()
print(f"  平衡点处，质心到左脚支撑多边形边缘的最小距离 {margin_inside(poly, com[:2]) * 100:.2f} cm")

tau0 = qfrc0[6:]                                           # 各关节需要的力矩
ratio = np.abs(tau0) / m.actuator_forcerange[:, 1]
for a in np.argsort(-ratio)[:4]:
    print(f"  {m.actuator(a).name:<25} 需要 {tau0[a]:+7.2f} N·m，占上限 {ratio[a] * 100:4.1f}%")

# =====================================================================
# 3. 力矩 → position 执行器的目标角度
# =====================================================================
# position 执行器：force = gain*ctrl + bias0 + bias1*q + bias2*qdot
#                       = kp*ctrl      + 0     - kp*q  - kv*qdot
# 静止时 qdot=0，要让 force = tau，就得 ctrl = (tau - bias0 - bias1*q) / gain
gain = m.actuator_gainprm[:, 0]
bias0, bias1 = m.actuator_biasprm[:, 0], m.actuator_biasprm[:, 1]


def torque_to_ctrl(qpos, tau):
    q_joint = qpos[7:]                                     # 本模型 ctrl 顺序 = qpos[7:] 关节顺序
    return (tau - bias0 - bias1 * q_joint) / gain          # 即 q + tau/kp


print("\n=== 3. 求 ctrl0 ===")
ctrl0 = torque_to_ctrl(qpos0, tau0)
d.qpos[:] = qpos0
d.qvel[:] = 0
d.ctrl[:] = ctrl0
mujoco.mj_forward(m, d)
print(f"代回检验：执行器力与 tau0 最大相差 {np.max(np.abs(d.qfrc_actuator[6:] - tau0)):.2e} N·m")
lo, hi = m.actuator_ctrlrange.T
diff = ctrl0 - qpos0[7:]
print(f"ctrl0 与关节角 q 最大相差 {np.max(np.abs(diff)):.3f} rad（{m.actuator(np.argmax(np.abs(diff))).name}）")
ar = m.actuator("left_ankle_roll_joint").id
print(f"左踝 roll：ctrl0 = {ctrl0[ar]:+.3f}，ctrlrange = [{lo[ar]:+.3f}, {hi[ar]:+.3f}]，"
      f"离最近的限位 {min(ctrl0[ar] - lo[ar], hi[ar] - ctrl0[ar]):.3f} rad")


# =====================================================================
# 4. 不加反馈的对照实验
# =====================================================================
def survive(qpos_init, controller, t_end, push=None):
    """跑一段仿真；倒下或右脚落地就返回 (False, 时间)。push=(t0, 时长, 力向量) 推躯干。"""
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos_init
    mujoco.mj_forward(m, d)
    while d.time < t_end:
        d.ctrl[:] = controller()
        d.xfrc_applied[TORSO, :3] = 0
        if push is not None and push[0] <= d.time < push[0] + push[1]:
            d.xfrc_applied[TORSO, :3] = push[2]
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            return False, d.time                          # 数值发散（MuJoCo 会自动重置，要自己拦住）
        for c in d.contact[:d.ncon]:
            other = c.geom2 if c.geom1 == FLOOR else c.geom1 if c.geom2 == FLOOR else -1
            if other >= 0 and m.geom_bodyid[other] != LFOOT:
                return False, d.time                      # 除左脚外有东西碰到地面
        if d.qpos[2] < 0.5:
            return False, d.time
    return True, d.time


def report(label, result):
    ok, t = result
    print(f"  {label}：{'10 s 都站住了' if ok else f'{t:.2f} s 失败'}")


def max_push(controller, direction, step=10, f_max=400):
    """逐级加力，返回仍能单腿站住的最大推力（N）。"""
    last_ok = 0
    for f in range(step, f_max + 1, step):
        ok, _ = survive(qpos0, controller, 4.0, push=(1.0, 0.1, f * np.asarray(direction, float)))
        if not ok:
            break
        last_ok = f
    return last_ok


DIRS = {"向前 +x": (1, 0, 0), "向后 -x": (-1, 0, 0), "向左 +y": (0, 1, 0), "向右 -y": (0, -1, 0)}

if not FAST:
    print("\n=== 4. 不加反馈会怎样（每种最多跑 10 s）===")
    ctrl_h = torque_to_ctrl(q_height, inverse(q_height)[6:])
    report("A. ctrl = 关节角（只有 PD，不补偿重力）", survive(qpos0, lambda: qpos0[7:], 10.0))
    report("B. ctrl = 只调高度的平衡点（notebook 做法）", survive(q_height, lambda: ctrl_h, 10.0))
    report("C. ctrl = ctrl0（高度 + 倾角都调准）", survive(qpos0, lambda: ctrl0, 10.0))

# =====================================================================
# 5. 线性化 + 代价矩阵
# =====================================================================
print("\n=== 5. 线性化 ===")
mujoco.mj_resetData(m, d)
d.qpos[:] = qpos0
d.ctrl[:] = ctrl0
A = np.zeros((2 * nv, 2 * nv))
B = np.zeros((2 * nv, nu))
mujoco.mjd_transitionFD(m, d, 1e-6, True, A, B, None, None)
eig = np.abs(np.linalg.eigvals(A))
print(f"A: {A.shape}，B: {B.shape}；A 的最大特征值模 {eig.max():.6f}，"
      f"模 > 1.0001 的有 {np.sum(eig > 1.0001)} 个")
for leg in ("left", "right"):
    jn = f"{leg}_knee_joint"
    dv = B[nv + dof(jn), m.actuator(jn).id]
    print(f"B 的含义：{jn} 目标角 +1 rad，一步（{m.opt.timestep * 1000:.0f} ms）后它的角速度变化 {dv:.2f} rad/s")

# 找 A 中“一步之后原样不变”的状态方向：(A - I) x = 0
_, sv, Vt = np.linalg.svd(A - np.eye(2 * nv))
drift = Vt[sv < 1e-6].T                                    # 每一列是一个漂移方向
print(f"A - I 最小的 4 个奇异值 {np.sort(sv)[:4]}，于是有 {drift.shape[1]} 个“漂移方向”：")
for i in range(drift.shape[1]):
    top = np.argsort(-np.abs(drift[:, i]))[:2]
    print(f"  主要成分：qvel 下标 {top[0]}（{drift[top[0], i]:+.3f}）、{top[1]}（{drift[top[1], i]:+.3f}）")

mujoco.mj_forward(m, d)
jac_com = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(m, d, jac_com, PELVIS)
jac_foot = np.zeros((3, nv))
mujoco.mj_jacBodyCom(m, d, jac_foot, None, LFOOT)
jac_diff = jac_com - jac_foot
Q_balance = jac_diff.T @ jac_diff                          # 质心相对左脚的偏移

balance_dofs = [dof(n) for n in (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint")]
other_dofs = np.setdiff1d(np.arange(6, nv), balance_dofs)
BALANCE_COST, BALANCE_JOINT_COST, OTHER_JOINT_COST = 1000, 3, 0.3   # 与 notebook 相同
Q_joint = np.eye(nv)
Q_joint[:6, :6] = 0
Q_joint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Q_joint[other_dofs, other_dofs] *= OTHER_JOINT_COST
Q = np.zeros((2 * nv, 2 * nv))
Q[:nv, :nv] = BALANCE_COST * Q_balance + Q_joint           # 速度不直接惩罚
Q_raw = Q.copy()
# 漂移方向（整机平移 x、y，整机绕竖直轴转）关节控制不了，代价里必须“不在乎”它们，
# 否则 Riccati 方程没有有限解。做法：把 Q 投影到漂移方向的补空间
Pi = np.eye(2 * nv) - drift @ drift.T
Q = Pi @ Q_raw @ Pi
Q = 0.5 * (Q + Q.T)
print(f"漂移方向上的代价 x'Qx：投影前最大 {max(v @ Q_raw @ v for v in drift.T):.1e}，"
      f"投影后最大 {max(abs(v @ Q @ v) for v in drift.T):.1e}")
R = np.eye(nu)                                             # 单位：rad²（目标角），不是 N·m²
if args.r_scale != 1.0:
    R = args.r_scale * np.eye(nu)                          # 练习 1
    print(f"R = {args.r_scale:g} × 单位阵")


# =====================================================================
# 6. 解离散 Riccati 方程（只用 numpy）
# =====================================================================
def riccati_iterate(A, B, Q, R, tol=1e-10, max_iter=20000):
    """最朴素的不动点迭代：把 notebook 里的 Riccati 方程反复代入，直到 P 不再变化。"""
    P = Q.copy()
    for k in range(1, max_iter + 1):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_new = Q + A.T @ P @ (A - B @ K)
        P_new = 0.5 * (P_new + P_new.T)                    # 保持对称，抑制舍入误差
        change = np.max(np.abs(P_new - P)) / np.max(np.abs(P_new))
        P = P_new
        if change < tol:
            return P, k, change
    return P, max_iter, change                             # 没收敛也返回，让调用者看 change


def riccati_doubling(A, B, Q, R, tol=1e-12, max_iter=100):
    """倍增迭代（SDA）：每迭代一次，相当于朴素迭代的步数翻一倍。"""
    n = A.shape[0]
    I = np.eye(n)
    Ak, G, H = A.copy(), B @ np.linalg.solve(R, B.T), Q.copy()
    for k in range(1, max_iter + 1):
        W = np.linalg.solve(I + G @ H, np.hstack([Ak, G]))   # (I+GH)^-1 [A, G]
        Wa, Wg = W[:, :n], W[:, n:]
        H_new = H + Ak.T @ H @ Wa
        G = G + Ak @ Wg @ Ak.T
        Ak = Ak @ Wa
        if np.max(np.abs(H_new - H)) <= tol * np.max(np.abs(H_new)):
            return H_new, k
        H = H_new
    raise RuntimeError("倍增迭代没有收敛")


def gain_from(P):
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def riccati_residual(P):
    rhs = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return np.max(np.abs(rhs - P)) / np.max(np.abs(P))


print("\n=== 6. 解 Riccati 方程 ===")
t0 = time.perf_counter()
P, n_it, change = riccati_iterate(A, B, Q, R)
K = gain_from(P)
print(f"朴素迭代：{n_it} 次收敛，用时 {time.perf_counter() - t0:.2f} s，方程相对残差 {riccati_residual(P):.1e}")
t0 = time.perf_counter()
P2, n_dbl = riccati_doubling(A, B, Q, R)
print(f"倍增迭代：{n_dbl} 次收敛，用时 {time.perf_counter() - t0:.3f} s，"
      f"与朴素迭代的 K 相对差 {np.linalg.norm(gain_from(P2) - K) / np.linalg.norm(K):.1e}")
if not FAST:                                               # 快速模式不做这些核对
    try:
        import scipy.linalg
        Ps = scipy.linalg.solve_discrete_are(A, B, Q, R)
        print(f"（装了 scipy，核对一下：与 solve_discrete_are 的 P 相对差 {np.linalg.norm(Ps - P) / np.linalg.norm(Ps):.1e}）")
        try:
            Ps_raw = scipy.linalg.solve_discrete_are(A, B, Q_raw, R)
            print(f"（scipy 对不投影的 Q 也会给出一个 P，但 max|P| 是投影后的 {np.abs(Ps_raw).max() / np.abs(P).max():.0f} 倍）")
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"（scipy 对不投影的 Q 求解失败：{e}）")
    except ImportError:
        pass
    _, n_raw, change_raw = riccati_iterate(A, B, Q_raw, R, max_iter=3000)
    print(f"对照：不做投影，朴素迭代 {n_raw} 次后 P 每步仍变化 {change_raw:.1e}（收敛判据 1e-10）")
cl = np.sort(np.abs(np.linalg.eigvals(A - B @ K)))[::-1]
print(f"K: {K.shape}；闭环 A-BK 最大的几个特征值模 {cl[:5].round(5)}")
print(f"K 中左踝 roll 一行的最大元素 {np.max(np.abs(K[ar])):.2f}")

# =====================================================================
# 7. 闭环：无扰动站 10 s
# =====================================================================
dq = np.zeros(nv)


def lqr():
    mujoco.mj_differentiatePos(m, dq, 1, qpos0, d.qpos)   # 位置差：nq 维 → nv 维
    return ctrl0 - K @ np.concatenate([dq, d.qvel])


# =====================================================================
# 快速模式（--once / --web / --view --force）：只用上面算好的 qpos0、ctrl0、K
# =====================================================================
DIR_NAME = {"front": "向前 +x", "back": "向后 -x", "left": "向左 +y", "right": "向右 -y"}
T_PUSH, PUSH_DUR, T_TRIAL = 1.0, 0.1, 4.0      # 和推力测试一样：第 1 s 推 0.1 s，看到第 4 s


def failed():
    """和 survive() 相同的判据：数值发散、除左脚外有东西碰地、骨盆低于 0.5 m。"""
    if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
        return True
    for c in d.contact[:d.ncon]:
        other = c.geom2 if c.geom1 == FLOOR else c.geom1 if c.geom2 == FLOOR else -1
        if other >= 0 and m.geom_bodyid[other] != LFOOT:
            return True
    return d.qpos[2] < 0.5


def trace(controller, push_vec):
    """推一次并记录 [时间, 质心 x, 质心 y, 左踝 roll 力矩]；返回 (记录, 失败时刻或 None)。"""
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    mujoco.mj_forward(m, d)
    tr, t_fail = [], None
    while d.time < T_TRIAL:
        d.ctrl[:] = controller()
        d.xfrc_applied[TORSO, :3] = push_vec if T_PUSH <= d.time < T_PUSH + PUSH_DUR else 0
        mujoco.mj_step(m, d)
        if failed():
            t_fail = d.time
            break
        tr.append([d.time, *d.subtree_com[PELVIS][:2], d.actuator_force[ar]])
    return np.array(tr), t_fail


def once():
    name = DIR_NAME[args.dir]
    u = np.asarray(DIRS[name], float)
    f = args.force
    edge = room_along(poly, com[:2], u[:2])                # 质心沿推力方向到脚边的距离
    Mt = m.body_subtreemass[PELVIS]
    w = np.sqrt(np.linalg.norm(m.opt.gravity) / com[2])
    print(f"\n=== 推一次：{name} 推躯干 {f:g} N × {PUSH_DUR} s（t = {T_PUSH} s 起）===")
    print(f"  R = {args.r_scale:g} × 单位阵，ctrl 截断{'已取消' if args.no_clip else '照常'}；"
          f"质心到这一侧脚边 {edge * 100:.1f} cm，倒立摆估计上限 ≈ {Mt * w * edge / PUSH_DUR:.0f} N × 0.1 s")
    runs = {}
    for label, fn in (("不加反馈", lambda: ctrl0), ("LQR", lqr)):
        tr, t_fail = trace(fn, f * u)
        dev = (tr[:, 1:3] - com[:2]) @ u[:2] * 1000         # 质心沿推力方向的偏移 (mm)
        if t_fail is None:
            print(f"  {label}：站住了。质心沿推力方向最多偏 {dev.max():.1f} mm，"
                  f"左踝 roll 力矩峰值 {np.abs(tr[:, 3]).max():.1f} N·m")
        else:
            print(f"  {label}：{t_fail:.2f} s 失败（倒下或右脚碰地）")
        runs[label] = (tr, t_fail, dev)
    if args.search:
        t0 = time.perf_counter()
        f_open, f_lqr = max_push(lambda: ctrl0, u), max_push(lqr, u)
        print(f"  {name} 的极限（10 N 一档）：不加反馈 {f_open} N，LQR {f_lqr} N（用时 {time.perf_counter() - t0:.1f} s）")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    closed = np.vstack([poly, poly[:1]])
    ax[0].plot(closed[:, 0] * 100, closed[:, 1] * 100, "k-o", label="左脚支撑多边形（球心）")
    ax[0].plot(com[0] * 100, com[1] * 100, "r+", ms=12, label="平衡点质心")
    colors = {"不加反馈": "#eb6834", "LQR": "#2a78d6"}
    for label, (tr, t_fail, dev) in runs.items():
        tag = "站住" if t_fail is None else f"{t_fail:.2f} s 失败"
        ax[0].plot(tr[:, 1] * 100, tr[:, 2] * 100, color=colors[label], lw=2, label=f"{label}（{tag}）")
        ax[1].plot(tr[:, 0], dev, color=colors[label], lw=2, label=f"{label}（{tag}）")
    ax[0].set(xlabel="x (cm)", ylabel="y (cm)", title="质心投影的轨迹（只画脚附近）", aspect="equal")
    ax[0].set_xlim(poly[:, 0].min() * 100 - 6, poly[:, 0].max() * 100 + 6)
    ax[0].set_ylim(poly[:, 1].min() * 100 - 10, poly[:, 1].max() * 100 + 6)   # 下面留出放图例的地方
    ax[0].legend(fontsize=8, loc="lower right")
    ax[1].axvspan(T_PUSH, T_PUSH + PUSH_DUR, color="orange", alpha=0.3)
    ax[1].axhline(edge * 1000, color="k", ls=":", label="这一侧的脚边")
    ax[1].set_ylim(min(-20, -0.6 * edge * 1000), max(1.5 * edge * 1000, 20))
    ax[1].set_xlim(0, T_TRIAL)
    ax[1].set(xlabel="时间 (s)（橙带：推的 0.1 s）", ylabel="质心沿推力方向的偏移 (mm)",
              title=f"{name} 推 {f:g} N × {PUSH_DUR} s，R × {args.r_scale:g}")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "08_once.png", dpi=100)
    plt.close(fig)
    print(f"图已保存：{OUT / '08_once.png'}")


def live(web_port=None):
    """实时观看：每 T_TRIAL 秒复位一次，第 T_PUSH 秒推一下。web_port 为 None 时用桌面查看器。"""
    name = DIR_NAME[args.dir]
    push = args.force * np.asarray(DIRS[name], float)
    ctrl_fn = lqr if args.ctrl == "lqr" else (lambda: ctrl0)
    print(f"每 {T_TRIAL:.0f} s 重来一次：第 {T_PUSH} s 起 {name} 推躯干 {args.force:g} N、持续 {PUSH_DUR} s；"
          f"控制：{'LQR' if args.ctrl == 'lqr' else '不加反馈（只送 ctrl0）'}。" + ("" if web_port else "关掉窗口结束。"),
          flush=True)

    def reset():
        mujoco.mj_resetData(m, d)                          # 也会清掉 xfrc_applied 和警告计数
        d.qpos[:] = qpos0
        mujoco.mj_forward(m, d)

    reset()
    pushing = False
    if web_port:
        from webview import open_web                       # 同目录的 webview.py
        viewer_cm = open_web(m, d, web_port, title="第 8 课：单腿 LQR 平衡")
    else:
        from mujoco import viewer as mj_viewer
        viewer_cm = mj_viewer.launch_passive(m, d)
    with viewer_cm as viewer:
        while viewer.is_running():
            start = time.time()
            if d.time >= T_TRIAL or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                reset()                                    # 到时间了，或者摔得数值发散了
                pushing = False
            d.ctrl[:] = ctrl_fn()
            now = T_PUSH <= d.time < T_PUSH + PUSH_DUR
            if now:
                d.xfrc_applied[TORSO, :3] = push
            elif pushing:                                  # 推完清零一次，不干扰鼠标施加的力
                d.xfrc_applied[TORSO, :3] = 0
            pushing = now
            mujoco.mj_step(m, d)
            if web_port:
                viewer.sync(m, d)
            else:
                viewer.sync()
            wait = m.opt.timestep - (time.time() - start)
            if wait > 0:
                time.sleep(wait)


if FAST:
    if args.no_clip:
        m.actuator_ctrllimited[:] = 0                      # 练习 2
    print(f"\n准备完成（用时 {time.perf_counter() - T_START:.1f} s）", flush=True)
    if args.once:
        once()
    else:
        live(args.web)
    sys.exit(0)


print("\n=== 7. LQR 闭环，无扰动 10 s ===")
mujoco.mj_resetData(m, d)
d.qpos[:] = qpos0
log = []
while d.time < 10.0:
    u = lqr()
    d.ctrl[:] = u
    mujoco.mj_step(m, d)
    log.append([d.time, *d.subtree_com[PELVIS], d.qpos[2], np.max(np.abs(u - ctrl0)),
                np.sum((u < lo) | (u > hi)), d.ncon])
log = np.array(log)
com_dev = np.linalg.norm(log[:, 1:3] - com[:2], axis=1)
print(f"10 s 后骨盆高度 {d.qpos[2]:.3f} m，接触点 {d.ncon} 个（全是左脚）")
print(f"质心水平偏离平衡点：最大 {com_dev.max() * 1e6:.2f} μm，最后 {com_dev[-1] * 1e6:.2f} μm")
print(f"|ctrl - ctrl0| 最大 {log[:, 5].max():.4f} rad；被 ctrlrange 截断的步数 {int(np.sum(log[:, 6] > 0))}")
np.savetxt(OUT / "08_lqr_free.csv", log, delimiter=",", fmt="%.6f",
           header="time,com_x,com_y,com_z,pelvis_z,max_dctrl,n_clipped,ncon", comments="")


# =====================================================================
# 8. 抗扰测试：t=1 s 时推躯干 0.1 s，再观察 3 s（max_push 和 DIRS 定义在第 4 部分后面）
# =====================================================================
if args.no_clip:
    m.actuator_ctrllimited[:] = 0                          # 练习 2：取消所有执行器的 ctrl 截断
    print("\n（已取消所有执行器的 ctrl 截断）")
print("\n=== 8. 推力测试（推躯干 0.1 s，之后单腿站住 3 s 算成功）===")
# 粗估：把机器人看成倒立摆。冲量 I 让质心获得速度 v = I/M，
# 质心还要再“冲出” v/ω 才停得住（ω = sqrt(g/h)，捕获点），这段距离不能超过脚边
M = m.body_subtreemass[PELVIS]
omega = np.sqrt(np.linalg.norm(m.opt.gravity) / com[2])
print(f"  （M = {M:.2f} kg，质心高 h = {com[2]:.3f} m，ω = {omega:.3f} 1/s）")
# 从平衡点质心沿推力方向走到脚边（支撑多边形的边）的距离
room = {name: room_along(poly, com[:2], np.asarray(vec[:2], float)) for name, vec in DIRS.items()}
t0 = time.perf_counter()
results = {}
print(f"  {'方向':<8}{'不加反馈':>8}{'LQR':>8}   质心到脚边   倒立摆估计上限")
for name, vec in DIRS.items():
    f_open = max_push(lambda: ctrl0, vec)
    f_lqr = max_push(lqr, vec)
    results[name] = (f_open, f_lqr)
    print(f"  {name:<8}{f_open:>6} N{f_lqr:>6} N   {room[name] * 100:6.1f} cm   "
          f"{M * omega * room[name]:5.1f} N·s（≈ {M * omega * room[name] / 0.1:.0f} N × 0.1 s）")
print(f"  用时 {time.perf_counter() - t0:.1f} s")
with open(OUT / "08_push_test.csv", "w", encoding="utf-8") as f:
    f.write("direction,open_loop_N,lqr_N,room_cm\n")
    for name, (fo, fl) in results.items():
        f.write(f"{name},{fo},{fl},{room[name] * 100:.2f}\n")

# 记录一次侧推的过程，用来画图：用 LQR 能扛住、不加反馈扛不住的力
f_demo = results["向左 +y"][1]
traces = {}
for label, ctrl_fn in (("不加反馈", lambda: ctrl0), ("LQR", lqr)):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    tr = []
    while d.time < 4.0:
        d.ctrl[:] = ctrl_fn()
        d.xfrc_applied[TORSO, :3] = [0, f_demo, 0] if 1.0 <= d.time < 1.1 else 0
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            break
        tr.append([d.time, *d.subtree_com[PELVIS][:2], d.actuator_force[ar]])
        if d.qpos[2] < 0.5:
            break
    traces[label] = np.array(tr)
lq = traces["LQR"]
print(f"\n演示：向左推 {f_demo} N × 0.1 s")
print(f"  LQR：质心 y 最多偏 {np.max(np.abs(lq[:, 2] - com[1])) * 1000:.1f} mm（离脚边 {room['向左 +y'] * 1000:.0f} mm），"
      f"左踝 roll 力矩峰值 {np.max(np.abs(lq[:, 3])):.1f} N·m（上限 {m.actuator_forcerange[ar, 1]:.0f}）")
dev = (lq[:, 2] - com[1]) * 1000
settle = lq[np.where(np.abs(dev) >= 1.0)[0].max(), 0]
print(f"       {lq[np.argmax(dev), 0]:.2f} s 到最远，{lq[np.argmin(dev), 0]:.2f} s 反向冲过头 {dev.min():.1f} mm，"
      f"{settle:.2f} s 以后一直在 1 mm 以内")
op = traces["不加反馈"]
t_out = op[np.argmax([margin_inside(poly, p) < 0 for p in op[:, 1:3]]), 0]   # 质心第一次出多边形
print(f"  不加反馈：质心在 {t_out:.2f} s 越过脚边，{op[-1, 0]:.2f} s 骨盆低于 0.5 m")
np.savetxt(OUT / "08_push_demo.csv", np.hstack([lq, np.full((len(lq), 1), f_demo)]), delimiter=",",
           fmt="%.6f", header="time,com_x,com_y,ankle_roll_force,push_N", comments="")

# =====================================================================
# 9. 画图
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(offsets * 1000, fz)
ax[0].axvline(offsets[best] * 1000, color="r", ls="--", label=f"最佳 {offsets[best] * 1000:+.3f} mm")
ax[0].axhline(weight, color="g", ls="--", label=f"体重 {weight:.0f} N")
ax[0].set(xlabel="高度偏移 (mm)", ylabel="竖直魔法力 (N)", title="高度扫描（mj_inverse）")
ax[0].legend()
closed = np.vstack([poly, poly[:1]])
ax[1].plot(closed[:, 0] * 100, closed[:, 1] * 100, "k-o", label="左脚支撑多边形（球心）")
ax[1].plot(com[0] * 100, com[1] * 100, "r+", ms=12, label="平衡点质心")
for label, tr in traces.items():
    ax[1].plot(tr[:, 1] * 100, tr[:, 2] * 100, lw=2, label=f"侧推后质心轨迹（{label}）")
ax[1].set(xlabel="x (cm)", ylabel="y (cm)", title="质心投影与支撑多边形", aspect="equal")
ax[1].set_ylim(poly[:, 1].min() * 100 - 7, poly[:, 1].max() * 100 + 3)   # 下面留出放图例的地方
ax[1].legend(fontsize=8, loc="lower right")
for label, tr in traces.items():
    ax[2].plot(tr[:, 0], (tr[:, 2] - com[1]) * 1000, label=f"{label}：质心 y 偏移")
ax[2].axvspan(1.0, 1.1, color="orange", alpha=0.3, label=f"向左推 {f_demo} N")
ax[2].set(xlabel="时间 (s)", ylabel="mm", title="侧推之后")
ax[2].axhline(room["向左 +y"] * 1000, color="k", ls=":", label="左脚外侧边缘")
ax[2].set_ylim(-60, 60)
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "08_lqr_balance.png", dpi=120)
print(f"\n图已保存：{OUT / '08_lqr_balance.png'}")


# =====================================================================
# 10. 实时观看（可选）
# =====================================================================
def view():
    import mujoco.viewer

    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    mujoco.mj_forward(m, d)
    # 每个方向用该方向 LQR 极限的 80%，左右前后轮换
    plan = [(name, 0.8 * results[name][1] * np.array(DIRS[name], float)) for name in DIRS]
    print("查看器：每 3 s 推一次躯干（0.1 s），力的大小取各方向 LQR 极限的 80%。关掉窗口结束。")
    print("  也可以自己双击选中某个部件，按住 Ctrl + 右键拖动去推它。")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            start = time.time()
            d.ctrl[:] = lqr()
            n = int(d.time // 3.0)
            if n >= 1 and d.time - 3.0 * n < 0.1:
                d.xfrc_applied[TORSO, :3] = plan[(n - 1) % 4][1]
            elif n >= 1 and d.time - 3.0 * n < 0.1 + m.opt.timestep:
                d.xfrc_applied[TORSO, :3] = 0                  # 推完清零一次，不干扰鼠标施加的力
            mujoco.mj_step(m, d)
            viewer.sync()
            wait = m.opt.timestep - (time.time() - start)
            if wait > 0:
                time.sleep(wait)


if args.view:
    view()
