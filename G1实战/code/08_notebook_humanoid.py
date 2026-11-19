"""第 8 课附带脚本：按官方 LQR notebook 的步骤处理官方 humanoid，和 G1 对照。

notebook 的 humanoid 用 motor 执行器（ctrl 就是力矩），G1 用 position 执行器（ctrl 是目标角）。
这个脚本只做三件事：找平衡控制量、不加反馈跑一跑、看线性化矩阵 A 有几个不稳定特征值。
"""
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = [
    ROOT.parent / "官方仓库_模型与教程",                     # 资料包里的位置
    ROOT.parent.parent / "vendor" / "官方仓库_模型与教程",   # 源码目录里的位置
]
for base in CANDIDATES:
    XML = base / "model" / "humanoid" / "humanoid.xml"
    if XML.exists():
        break
else:
    sys.exit("找不到 官方仓库_模型与教程/model/humanoid/humanoid.xml，请确认资料包完整")

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
KEY = 1                                                     # notebook 用的单腿站立关键帧
print(f"humanoid：nu={m.nu}，执行器类型 gaintype={m.actuator_gaintype[0]}（0 = fixed，即 motor），"
      f"gear={m.actuator_gear[0, 0]:.0f}，时间步长 {m.opt.timestep * 1000:.0f} ms")

# 1. 高度扫描（与 notebook 相同：±1 mm，步长 1 μm）
offsets = np.linspace(-0.001, 0.001, 2001)
fz = []
for off in offsets:
    mujoco.mj_resetDataKeyframe(m, d, KEY)
    mujoco.mj_forward(m, d)
    d.qacc = 0
    d.qpos[2] += off
    mujoco.mj_inverse(m, d)
    fz.append(d.qfrc_inverse[2])
best = offsets[np.argmin(np.abs(fz))]
mujoco.mj_resetDataKeyframe(m, d, KEY)
mujoco.mj_forward(m, d)
d.qacc = 0
d.qpos[2] += best
qpos0 = d.qpos.copy()
mujoco.mj_inverse(m, d)
qfrc0 = d.qfrc_inverse.copy()
print(f"最佳高度偏移 {best * 1000:+.3f} mm，骨盆 6 维魔法力 {qfrc0[:6].round(3)}")

# 2. motor 执行器：力 = gear * ctrl，用力臂矩阵的伪逆“除回去”
moment = np.zeros((m.nu, m.nv))
mujoco.mju_sparse2dense(moment, d.actuator_moment.reshape(-1), d.moment_rownnz,
                        d.moment_rowadr, d.moment_colind.reshape(-1))
ctrl0 = (np.atleast_2d(qfrc0) @ np.linalg.pinv(moment)).flatten()

# 3. 不加反馈，只给 ctrl0
mujoco.mj_resetData(m, d)
d.qpos = qpos0
d.ctrl = ctrl0
torso = m.body("torso").id
mujoco.mj_forward(m, d)
z0 = d.xpos[torso][2]
while d.time < 5.0 and d.xpos[torso][2] > z0 - 0.2:
    mujoco.mj_step(m, d)
print(f"只给 ctrl0、不加反馈：躯干起始高度 {z0:.2f} m，"
      f"{'5 s 内下降不到 0.2 m' if d.time >= 5.0 else f'{d.time:.2f} s 时已下降 0.2 m'}")

# 4. 线性化，看 A 的特征值
mujoco.mj_resetData(m, d)
d.qpos = qpos0
d.ctrl = ctrl0
A = np.zeros((2 * m.nv, 2 * m.nv))
B = np.zeros((2 * m.nv, m.nu))
mujoco.mjd_transitionFD(m, d, 1e-6, True, A, B, None, None)
eig = np.abs(np.linalg.eigvals(A))
print(f"A 的最大特征值模 {eig.max():.4f}，模 > 1.0001 的有 {np.sum(eig > 1.0001)} 个（不稳定方向）")
j = m.joint("knee_left").dofadr[0]
a = [i for i in range(m.nu) if m.actuator_trnid[i, 0] == m.joint("knee_left").id][0]
print(f"B 的含义：左膝 ctrl +1（= {m.actuator_gear[a, 0]:.0f} N·m），一步后左膝角速度变化 {B[m.nv + j, a]:.2f} rad/s")
