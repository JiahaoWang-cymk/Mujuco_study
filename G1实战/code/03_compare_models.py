"""第 3 课练习参考答案：对比宇树 unitree_rl_gym 的 12 自由度 G1 与 Menagerie 的 29 自由度 G1。

rl_gym 的 g1_12dof.xml 看起来是从宇树官方 URDF 转出来的 MJCF（练习 3 找的就是它身上 URDF 转换的痕迹）。
"""
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEN = ROOT / "models" / "unitree_g1" / "g1.xml"
GYM = ROOT / "models" / "unitree_rl_gym_g1" / "g1_description" / "g1_12dof.xml"

men = mujoco.MjModel.from_xml_path(str(MEN))
gym = mujoco.MjModel.from_xml_path(str(GYM))


def names(m, objtype, n):
    return [mujoco.mj_id2name(m, objtype, i) for i in range(n)]


# ---------- 1. 总量 ----------
print(f"{'':<16}{'Menagerie':>12}{'rl_gym 12dof':>14}")
for label, f in [("nbody", lambda m: m.nbody), ("njnt", lambda m: m.njnt),
                 ("nq", lambda m: m.nq), ("nv", lambda m: m.nv), ("nu", lambda m: m.nu),
                 ("ngeom", lambda m: m.ngeom), ("总质量 kg", lambda m: round(m.body_subtreemass[0], 3))]:
    print(f"{label:<16}{f(men):>12}{f(gym):>14}")

# ---------- 2. 关节：谁有谁没有 ----------
jm = set(names(men, mujoco.mjtObj.mjOBJ_JOINT, men.njnt))
jg = set(names(gym, mujoco.mjtObj.mjOBJ_JOINT, gym.njnt))
print(f"\n两边都有的关节 {len(jm & jg)} 个；只在 Menagerie 里的 {len(jm - jg)} 个：")
print("  " + ", ".join(sorted(n.replace("_joint", "") for n in jm - jg)))
print("只在 rl_gym 里的：", sorted(jg - jm) or "无")

# ---------- 3. 共有关节的参数 ----------
print(f"\n{'关节':<26}{'range 相同':>10}{'armature':>16}{'frictionloss':>16}{'damping':>14}{'力矩上限':>14}")
for n in sorted(jm & jg):
    if n == "floating_base_joint":
        continue
    a, b = men.joint(n), gym.joint(n)
    same = np.allclose(a.range, b.range, atol=1e-4)
    lim_m = men.actuator_forcerange[men.actuator(n).id, 1]        # Menagerie：写在执行器上
    lim_g = gym.jnt_actfrcrange[b.id, 1]          # rl_gym：写在关节上
    print(f"{n.replace('_joint', ''):<26}{str(same):>10}"
          f"{a.armature[0]:>8.3f}/{b.armature[0]:<7.3f}{a.frictionloss[0]:>8.2f}/{b.frictionloss[0]:<7.2f}"
          f"{a.damping[0]:>6.3f}/{b.damping[0]:<7.3f}{lim_m:>7.0f}/{lim_g:<6.0f}")

# ---------- 4. 质量 ----------
bm = set(names(men, mujoco.mjtObj.mjOBJ_BODY, men.nbody))
bg = set(names(gym, mujoco.mjtObj.mjOBJ_BODY, gym.nbody))
diff = [n for n in sorted(bm & bg) if n != "world"
        and not np.isclose(men.body(n).mass[0], gym.body(n).mass[0], atol=1e-4)]
print(f"\n共有 body {len(bm & bg) - 1} 个，质量不同的：{diff}")
for n in diff:
    print(f"  {n}: Menagerie {men.body(n).mass[0]:.4f} kg，rl_gym {gym.body(n).mass[0]:.4f} kg")

# rl_gym 把腰、躯干、手臂全焊在骨盆上，合成了一个大 body
upper = men.body("pelvis").mass[0] + men.body_subtreemass[men.body("waist_yaw_link").id]
print(f"Menagerie 骨盆 + 腰以上整棵子树 = {upper:.3f} kg；rl_gym 的 pelvis = {gym.body('pelvis').mass[0]:.3f} kg")
legs_m = sum(men.body_subtreemass[men.body(f"{s}_hip_pitch_link").id] for s in ("left", "right"))
legs_g = sum(gym.body_subtreemass[gym.body(f"{s}_hip_pitch_link").id] for s in ("left", "right"))
print(f"两条腿：Menagerie {legs_m:.3f} kg，rl_gym {legs_g:.3f} kg")

# ---------- 5. 执行器 ----------
GAIN = {0: "fixed", 1: "affine"}
for label, m in [("Menagerie", men), ("rl_gym", gym)]:
    print(f"{label} 执行器：gaintype={GAIN.get(m.actuator_gaintype[0])}，"
          f"biastype={['none', 'affine', 'muscle', 'user'][m.actuator_biastype[0]]}，"
          f"增益 gainprm[0]={m.actuator_gainprm[0, 0]:.0f}（position 执行器里就是 kp）")
