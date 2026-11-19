"""第 1 课：清点 G1 模型。"""
import pathlib

import mujoco

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)

print(f"MuJoCo 版本 {mujoco.__version__}")
print(f"nbody={m.nbody}  njnt={m.njnt}  ngeom={m.ngeom}  nsite={m.nsite}  nmesh={m.nmesh}")
print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  nsensor={m.nsensor}  nsensordata={m.nsensordata}")
print(f"总质量 {m.body_subtreemass[0]:.2f} kg")

# ---------- 1. body 树 ----------
def print_tree(b=0, depth=0):
    print("  " * depth + m.body(b).name)
    for child in range(1, m.nbody):
        if m.body_parentid[child] == b:
            print_tree(child, depth + 1)

print("\n=== body 树 ===")
print_tree()

# ---------- 2. 关节表：每个关节在 qpos / qvel 里的位置 ----------
JOINT_TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
print("\n=== 关节 ===")
print(f"{'id':>2}  {'名字':<28}{'类型':<7}{'qpos':>5}{'qvel':>5}   范围(rad)")
for j in range(m.njnt):
    lo, hi = m.jnt_range[j]
    rng = f"[{lo:+.3f}, {hi:+.3f}]" if m.jnt_limited[j] else "无限制"
    print(f"{j:>2}  {m.joint(j).name:<28}{JOINT_TYPE[m.jnt_type[j]]:<7}"
          f"{m.jnt_qposadr[j]:>5}{m.jnt_dofadr[j]:>5}   {rng}")

# ---------- 3. 执行器表：谁驱动谁，参数是多少 ----------
print("\n=== 执行器 ===")
for a in range(m.nu):
    j = m.actuator_trnid[a, 0]               # 这个执行器驱动的关节 id
    kp = m.actuator_gainprm[a, 0]
    bias = m.actuator_biasprm[a, :3]
    flo, fhi = m.actuator_forcerange[a]
    print(f"{a:>2}  {m.actuator(a).name:<28} 关节 {j:>2}  kp={kp:5.0f}  "
          f"biasprm=[{bias[0]:.1f}, {bias[1]:.1f}, {bias[2]:.2f}]  力矩 ±{fhi:.0f} N·m")

# ---------- 4. 加载关键帧，看看各部件在哪 ----------
key = m.key("stand")
mujoco.mj_resetDataKeyframe(m, d, key.id)
mujoco.mj_forward(m, d)                      # 只算运动学等，不推进时间
print("\n=== stand 关键帧 ===")
print("骨盆位置    ", d.body("pelvis").xpos.round(3))
print("躯干 IMU 位置", d.site("imu_in_torso").xpos.round(3))
print("左脚 site   ", d.site("left_foot").xpos.round(3))
print("整机质心    ", d.subtree_com[0].round(3))
