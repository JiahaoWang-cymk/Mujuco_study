"""第 9 课：对比宇树 12 自由度模型（g1_12dof.xml）和 Menagerie 版（g1.xml）。"""
import ast
import pathlib
import unicodedata

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = {
    "Menagerie g1.xml": ROOT / "models" / "unitree_g1" / "scene.xml",
    "rl_gym g1_12dof.xml": ROOT / "models" / "unitree_rl_gym_g1" / "g1_description" / "scene.xml",
}
INTEGRATOR = {int(v): k for k, v in mujoco.mjtIntegrator.__members__.items()}


def pad(text, width):
    """按显示宽度补空格（一个汉字占两格），让表格对齐。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - w, 1)


def actuator_kind(m, a):
    """按 gaintype / biastype 判断执行器类型。"""
    if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE:
        return f"motor（力 = {m.actuator_gainprm[a, 0]:g} × ctrl）"
    return (f"position（kp = {m.actuator_gainprm[a, 0]:g}，"
            f"kv = {-m.actuator_biasprm[a, 2]:.2f}）")


def summary(name, path):
    m = mujoco.MjModel.from_xml_path(str(path))
    knee = m.joint("left_knee_joint").id
    knee_dof = m.jnt_dofadr[knee]
    a_knee = m.actuator("left_knee_joint").id
    collide = [g for g in range(m.ngeom)
               if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0]
    foot = m.body("left_ankle_roll_link").id
    foot_geoms = [g for g in collide if m.geom_bodyid[g] == foot]
    fg = foot_geoms[0]
    # 力矩上限：可能写在执行器上（forcerange），也可能写在关节上（actuatorfrcrange）
    if m.actuator_forcelimited[a_knee]:
        limit = f"执行器 forcerange ±{m.actuator_forcerange[a_knee, 1]:g}"
    else:
        limit = f"关节 actuatorfrcrange ±{m.jnt_actfrcrange[knee, 1]:g}"
    rows = {
        "nq / nv / nu": f"{m.nq} / {m.nv} / {m.nu}",
        "nbody / njnt": f"{m.nbody} / {m.njnt}",
        "总质量 (kg)": f"{m.body_subtreemass[0]:.2f}",
        "参与碰撞的 geom 数": f"{len(collide)}",
        "左脚碰撞 geom": f"{len(foot_geoms)} 个 {mujoco.mjtGeom(m.geom_type[fg]).name[7:].lower()}，"
                        f"半径 {m.geom_size[fg, 0]:g}，摩擦 {m.geom_friction[fg, 0]:g}",
        "timestep / 积分器": f"{m.opt.timestep:g} / {INTEGRATOR[m.opt.integrator]}",
        "左膝执行器": actuator_kind(m, a_knee),
        "左膝力矩上限 (N·m)": limit,
        "左膝 armature / damping / frictionloss":
            f"{m.dof_armature[knee_dof]:g} / {m.dof_damping[knee_dof]:g} / {m.dof_frictionloss[knee_dof]:g}",
        "传感器数 / 关键帧数": f"{m.nsensor} / {m.nkey}",
    }
    return m, rows


ms, tables = {}, {}
for name, path in MODELS.items():
    ms[name], tables[name] = summary(name, path)

names = list(MODELS)
print(pad("", 40) + pad(names[0], 38) + names[1])
for key in tables[names[0]]:
    print(pad(key, 40) + pad(tables[names[0]][key], 38) + tables[names[1]][key])

# 腿部 12 个关节：名字、范围是否一致
m_men, m_rl = ms[names[0]], ms[names[1]]
legs = [m_rl.joint(j).name for j in range(1, m_rl.njnt)]
same_range = all(np.allclose(m_men.joint(n).range, m_rl.joint(n).range, atol=1e-4) for n in legs)
print(f"\n12dof 的 {len(legs)} 个关节在 Menagerie 里都有：{all(n in [m_men.joint(j).name for j in range(m_men.njnt)] for n in legs)}；"
      f"关节范围一致：{same_range}")
only = [m_men.joint(j).name for j in range(1, m_men.njnt) if m_men.joint(j).name not in legs]
print(f"Menagerie 多出的 {len(only)} 个关节：腰 {sum('waist' in n for n in only)} 个，"
      f"手臂 {sum(('shoulder' in n) or ('elbow' in n) or ('wrist' in n) for n in only)} 个")

# 默认关节角（g1.yaml）对比 Menagerie 的 stand 关键帧
key = m_men.key("stand")
stand_legs = np.array([key.qpos[m_men.jnt_qposadr[m_men.joint(n).id]] for n in legs])
# 从 g1.yaml 里读 default_angles（列表跨两行，这里简单地拼起来再解析，不依赖 PyYAML）
yaml_text = (ROOT / "models" / "unitree_rl_gym_g1" / "g1.yaml").read_text(encoding="utf-8")
block = yaml_text.split("default_angles:", 1)[1]
default_angles = np.array(ast.literal_eval(block[:block.index("]") + 1].strip()))
print("g1.yaml default_angles 左腿：", default_angles[:6])
print("Menagerie stand 关键帧 左腿：", stand_legs[:6].round(3))
