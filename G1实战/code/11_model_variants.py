"""附录：对比 G1 的三个模型版本，并测一测本机 CPU 仿真的吞吐量。

g1.xml       —— 本课程第 1–8 课用的版本（CPU 仿真）
g1_mjx.xml   —— 为 MJX / MuJoCo Playground 简化过碰撞的版本（scene_mjx.xml 引用它）
g1_mjlab.xml —— 为 mjlab 准备的版本（不含执行器和求解器选项，由 mjlab 用代码补上）

用法：
    python 11_model_variants.py            # 默认：对比 + 单线程 + 多线程测速
    python 11_model_variants.py --quick    # 只做对比和单线程测速
"""
import argparse
import os
import pathlib
import time

import mujoco
import numpy as np
from mujoco import rollout

ROOT = pathlib.Path(__file__).resolve().parent.parent
G1 = ROOT / "models" / "unitree_g1"
OUT = ROOT / "output"

VARIANTS = [
    ("g1.xml（scene.xml）", G1 / "scene.xml", "stand"),
    ("g1_mjx.xml（scene_mjx.xml）", G1 / "scene_mjx.xml", "home"),
    ("g1_mjlab.xml（单独加载，无地面）", G1 / "g1_mjlab.xml", None),
]


def n_colliding_geoms(m):
    """能参与碰撞的 geom 数：contype/conaffinity 不全为 0，或者出现在 <pair> 里。"""
    by_mask = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    in_pair = np.zeros(m.ngeom, bool)
    in_pair[m.pair_geom1] = True
    in_pair[m.pair_geom2] = True
    return int(np.sum(by_mask | in_pair))


def describe(name, path, keyname):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    ncon = "-"
    if keyname is not None:
        mujoco.mj_resetDataKeyframe(m, d, m.key(keyname).id)
        mujoco.mj_forward(m, d)
        ncon = d.ncon
    integ = mujoco.mjtIntegrator(m.opt.integrator).name.replace("mjINT_", "")
    print(f"\n--- {name} ---")
    print(f"  nbody={m.nbody}  nq={m.nq}  nv={m.nv}  nu={m.nu}  nsensor={m.nsensor}")
    print(f"  ngeom={m.ngeom}  可碰撞 geom={n_colliding_geoms(m)}  "
          f"显式 pair={m.npair}  exclude={m.nexclude}")
    print(f"  timestep={m.opt.timestep}  iterations={m.opt.iterations}  "
          f"ls_iterations={m.opt.ls_iterations}  integrator={integ}")
    print(f"  关键帧={[m.key(i).name for i in range(m.nkey)]}  站立时接触点数={ncon}")
    return m


def bench_single(path, keyname, sim_seconds=10.0, kp=None):
    """单线程：保持关键帧的 ctrl，跑 sim_seconds 秒仿真时间，计时。

    kp 不为 None 时，先把所有 position 执行器的刚度改成 kp（只改内存里的模型）。
    position 执行器：力 = kp*ctrl - kp*qpos - kv*qvel，所以要同时改 gainprm[0] 和 biasprm[1]。
    """
    m = mujoco.MjModel.from_xml_path(str(path))
    if kp is not None:
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyname).id)
    nstep = int(round(sim_seconds / m.opt.timestep))
    t0 = time.perf_counter()
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    wall = time.perf_counter() - t0
    return nstep / wall, sim_seconds / wall, d.qpos[2]


def bench_rollout(path, keyname, nbatch, nthread, sim_seconds=5.0):
    """多线程：用 mujoco.rollout 同时推演 nbatch 条轨迹（每个线程一个 MjData）。"""
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyname).id)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    x0 = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, x0, spec)
    nstep = int(round(sim_seconds / m.opt.timestep))
    init = np.tile(x0, (nbatch, 1))
    ctrl = np.tile(d.ctrl, (nbatch, nstep, 1))     # 每一步都保持关键帧的目标角
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    rollout.rollout(m, datas, init[:nthread], ctrl[:nthread, :10])   # 预热一次，不计时
    t0 = time.perf_counter()
    state, _ = rollout.rollout(m, datas, init, ctrl)
    wall = time.perf_counter() - t0
    total_steps = nbatch * nstep
    # FULLPHYSICS 状态的排布是 [time, qpos, qvel, ...]，所以骨盆高度 qpos[2] 在第 3 列
    z_end = state[:, -1, 1 + 2]
    return total_steps / wall, state.shape, float(np.mean(z_end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过多线程测速")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print(f"MuJoCo 版本 {mujoco.__version__}，本机逻辑 CPU 数 {os.cpu_count()}")
    print("\n===== 1. 三个模型版本对比 =====")
    for name, path, key in VARIANTS:
        describe(name, path, key)

    print("\n===== 2. 单线程 CPU 吞吐量（站立 10 s 仿真时间）=====")
    rows = []
    for name, path, key in VARIANTS[:2]:
        sps, rtf, z = bench_single(path, key)
        flag = "（已倒地）" if z < 0.5 else "（仍站着）"
        print(f"  {name:<28} {sps:>10,.0f} 步/秒  实时倍数 {rtf:6.1f}×  "
              f"结束时骨盆高度 {z:.3f} m{flag}")
        rows.append((name, "1 线程", sps))

    # 对照实验：g1_mjx.xml 倒下，是不是因为增益低？把 kp 全改成 g1.xml 的 500 再试
    _, _, z = bench_single(VARIANTS[1][1], "home", kp=500.0)
    flag = "（已倒地）" if z < 0.5 else "（仍站着）"
    print(f"  对照：g1_mjx.xml 的 29 个执行器 kp 全改成 500 → 结束时骨盆高度 {z:.3f} m{flag}")

    if not args.quick:
        nthread = os.cpu_count() or 1
        nbatch = 8 * nthread
        print(f"\n===== 3. 多线程 mujoco.rollout（{nbatch} 条轨迹 × 5 s，{nthread} 线程）=====")
        for name, path, key in VARIANTS[:2]:
            sps, shape, z = bench_rollout(path, key, nbatch, nthread)
            print(f"  {name:<28} {sps:>10,.0f} 步/秒（合计）  state 形状 {shape}  "
                  f"平均末端骨盆高度 {z:.3f} m")
            rows.append((name, f"{nthread} 线程", sps))

    # ---------- 4. g1_mjlab.xml 的 4 个 exclude 是干什么的 ----------
    print("\n===== 4. 删掉 g1_mjlab.xml 的 exclude，在零位姿态下会有哪些接触 =====")
    m0 = mujoco.MjModel.from_xml_path(str(G1 / "g1_mjlab.xml"))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)                     # 原样加载，qpos 全为零
    print(f"  保留 exclude 时接触点数 {d0.ncon}")
    spec = mujoco.MjSpec.from_file(str(G1 / "g1_mjlab.xml"))
    for e in list(spec.excludes):
        print(f"  删除 exclude：{e.bodyname1} — {e.bodyname2}")
        spec.delete(e)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                       # qpos 全为零（零位姿态）
    print(f"  接触点数 {d.ncon}")
    for c in d.contact[:d.ncon]:
        b1 = m.body(m.geom_bodyid[c.geom1]).name
        b2 = m.body(m.geom_bodyid[c.geom2]).name
        print(f"    {b1:<22} — {b2:<24} 穿透 {-c.dist * 1000:5.1f} mm")

    csv = OUT / "11_cpu_throughput.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("model,threads,steps_per_second\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.0f}\n")
    print(f"\n已保存 {csv}")


if __name__ == "__main__":
    main()
