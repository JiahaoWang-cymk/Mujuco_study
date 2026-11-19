"""附录：自动识别 NVIDIA GPU，用 MuJoCo Warp 同时仿真成百上千台 G1，和 CPU 版 MuJoCo 比吞吐量。

有 NVIDIA 显卡（装好驱动）时，Warp 会自动识别到 cuda:0，在 GPU 上跑；没有就退回 Warp 的 CPU 模式，
用很小的规模把同一套代码跑通（这时的数字只说明“代码路径是通的”，不代表 GPU 的速度）。

mujoco_warp 和 warp-lang 是装 mujoco 4.x 时一起装上的依赖，不用另外安装；GPU 只需要显卡驱动。
第一次在某个设备上运行要编译 Warp 内核（CPU 上约半分钟到一分钟），结果会缓存，之后快很多。

用法：
    python 11_gpu_rollout.py                    # 自动选设备，自动选规模
    python 11_gpu_rollout.py --nworld 8192      # 指定同时仿真多少台
    python 11_gpu_rollout.py --device cpu       # 强制用 Warp 的 CPU 模式（调试用）
    python 11_gpu_rollout.py --model full       # 用第 1–8 课的 g1.xml（默认用 GPU 友好的 g1_mjx.xml）
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import time

import mujoco
import numpy as np
from mujoco import rollout

ROOT = pathlib.Path(__file__).resolve().parent.parent
G1 = ROOT / "models" / "unitree_g1"
OUT = ROOT / "output"
MODELS = {"mjx": (G1 / "scene_mjx.xml", "home"), "full": (G1 / "scene.xml", "stand")}


# ---------------------------------------------------------------- 1. 识别设备
def detect():
    """返回 (warp 模块或 None, CUDA 设备列表)。只读信息，不改任何东西。"""
    print("===== 1. 识别设备 =====")
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            print(f"  nvidia-smi：{line}")
    else:
        print("  没找到 nvidia-smi（通常说明没装 NVIDIA 驱动，或者没有 NVIDIA 显卡）")
    try:
        import warp as wp
    except ImportError as e:
        print(f"  导入 warp 失败：{e}\n  mujoco_warp 依赖 warp-lang，按第 0 课补装后再试")
        return None, []
    wp.config.log_level = wp.LOG_WARNING           # 不打印每个内核的编译日志
    wp.config.quiet = True
    wp.init()
    cuda = wp.get_cuda_devices()
    print(f"  warp {wp.__version__}：CUDA 设备 {len(cuda)} 个")
    for dev in cuda:
        arch = getattr(dev, "arch", 0)
        print(f"    {dev.alias}：{dev.name}，显存 {getattr(dev, 'total_memory', 0) / 2**30:.1f} GB，"
              f"计算能力 {arch // 10}.{arch % 10}")
    try:
        import jax
        print(f"  （参考）jax {jax.__version__} 的设备：{[str(d) for d in jax.devices()]}")
    except Exception:
        pass
    return wp, cuda


# ---------------------------------------------------------------- 2. CPU 版 MuJoCo 基准
def bench_cpu(m, d, nworld, nstep):
    """mujoco.rollout 多线程推演 nworld 条轨迹，保持关键帧的 ctrl。"""
    nthread = os.cpu_count() or 1
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    x0 = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, x0, spec)
    init = np.tile(x0, (nworld, 1))
    ctrl = np.tile(d.ctrl, (nworld, nstep, 1))
    datas = [mujoco.MjData(m) for _ in range(nthread)]
    rollout.rollout(m, datas, init[:nthread], ctrl[:nthread, :10])     # 预热，不计时
    t0 = time.perf_counter()
    state, _ = rollout.rollout(m, datas, init, ctrl)
    wall = time.perf_counter() - t0
    z = state[:, -1, 1 + 2]                     # FULLPHYSICS 排布 [time, qpos...]，骨盆高度 qpos[2]
    return nworld * nstep / wall, wall, nthread, z


# ---------------------------------------------------------------- 3. MuJoCo Warp
def bench_warp(wp, m, d, device, nworld, nstep, seed=0):
    import mujoco_warp as mjw
    rng = np.random.default_rng(seed)
    with wp.ScopedDevice(device):
        t0 = time.perf_counter()
        mw = mjw.put_model(m)
        # “求解器迭代次数到上限”不算错误：g1_mjx.xml 故意只给 5 次迭代（见 11.3 节），关掉这条逐步打印的提示
        iters = int(mjw.OverflowType.ITERATIONS) | int(mjw.OverflowType.LS_ITERATIONS)
        mw.opt.warn_overflow = int(mw.opt.warn_overflow) & ~iters
        # nconmax / njmax：每个世界预留的接触数、约束行数。站着的 G1 只有脚底几个接触，留足余量
        dw = mjw.put_data(m, d, nworld=nworld, nconmax=64, njmax=256)
        # 每个世界的初速度加一点随机扰动，让几千台 G1 不完全一样
        qvel = np.tile(d.qvel, (nworld, 1)) + rng.normal(0, 0.05, (nworld, m.nv))
        dw.qvel.assign(wp.array(qvel.astype(np.float32), dtype=float, device=device).reshape(dw.qvel.shape))
        setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        mjw.step(mw, dw)                        # 第一步要编译内核（有缓存时很快）
        wp.synchronize()
        compile_s = time.perf_counter() - t0

        if wp.get_device(device).is_cuda:       # GPU：把一步录成 CUDA graph，之后反复重放
            with wp.ScopedCapture() as cap:
                mjw.step(mw, dw)
            step = lambda: wp.capture_launch(cap.graph)
        else:
            step = lambda: mjw.step(mw, dw)
        step()
        wp.synchronize()

        t0 = time.perf_counter()
        for _ in range(nstep):
            step()
        wp.synchronize()
        wall = time.perf_counter() - t0
        z = dw.qpos.numpy()[:, 2]
        flags = dw.overflow.numpy()
        overflow = int(np.count_nonzero(flags & ~iters))          # 真正的缓冲区溢出
        hit_iters = int(np.count_nonzero(flags & iters))          # 只是迭代次数到上限
    return nworld * nstep / wall, wall, setup, compile_s, z, overflow, hit_iters


def main():
    ap = argparse.ArgumentParser(description="自动识别 GPU，用 MuJoCo Warp 批量仿真 G1")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--nworld", type=int, help="同时仿真多少台 G1（默认：GPU 4096，CPU 16）")
    ap.add_argument("--seconds", type=float, default=1.0, help="每台仿真多少秒（默认 1 s）")
    ap.add_argument("--model", choices=list(MODELS), default="mjx",
                    help="mjx = g1_mjx.xml（碰撞简化，GPU 训练用）；full = 第 1–8 课的 g1.xml")
    ap.add_argument("--skip-cpu", action="store_true", help="不跑 CPU 版 MuJoCo 的对照")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    wp, cuda = detect()
    if args.device == "cuda" and not cuda:
        print("\n指定了 --device cuda，但 Warp 没识别到 CUDA 设备，改用 CPU 模式。")
    use_gpu = bool(cuda) and args.device != "cpu"
    device = "cuda:0" if use_gpu else "cpu"
    nworld = args.nworld or (4096 if use_gpu else 16)

    path, key = MODELS[args.model]
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    mujoco.mj_forward(m, d)
    nstep = int(round(args.seconds / m.opt.timestep))
    print(f"\n模型：{path.name}（关键帧 {key}，timestep {m.opt.timestep} s）。"
          f"{nworld} 台 G1 × {nstep} 步（{args.seconds:g} s 仿真时间），关节保持关键帧的目标角。")

    rows, cpu_z = [], None
    if not args.skip_cpu:
        print("\n===== 2. CPU 版 MuJoCo（mujoco.rollout 多线程）=====")
        sps, wall, nthread, z = bench_cpu(m, d, nworld, nstep)
        cpu_z = z
        print(f"  {nthread} 线程：{wall:.2f} s，约 {sps:,.0f} 步/秒；结束时骨盆高度平均 {z.mean():.3f} m")
        rows.append(("CPU mujoco.rollout", f"{nthread} 线程", nworld, sps))

    if wp is None:
        print("\n没有 warp，跳过 MuJoCo Warp。")
    else:
        where = f"GPU（{cuda[0].name}）" if use_gpu else "CPU（Warp 的 CPU 模式，只验证代码路径）"
        print(f"\n===== 3. MuJoCo Warp，设备 {device}：{where} =====")
        if not use_gpu:
            print("  未检测到可用的 NVIDIA GPU，改在 CPU 上用小规模跑通同一套代码。")
            print("  第一次运行要编译内核，可能要等半分钟到一分钟。")
        try:
            result = bench_warp(wp, m, d, device, nworld, nstep)
        except Exception as e:                  # 例如显卡驱动太旧、显存不够
            if not use_gpu:
                raise
            print(f"  在 GPU 上运行失败：{type(e).__name__}: {e}")
            print("  常见原因：显卡驱动版本太旧（升级 NVIDIA 驱动），或 --nworld 太大、显存不够（调小再试）。")
            print("  先退回 Warp 的 CPU 模式，用 16 台跑通代码路径。")
            device, nworld, use_gpu = "cpu", 16, False
            if not args.skip_cpu:
                cpu_sps, _, _, cpu_z = bench_cpu(m, d, nworld, nstep)   # 对照也换成同样的 16 台
                rows[0] = ("CPU mujoco.rollout", rows[0][1], nworld, cpu_sps)
            result = bench_warp(wp, m, d, device, nworld, nstep)
        sps, wall, setup, compile_s, z, overflow, hit_iters = result
        print(f"  拷到设备 {setup:.2f} s，第一步（含编译）{compile_s:.1f} s")
        print(f"  计时部分：{wall:.2f} s，约 {sps:,.0f} 步/秒")
        print(f"  结束时骨盆高度：平均 {z.mean():.3f} m，最低 {z.min():.3f} m"
              f"（站立约 {d.qpos[2]:.3f} m；低于 0.5 m 的有 {int(np.sum(z < 0.5))} 台）")
        if overflow:
            print(f"  注意：{overflow} 个世界的接触/约束缓冲区溢出了，结果不可信。把 nconmax / njmax 调大再试。")
        if hit_iters:
            print(f"  {hit_iters} 个世界的求解器迭代达到上限（{m.opt.iterations} 次）：g1_mjx.xml 故意这样设，用精度换速度，不是错误。")
        if cpu_z is not None:
            print(f"  核对：CPU 版结束时骨盆平均高度 {cpu_z.mean():.3f} m，两边应当接近"
                  f"（Warp 用 float32，初速度也加了随机扰动，不会完全相同）")
        rows.append((f"MuJoCo Warp {device}", "GPU" if use_gpu else "Warp CPU 模式", nworld, sps))

    if len(rows) == 2:
        ratio = rows[1][3] / rows[0][3]
        print(f"\n===== 4. 对比 =====\n  MuJoCo Warp / CPU 版 = {ratio:.2f} 倍")
        if not use_gpu:
            print("  （Warp 的 CPU 模式本来就比 CPU 版 MuJoCo 慢，这个比值没有意义；"
                  "在有 NVIDIA GPU 的机器上再运行一次，看 GPU 的数字。）")
    path_csv = OUT / "11_gpu_rollout.csv"
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("method,device,nworld,steps_per_second\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.0f}\n")
    print(f"\n已保存 {path_csv}")


if __name__ == "__main__":
    main()
