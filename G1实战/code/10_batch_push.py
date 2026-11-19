"""第 10 课（下）：批量随机推力测试，统计摔倒率随推力大小的变化。

对站立的 G1（stand 关键帧 + 模型自带的 position 执行器）在随机方向、随机大小、
随机时刻推一下躯干，跑几百次，看推力多大时它会倒。
同一批实验用三种方式各跑一遍：串行、多进程、mujoco.batch 多线程，比较耗时并核对结果。

用法：
    python 10_batch_push.py                 # 默认 500 次，种子 0
    python 10_batch_push.py --n 1000 --seed 1
    python 10_batch_push.py --skip-serial   # 跳过最慢的串行版
    python 10_batch_push.py --skip-serial --push-chunks 20   # 推 0.2 s
"""
import argparse
import concurrent.futures as cf
import csv
import os
import pathlib
import time

import matplotlib
matplotlib.use("Agg")                      # 只存图，不弹窗
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import batch

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML = ROOT / "models" / "unitree_g1" / "scene.xml"
OUT = ROOT / "output"

# ---------- 实验设置 ----------
F_MAX = 200.0          # 推力大小在 [0, F_MAX] N 里均匀随机
CHUNK = 5              # 每 5 步（10 ms）检查一次要不要推：推力时刻精确到 10 ms
PUSH_CHUNKS = 10       # 每次推 10 个 chunk = 0.1 s（可用 --push-chunks 改）
T_END = 5.0            # 每次实验仿真 5 s：个别“慢慢倒”的实验到 4 s 还没倒完
FALL_Z = 0.5           # 结束时骨盆高度低于 0.5 m 算摔倒（站立时约 0.79 m）


def make_trials(n, seed):
    """用一个带种子的随机数生成器产生全部实验参数。种子相同，参数就完全相同。"""
    rng = np.random.default_rng(seed)
    force = rng.uniform(0.0, F_MAX, n)                # 推力大小 N
    angle = rng.uniform(-np.pi, np.pi, n)             # 水平方向，0 = +x（G1 面朝 +x，即从背后往前推）
    start = rng.integers(50, 150, n)                  # 开始推的 chunk 序号：0.5–1.5 s
    fvec = np.stack([force * np.cos(angle), force * np.sin(angle), np.zeros(n)], axis=1)
    return force, angle, start, fvec


def load_model():
    m = mujoco.MjModel.from_xml_path(str(XML))
    return m, m.key("stand").id, m.body("torso_link").id


# ---------- 方式 1：串行，一个 MjData 依次跑 ----------
def run_serial(m, key, torso, fvec, start, push_chunks=PUSH_CHUNKS):
    d = mujoco.MjData(m)
    nchunk = round(T_END / (CHUNK * m.opt.timestep))
    z = np.empty(len(fvec))
    for i in range(len(fvec)):
        mujoco.mj_resetDataKeyframe(m, d, key)        # qpos、ctrl 都恢复成 stand
        for c in range(nchunk):
            pushing = start[i] <= c < start[i] + push_chunks
            d.xfrc_applied[torso, :3] = fvec[i] if pushing else 0.0
            mujoco.mj_step(m, d, nstep=CHUNK)
        z[i] = d.qpos[2]
    return z


# ---------- 方式 2：多进程，每个进程自己加载模型、跑一部分 ----------
_WORKER = {}


def _init_worker():
    _WORKER["model"] = load_model()


def _work(args):
    fvec, start, push_chunks = args
    m, key, torso = _WORKER["model"]
    return run_serial(m, key, torso, fvec, start, push_chunks)


def run_multiprocess(fvec, start, nworker, push_chunks=PUSH_CHUNKS):
    parts = np.array_split(np.arange(len(fvec)), nworker * 4)   # 切成小块，进程之间负载更均匀
    jobs = [(fvec[p], start[p], push_chunks) for p in parts]      # 参数随任务一起传，子进程不读主进程改过的全局变量
    with cf.ProcessPoolExecutor(nworker, initializer=_init_worker) as pool:
        return np.concatenate(list(pool.map(_work, jobs)))      # map 按提交顺序返回


# ---------- 方式 3：mujoco.batch，nsim 个仿真在 C++ 线程池里一起步进 ----------
def run_batch(m, key, torso, fvec, start, nthread, push_chunks=PUSH_CHUNKS):
    n = len(fvec)
    b = batch.Batch(m, nsim=n, nthread=nthread)       # nthread=0 表示用全部 CPU 核
    b.reset(keyframe=key)                             # 所有仿真都重置到 stand 关键帧
    xfrc = b.bind("xfrc_applied")                     # 形状 (n, nbody, 6)，写它就是给每个仿真施力
    qpos = b.bind("qpos")                             # 形状 (n, nq)
    nchunk = round(T_END / (CHUNK * m.opt.timestep))
    for c in range(nchunk):
        pushing = (start <= c) & (c < start + push_chunks)      # (n,) 布尔数组
        xfrc[:, torso, :3] = fvec * pushing[:, None]
        b.step(nstep=CHUNK)
    return qpos[:, 2].copy()


# ---------- 统计与画图 ----------
def direction_group(angle):
    """按推力方向分三组。力沿 +x 就是“从背后推”，G1 会往前倒。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.where(np.abs(s) > np.abs(c), "侧推", np.where(c > 0, "从背后推", "从正面推"))


def fall_rate(force, fell, edges):
    idx = np.digitize(force, edges) - 1
    cnt = np.array([(idx == k).sum() for k in range(len(edges) - 1)])
    rate = np.array([fell[idx == k].mean() if cnt[k] else np.nan for k in range(len(edges) - 1)])
    return rate, cnt


def f50(centers, rate):
    """线性插值找摔倒率第一次越过 50% 的推力。"""
    for k in range(1, len(rate)):
        if rate[k - 1] < 0.5 <= rate[k]:
            return centers[k - 1] + (0.5 - rate[k - 1]) / (rate[k] - rate[k - 1]) * (centers[k] - centers[k - 1])
    return float("nan")


def plot(force, angle, fell, n, seed, path, push_s):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    edges = np.arange(0, F_MAX + 1, 25)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    rate, cnt = fall_rate(force, fell, edges)
    err = np.sqrt(rate * (1 - rate) / np.maximum(cnt, 1))      # 二项分布标准误
    ax = axes[0]
    ax.errorbar(centers, rate * 100, yerr=err * 100, color="#2a78d6", lw=2, marker="o", ms=6, capsize=3)
    ax.set_title(f"全部方向（{n} 次，种子 {seed}）")
    ax.set_xlabel(f"推力大小 / N（推 {push_s:g} s）")
    ax.set_ylabel("摔倒率 / %")

    ax = axes[1]
    grp = direction_group(angle)
    for name, color, mk in [("从正面推", "#eb6834", "s"), ("从背后推", "#1baf7a", "^"), ("侧推", "#2a78d6", "o")]:
        sel = grp == name
        r, _ = fall_rate(force[sel], fell[sel], edges)
        ax.plot(centers, r * 100, color=color, lw=2, marker=mk, ms=6, label=name)
    ax.set_title("按推力方向分组")
    ax.set_xlabel(f"推力大小 / N（推 {push_s:g} s）")
    ax.legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return edges, centers, rate, cnt, grp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="实验次数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--nthread", type=int, default=0, help="mujoco.batch 线程数，0 = 全部核")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="多进程的进程数")
    p.add_argument("--skip-serial", action="store_true", help="跳过串行版（最慢）")
    p.add_argument("--push-chunks", type=int, default=PUSH_CHUNKS, help="推多少个 chunk（每个 10 ms），默认 10 = 0.1 s")
    args = p.parse_args()

    m, key, torso = load_model()
    force, angle, start, fvec = make_trials(args.n, args.seed)
    nstep = round(T_END / m.opt.timestep)
    print(f"MuJoCo {mujoco.__version__}，CPU 逻辑核数 {os.cpu_count()}")
    print(f"{args.n} 次实验 × {nstep} 步 = {args.n * nstep:,} 步，种子 {args.seed}")
    pc = args.push_chunks
    push_s = pc * CHUNK * m.opt.timestep
    print(f"每次推 {push_s:g} s，仿真 {T_END:g} s")
    print(f"前 3 次实验的参数：力 {force[:3].round(1)} N，方向 {np.degrees(angle[:3]).round(1)}°，"
          f"开始时刻 {start[:3] * CHUNK * m.opt.timestep} s")

    results, times = {}, {}
    if not args.skip_serial:
        t = time.perf_counter()
        results["串行"] = run_serial(m, key, torso, fvec, start, pc)
        times["串行"] = time.perf_counter() - t
    t = time.perf_counter()
    results["多进程"] = run_multiprocess(fvec, start, args.workers, pc)
    times["多进程"] = time.perf_counter() - t
    t = time.perf_counter()
    results["batch"] = run_batch(m, key, torso, fvec, start, args.nthread, pc)
    times["batch"] = time.perf_counter() - t

    print("\n=== 耗时 ===")
    base = times.get("串行")
    for name, sec in times.items():
        speed = f"  加速 {base / sec:4.1f}×" if base else ""
        print(f"{name:<6}{sec:7.2f} s   {args.n * nstep / sec / 1e3:8.0f} 千步/秒{speed}")
    ref = results["batch"]
    for name, z in results.items():
        if name != "batch":
            print(f"{name} 与 batch 的骨盆高度最大差 {np.abs(z - ref).max():.1e} m")

    fell = ref < FALL_Z
    print(f"\n=== 结果 ===\n摔倒 {fell.sum()} / {args.n}（{fell.mean():.1%}）")

    OUT.mkdir(exist_ok=True)
    tag = f"seed{args.seed}" if pc == PUSH_CHUNKS else f"seed{args.seed}_push{push_s:g}s"   # 推的时长不同就另存
    png = OUT / f"10_fall_rate_{tag}.png"
    edges, centers, rate, cnt, grp = plot(force, angle, fell, args.n, args.seed, png, push_s)
    print("推力区间 N     次数  摔倒率")
    for k in range(len(cnt)):
        print(f"[{edges[k]:3.0f}, {edges[k + 1]:3.0f})  {cnt[k]:5d}  {rate[k]:6.0%}")
    print("摔倒率 50% 对应的推力（线性插值）：")
    for name in ["从正面推", "从背后推", "侧推"]:
        sel = grp == name
        r, _ = fall_rate(force[sel], fell[sel], edges)
        print(f"  {name}：{f50(centers, r):5.0f} N（{sel.sum()} 次）")

    csv_path = OUT / f"10_push_trials_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig：Excel 打开中文不乱码
        w = csv.writer(f)
        w.writerow(["id", "force_N", "angle_deg", "direction", "start_s", "pelvis_z_end_m", "fell"])
        for i in range(args.n):
            w.writerow([i, f"{force[i]:.2f}", f"{np.degrees(angle[i]):.1f}", grp[i],
                        f"{start[i] * CHUNK * m.opt.timestep:.2f}", f"{ref[i]:.4f}", int(fell[i])])
    print(f"\n图 {png}\n表 {csv_path}")


if __name__ == "__main__":       # Windows 上多进程必须有这一句，否则子进程会重复执行主程序
    main()
