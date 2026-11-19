"""第 9 课：运行宇树预训练行走策略（unitree_rl_gym 的 G1 12 自由度版本）。

和官方 deploy_mujoco.py 的管线一模一样：
    仿真 500 Hz（dt = 0.002 s），每一步用 PD 公式算力矩写进 ctrl；
    每 10 步（50 Hz）组一次 47 维观测，喂给策略，得到 12 维动作，换算成新的目标关节角。

在它的基础上加了：改速度指令、在某时刻推一下、改地面摩擦、摔倒判定、CSV 和曲线图。

策略推理有三种后端（--policy）：
    torch  ：和官方一样，用 torch.jit.load 读 motion.pt（需要装 PyTorch）
    numpy  ：不装 PyTorch，用 numpy 照 motion.pt 里的网络结构（LSTM + 两层全连接）手写前向计算
    auto   ：有 PyTorch 用 torch，否则用 numpy（默认）
加 --dummy-policy 则不用策略，动作恒为 0，用来单独检验观测、分频、PD、推力、摔倒判定这条管线。

用法示例：
    python code/09_walk.py --dummy-policy              # 零动作，检查管线
    python code/09_walk.py                             # 跑真策略，默认指令 vx=0.5 m/s
    python code/09_walk.py --vx 0.8 --wz 0.3           # 改速度指令
    python code/09_walk.py --push-time 4 --push-fy 150 # 第 4 秒从侧面推 0.1 s
    python code/09_walk.py --friction 0.3              # 地面变滑
    python code/09_walk.py --view                      # 打开查看器实时看
"""
import argparse
import ast
import csv
import pathlib
import pickle
import time
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RL_DIR = ROOT / "models" / "unitree_rl_gym_g1"
XML = RL_DIR / "g1_description" / "scene.xml"
YAML = RL_DIR / "g1.yaml"
POLICY = RL_DIR / "motion.pt"
OUT = ROOT / "output"


# ---------------------------------------------------------------- 读配置
def load_config(path):
    """读 g1.yaml。有 PyYAML 就用它，没有就用一个够这个文件用的小解析器。"""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        pass
    cfg, buf = {}, ""
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        buf = buf + " " + line if buf else line
        if buf.count("[") > buf.count("]"):      # 列表跨行，接着读
            continue
        key, val = buf.split(":", 1)
        val = val.strip()
        try:
            cfg[key.strip()] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            cfg[key.strip()] = val.strip('"')
        buf = ""
    return cfg


# ---------------------------------------------------------------- 三种策略
class TorchPolicy:
    """官方做法：TorchScript 模型，内部自带 LSTM 记忆。"""
    def __init__(self, path):
        import torch
        self.torch = torch
        self.net = torch.jit.load(str(path))

    def __call__(self, obs):
        x = self.torch.from_numpy(obs).unsqueeze(0)
        return self.net(x).detach().numpy().squeeze()


class NumpyPolicy:
    """不依赖 PyTorch：从 motion.pt（本质是 zip 包）里取出权重，用 numpy 做同样的计算。

    网络结构（见 motion.pt 里的 code/__torch__/legged_gym/utils/helpers.py）：
        47 维观测 → LSTM(隐状态 64) → Linear(64→32) → ELU → Linear(32→12) → 12 维动作
    LSTM 的隐状态 h、c 在每次调用后保留，所以策略“记得”之前几步发生了什么。
    """

    def __init__(self, path):
        z = zipfile.ZipFile(path)
        prefix = z.namelist()[0].split("/")[0]

        class _Stub:                              # 代替 torch 里的各种类，只收集属性
            def __setstate__(self, state):
                self.__dict__.update(state)

        def rebuild_tensor(storage_key, offset, size, stride, *_):
            # 只处理按行连续存放的张量（motion.pt 里的 10 个张量都是）
            contiguous = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
            assert tuple(stride) == contiguous, f"张量不是连续存放的：{size} {stride}"
            raw = np.frombuffer(z.read(f"{prefix}/data/{storage_key}"), dtype="<f4")
            return raw[offset:offset + int(np.prod(size))].reshape(size).copy()

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    return rebuild_tensor
                if name in ("restore_type_tag",):
                    return lambda obj, _type: obj
                if name == "build_tensorlist":
                    return lambda lst: lst
                if module == "collections" and name == "OrderedDict":
                    import collections
                    return collections.OrderedDict
                return type(name, (_Stub,), {})   # 其余类一律用空壳

            def persistent_load(self, pid):       # ('storage', 类型, 编号, 设备, 元素数)
                return pid[2]

        root = _Unpickler(z.open(f"{prefix}/data.pkl")).load()
        lstm, actor = root.memory, root.actor
        self.W_ih, self.W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0     # (256, 47), (256, 64)
        self.b = lstm.bias_ih_l0 + lstm.bias_hh_l0                      # (256,)
        l0, l2 = getattr(actor, "0"), getattr(actor, "2")
        self.W1, self.b1 = l0.weight, l0.bias                           # (32, 64)
        self.W2, self.b2 = l2.weight, l2.bias                           # (12, 32)
        self.h = root.hidden_state.reshape(-1).copy()                   # 文件里存的初值（全 0）
        self.c = root.cell_state.reshape(-1).copy()
        self.n_in = self.W_ih.shape[1]

    def __call__(self, obs):
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        gates = self.W_ih @ obs + self.W_hh @ self.h + self.b
        i, f, g, o = np.split(gates, 4)           # PyTorch 的门顺序：输入、遗忘、候选、输出
        self.c = sig(f) * self.c + sig(i) * np.tanh(g)
        self.h = sig(o) * np.tanh(self.c)
        y = self.W1 @ self.h + self.b1
        y = np.where(y > 0, y, np.expm1(y))       # ELU(alpha=1)
        return (self.W2 @ y + self.b2).astype(np.float32)


class DummyPolicy:
    """零动作：目标关节角永远等于默认角，只剩 PD 在“抱住”默认姿势。"""
    def __init__(self, n):
        self.n = n

    def __call__(self, obs):
        return np.zeros(self.n, dtype=np.float32)


def make_policy(kind, n_act):
    if kind == "dummy":
        return DummyPolicy(n_act), "dummy（零动作）"
    if kind in ("auto", "torch"):
        try:
            return TorchPolicy(POLICY), "torch（torch.jit.load）"
        except ImportError:
            if kind == "torch":
                raise SystemExit("没有安装 PyTorch，请改用 --policy numpy")
    return NumpyPolicy(POLICY), "numpy（手写 LSTM 前向）"


# ---------------------------------------------------------------- 与官方脚本相同的两个函数
def get_gravity_orientation(q):
    """重力方向在骨盆坐标系里的投影（单位向量）。q 为 w x y z。"""
    qw, qx, qy, qz = q
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser(description="第 9 课：G1 行走策略")
    ap.add_argument("--policy", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--dummy-policy", action="store_true", help="动作恒为 0，只检验管线")
    ap.add_argument("--vx", type=float, help="前进速度指令 m/s（默认取 g1.yaml 的 cmd_init）")
    ap.add_argument("--vy", type=float, help="侧移速度指令 m/s")
    ap.add_argument("--wz", type=float, help="转向角速度指令 rad/s")
    ap.add_argument("--duration", type=float,
                    help="仿真时长 s（默认 10；加 --view 时默认取 g1.yaml 的 simulation_duration）")
    ap.add_argument("--push-time", type=float, help="开始推的时刻 s")
    ap.add_argument("--push-fx", type=float, default=0.0, help="推力 x 分量 N（世界坐标，向前为正）")
    ap.add_argument("--push-fy", type=float, default=0.0, help="推力 y 分量 N（向左为正）")
    ap.add_argument("--push-dur", type=float, default=0.1, help="推力持续时间 s")
    ap.add_argument("--friction", type=float, help="地面和脚底的滑动摩擦系数（默认 1.0）")
    ap.add_argument("--decimation", type=int, help="改控制分频（默认取 g1.yaml 的 control_decimation=10）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，批量实验时区分")
    ap.add_argument("--view", action="store_true", help="打开查看器实时观看")
    args = ap.parse_args()

    # ---------- 1. 读配置（和官方脚本一样的字段）----------
    cfg = load_config(YAML)
    dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    if args.decimation:
        decim = args.decimation
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    ang_vel_scale = cfg["ang_vel_scale"]
    dof_pos_scale = cfg["dof_pos_scale"]
    dof_vel_scale = cfg["dof_vel_scale"]
    action_scale = cfg["action_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    n_act = cfg["num_actions"]
    n_obs = cfg["num_obs"]
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)
    for i, v in enumerate([args.vx, args.vy, args.wz]):
        if v is not None:
            cmd[i] = v

    # 观测向量的布局，必须和训练时一致
    layout = [("角速度 omega", 3), ("重力投影 gravity", 3), ("速度指令 cmd", 3),
              ("关节位置偏差 q-q0", n_act), ("关节速度 dq", n_act),
              ("上一步动作 action", n_act), ("步态相位 sin,cos", 2)]
    total = sum(n for _, n in layout)
    print(f"仿真步长 {dt} s（{1/dt:.0f} Hz），策略每 {decim} 步算一次（{1/(dt*decim):g} Hz）")
    print("观测向量布局：")
    start = 0
    for name, n in layout:
        print(f"  obs[{start:2d}:{start+n:2d}]  {n:2d} 维  {name}")
        start += n
    print(f"  合计 {total} 维，g1.yaml 里 num_obs = {n_obs}  →  {'一致' if total == n_obs else '不一致！'}")
    assert total == n_obs

    # ---------- 2. 模型 ----------
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    m.opt.timestep = dt
    assert m.nu == n_act and m.nq == 7 + n_act, "模型和配置的关节数对不上"
    pelvis = m.body("pelvis").id
    floor = m.geom("floor").id
    foot_bodies = {m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id}
    if args.friction is not None:
        # 接触摩擦取两个 geom 的较大值，所以地面和脚（这里干脆所有 geom）都要改
        m.geom_friction[:, 0] = args.friction

    policy, policy_name = make_policy("dummy" if args.dummy_policy else args.policy, n_act)
    print(f"策略：{policy_name}")
    print(f"速度指令 cmd = [vx {cmd[0]:+.2f} m/s, vy {cmd[1]:+.2f} m/s, wz {cmd[2]:+.2f} rad/s]")
    if args.push_time is not None:
        print(f"推力：t = {args.push_time} s 起 {args.push_dur} s，"
              f"F = ({args.push_fx:+.0f}, {args.push_fy:+.0f}) N，作用在骨盆质心")

    # ---------- 3. 状态变量（和官方脚本同名）----------
    action = np.zeros(n_act, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(n_obs, dtype=np.float32)
    counter = 0
    if args.duration is None:
        args.duration = cfg["simulation_duration"] if args.view else 10.0
    n_steps = int(round(args.duration / dt))

    # 记录
    log = []
    n_policy_calls = 0
    n_push_steps = 0
    n_sat = 0                 # PD 力矩超过关节 actuatorfrcrange 的次数（被截断）
    tau_abs_max = np.zeros(n_act)
    fall_time, fall_reason = None, ""
    first_obs = None
    contact_mu = None
    frc_hi = m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1]

    viewer = None
    if args.view:
        viewer = mujoco.viewer.launch_passive(m, d)
        with viewer.lock():                       # 镜头跟着骨盆走
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis
            viewer.cam.distance = 3.0

    t_wall = time.perf_counter()
    for step in range(n_steps):
        step_start = time.perf_counter()

        # (a) 每个仿真步：PD 算力矩，写进 motor 执行器的 ctrl
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        tau_abs_max = np.maximum(tau_abs_max, np.abs(tau))
        n_sat += int(np.sum(np.abs(tau) > frc_hi))

        # (b) 推力：写进 xfrc_applied（世界坐标下的力，作用在 body 质心）
        #     没设推力时不碰它，这样在查看器里用 Ctrl+右键拖动的推力不会被清掉
        if args.push_time is not None:
            d.xfrc_applied[pelvis] = 0
            if args.push_time <= d.time < args.push_time + args.push_dur:
                d.xfrc_applied[pelvis, 0] = args.push_fx
                d.xfrc_applied[pelvis, 1] = args.push_fy
                n_push_steps += 1

        mujoco.mj_step(m, d)
        counter += 1

        # (c) 每 decim 步：组观测、跑策略、更新目标角
        if counter % decim == 0:
            qj = (d.qpos[7:] - default_angles) * dof_pos_scale
            dqj = d.qvel[6:] * dof_vel_scale
            gravity_orientation = get_gravity_orientation(d.qpos[3:7])
            omega = d.qvel[3:6] * ang_vel_scale

            period = 0.8
            phase = counter * dt % period / period
            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + n_act] = qj
            obs[9 + n_act:9 + 2 * n_act] = dqj
            obs[9 + 2 * n_act:9 + 3 * n_act] = action
            obs[9 + 3 * n_act:9 + 3 * n_act + 2] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            if first_obs is None:
                first_obs = (d.time, obs.copy())
                # 核对：手写公式 = 骨盆旋转矩阵的转置 × 世界重力方向 (0, 0, -1)
                # （d.xmat 是 mj_step 开头算的，比 qpos 晚一步，所以这里用四元数现算矩阵）
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, d.qpos[3:7])
                g_check = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
                grav_err = np.abs(g_check - gravity_orientation).max()
            action = policy(obs)
            target_dof_pos = action * action_scale + default_angles
            n_policy_calls += 1

        # (d) 记录与摔倒判定
        if contact_mu is None:
            for c in d.contact[:d.ncon]:
                if floor in (c.geom1, c.geom2):
                    contact_mu = c.friction[0]
                    break
        grav_z = get_gravity_orientation(d.qpos[3:7])[2]          # 直立时为 -1
        tilt = np.degrees(np.arccos(np.clip(-grav_z, -1, 1)))
        height = d.qpos[2]
        if fall_time is None:
            other_touch = any(
                floor in (c.geom1, c.geom2)
                and m.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1] not in foot_bodies
                for c in d.contact[:d.ncon])
            if height < 0.5:
                fall_time, fall_reason = d.time, f"骨盆高度 {height:.2f} m < 0.5 m"
            elif tilt > 45:
                fall_time, fall_reason = d.time, f"躯干倾角 {tilt:.1f}° > 45°"
            elif other_touch:
                fall_time, fall_reason = d.time, "脚以外的部位碰到了地面"
            if fall_time is not None:
                # 骨盆坐标系里的重力方向：前倾时 x 分量为正，向左倒时 y 分量为正
                gx, gy, _ = get_gravity_orientation(d.qpos[3:7])
                if abs(gx) >= abs(gy):
                    fall_dir = "向前" if gx > 0 else "向后"
                else:
                    fall_dir = "向左" if gy > 0 else "向右"
                fall_reason += f"，{fall_dir}倒"
        if step % 10 == 9:
            v_body = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, pelvis, v_body, 1)
            log.append([d.time, *d.qpos[:3], tilt, v_body[3], v_body[4], v_body[2]])

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            left = m.opt.timestep - (time.perf_counter() - step_start)
            if left > 0:
                time.sleep(left)

    wall = time.perf_counter() - t_wall
    if viewer is not None:
        viewer.close()

    # ---------- 4. 打印结果 ----------
    if first_obs is None or not log:
        print("仿真时间太短（查看器关得太早？），还没调用过策略，没有结果可打印。")
        return
    log = np.array(log)
    print(f"\n仿真 {d.time:.2f} s，共 {counter} 个仿真步，策略调用 {n_policy_calls} 次"
          f"（{counter}/{decim} = {counter // decim}），用时 {wall:.1f} s")
    t0, o0 = first_obs
    print(f"第一次组观测在 t = {t0:.3f} s：")
    print(f"  omega×{ang_vel_scale} = {np.round(o0[0:3], 3)}")
    print(f"  gravity  = {np.round(o0[3:6], 3)}")
    print(f"  cmd×scale= {np.round(o0[6:9], 3)}")
    print(f"  q-q0     = {np.round(o0[9:9 + n_act], 3)}")
    print(f"  相位     = {np.round(o0[-2:], 3)}")
    print(f"  重力投影公式与旋转矩阵算出的结果最大相差 {grav_err:.1e}")
    print(f"各关节 |PD 力矩| 最大值 (N·m)：{np.round(tau_abs_max, 0)}")
    print(f"PD 力矩超出关节力矩上限、被截断的次数：{n_sat}（共 {counter * n_act} 个关节·步）")
    if args.push_time is not None:
        print(f"推力实际作用了 {n_push_steps} 个仿真步 = {n_push_steps * dt:.3f} s，"
              f"冲量 ≈ {np.hypot(args.push_fx, args.push_fy) * n_push_steps * dt:.1f} N·s")
    if contact_mu is not None:
        print(f"脚与地面接触的实际摩擦系数：{contact_mu:.2f}")
    if fall_time is None:
        print(f"结果：没有摔倒（{d.time:.1f} s 内）")
    else:
        print(f"结果：摔倒，t = {fall_time:.2f} s，判据：{fall_reason}")
    alive = log[:, 0] <= (fall_time if fall_time is not None else np.inf)
    if alive.sum() > 50 and not args.dummy_policy:
        late = alive & (log[:, 0] > 2.0)
        if late.any():
            print(f"2 s 以后（摔倒前）骨盆坐标系下的平均速度：vx = {log[late, 5].mean():+.3f} m/s，"
                  f"vy = {log[late, 6].mean():+.3f} m/s，wz = {log[late, 7].mean():+.3f} rad/s")
    print(f"终点骨盆位置 x = {d.qpos[0]:+.2f} m，y = {d.qpos[1]:+.2f} m，z = {d.qpos[2]:.2f} m")

    # ---------- 5. 保存 CSV 和曲线 ----------
    OUT.mkdir(exist_ok=True)
    tag = ("dummy" if args.dummy_policy else "policy") + (f"_{args.tag}" if args.tag else "")
    csv_path = OUT / f"09_walk_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "tilt_deg", "vx_body", "vy_body", "wz"])
        w.writerows(np.round(log, 4).tolist())
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(log[:, 0], log[:, 3]); ax[0].set_ylabel("pelvis z (m)")
    ax[0].axhline(0.5, ls="--", c="r", lw=0.8)
    ax[1].plot(log[:, 0], log[:, 5], label="vx (body)")
    ax[1].axhline(cmd[0], ls="--", c="k", lw=0.8, label="cmd vx")
    ax[1].set_ylabel("m/s"); ax[1].legend(loc="upper right")
    ax[2].plot(log[:, 0], log[:, 4]); ax[2].set_ylabel("tilt (deg)")
    ax[2].axhline(45, ls="--", c="r", lw=0.8); ax[2].set_xlabel("time (s)")
    for a in ax:
        if fall_time is not None:
            a.axvline(fall_time, c="r", lw=0.8)
        if args.push_time is not None:
            a.axvspan(args.push_time, args.push_time + args.push_dur, color="orange", alpha=0.3)
    fig.suptitle(f"G1 walk: {policy_name.split('（')[0]}  cmd={cmd.round(2).tolist()}")
    fig.tight_layout()
    png_path = OUT / f"09_walk_{tag}.png"
    fig.savefig(png_path, dpi=100)
    print(f"已保存 {csv_path.name}、{png_path.name} 到 {OUT}")


if __name__ == "__main__":
    main()
