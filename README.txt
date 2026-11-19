MuJoCo 离线资料包（整理于 2026-11-17，对应 MuJoCo 4.0.1）

入口：双击 index.html（资料包首页），所有资料都能从首页点进去。

MuJoCo入门教程.pdf / .docx / .md
    中文入门教程。网页版在中文文档左侧导航第一项“入门教程”，每个小节开头有跳到官方文档对应部分的链接。
    PDF/Word 正文用微软雅黑、代码用 Consolas（PDF 已内嵌字体）。
    Markdown 版的示意图在 images\ 目录，与 .md 一起拷贝。

官方文档_中文HTML\
    官方文档 mujoco.readthedocs.io 的中文全译版（约 32 万字），包括 XML 参考、API 参考和更新日志。
    未翻译：XML 单位表（XMLunits，基本是属性名和符号）、MJX/Warp 的 Python API 自动生成页。
    由 Claude Sonnet 机器翻译，每页顶部有说明；如有出入，以英文版为准。
    中英对照：页面右上角“中英对照”按钮显示全部英文原文；鼠标移到某段左侧会出现 EN 按钮，只展开这一段。
    搜索支持中文。

官方文档_HTML\
    英文原版官方文档，用官方 GitHub 仓库 main 分支源码（提交 aa9a0ea，2026-11-16）在本地编译而成。
    数学公式所需的 KaTeX 字体和样式已放到本地。
    已知限制：页面里嵌入的 YouTube 视频、指向外部网站的链接需要联网才能打开。

MuJoCo_Live\
    官方 live.mujoco.org 的离线版：整个 MuJoCo（物理仿真 + 渲染 + 界面）编译成 WebAssembly 在浏览器里运行，不用装 Python 也能看模型。
    要通过“启动官方文档.bat”打开（直接双击 html 不行），首页“其他资料”里有入口，默认打开 G1 人形机器人。
    详见 MuJoCo_Live\说明.txt。

G1实战\（编写中）
    以宇树 Unitree G1 人形机器人为案例的实战课程。
    models\unitree_g1\          G1 模型（DeepMind MuJoCo Menagerie，BSD-3-Clause）
    models\unitree_rl_gym_g1\   宇树官方 G1 行走策略 motion.pt、部署脚本和 12 自由度模型（unitree_rl_gym，BSD-3-Clause）

术语表.md
    中文版用到的术语对照，例如 actuator 译作“执行器”。

启动官方文档.bat
    如果双击 index.html 后文档页面打开是空白，或者搜索不能用，改用这个脚本。
    它会在本机起一个 HTTP 服务（http://localhost:8765）再打开首页，需要电脑上装有 Python。

官方仓库_模型与教程\
    model\                示例模型（humanoid、car、flex 等），可直接用查看器打开
    python\*.ipynb        官方 Python 教程 notebook（tutorial、LQR、least_squares、rollout、mjspec）
    mjx\*.ipynb           MJX 教程 notebook（tutorial、training_apg）
    LICENSE               MuJoCo 的 Apache 2.0 许可证
    注意：notebook 里用到的 pip install、git clone、下载模型等步骤需要联网；没有网络时改用本地的 wheel 和模型文件。
