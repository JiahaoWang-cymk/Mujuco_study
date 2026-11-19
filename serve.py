"""资料包的本地服务：提供网页，并让 G1 实战课文页里的“实验台”运行课程脚本。

    python serve.py            # 默认端口 8765，只监听本机
    python serve.py 8800       # 换端口

由“启动官方文档.bat”调用。只用 Python 标准库；运行课程脚本用的是启动本服务的同一个 Python，
所以要用装了 mujoco 的那个 Python 来启动。

实验台接口（只接受本机、带 X-Lab 请求头的请求，网页以外的程序和其他网站调用不了）：
  GET  /api/lab/ping                  Python 和 MuJoCo 版本
  POST /api/lab/run   {script, args}  运行 G1实战/code/ 下的一个脚本，返回 id
  GET  /api/lab/poll?id=..&from=n     第 n 行起的新输出；结束后附带退出码和新生成的文件
  POST /api/lab/stop  {id}            停止
  POST /api/lab/stream {script, args} 用 --web 启动脚本的实时仿真画面（同时只有一个，新的会替换旧的），
                                      等网页查看器起来后返回它的地址；{stop: true} 则停止
"""
import http.server, json, os, pathlib, re, socket, subprocess, sys, threading, time, urllib.parse, uuid

ROOT = pathlib.Path(__file__).resolve().parent
G1 = ROOT / 'G1实战'
CODE, OUTPUT = G1 / 'code', G1 / 'output'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

SCRIPT_RE = re.compile(r'\d\d_[A-Za-z0-9_]+\.py')
ARG_RE = re.compile(r'[-A-Za-z0-9_.=]{1,64}')
MAX_RUNNING = 3
RUNS = {}
STREAM_PORT = PORT + 25                      # 实时仿真画面（Studio 网页查看器）用的端口，默认 8790
STREAM = {'run': None}
STREAM_LOCK = threading.Lock()              # 启动/替换实时画面的请求逐个处理，避免两个进程抢端口
STREAM_READY = '网页查看器已就绪'           # 与 G1实战/code/webview.py 的 READY 一致
LOCK = threading.Lock()
_versions = None


def versions():
    global _versions
    if _versions is None:
        try:
            r = subprocess.run([sys.executable, '-c', 'import mujoco; print(mujoco.__version__)'],
                               capture_output=True, text=True, timeout=60)
            mj = r.stdout.strip() or None
        except Exception:
            mj = None
        _versions = {'python': sys.version.split()[0], 'executable': sys.executable, 'mujoco': mj}
    return _versions


class Run:
    def __init__(self, script, args):
        self.id = uuid.uuid4().hex[:12]
        self.script, self.args = script, args
        self.lines, self.rc, self.files, self.stopped = [], None, [], False
        self.start = time.time()
        # LAB_PARENT_PID：脚本的网页查看器发现本服务退出后自己结束（见 code/webview.py）
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
                   LAB_PARENT_PID=str(os.getpid()))
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.proc = subprocess.Popen([sys.executable, '-u', str(CODE / script), *args], cwd=G1, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     creationflags=flags)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.proc.stdout:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            if line.startswith('libpng warning: iCCP'):   # 模型贴图自带的无害警告，会刷屏
                continue
            with LOCK:
                self.lines.append(line.split('\r')[-1])
        rc = self.proc.wait()
        files = []
        if OUTPUT.exists():
            for p in sorted(OUTPUT.iterdir()):
                if p.is_file() and p.stat().st_mtime >= self.start - 1:
                    files.append('output/' + p.name)
        with LOCK:
            self.files, self.rc = files, rc
            self.elapsed = time.time() - self.start

    def running(self):
        return self.rc is None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if self.path.startswith('/api/lab/run'):
            sys.stderr.write('[实验台] ' + (fmt % args) + '\n')

    def end_headers(self):
        # 脚本输出的图片每次运行都会变，不要让浏览器缓存
        if '/output/' in urllib.parse.unquote(self.path) or self.path.startswith('/api/'):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    # ---------------------------------------------------------------- 接口
    def _allowed(self):
        host = self.headers.get('Host', '')
        ok_host = host in (f'localhost:{PORT}', f'127.0.0.1:{PORT}', f'[::1]:{PORT}')
        return ok_host and self.headers.get('X-Lab') == '1'

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 10000:
            raise ValueError('请求太大')
        return json.loads(self.rfile.read(n) or b'{}')

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if not url.path.startswith('/api/lab/'):
            return super().do_GET()
        if not self._allowed():
            return self._json({'error': 'forbidden'}, 403)
        q = urllib.parse.parse_qs(url.query)
        if url.path == '/api/lab/ping':
            return self._json({'ok': True, **versions()})
        if url.path == '/api/lab/poll':
            run = RUNS.get(q.get('id', [''])[0])
            if not run:
                return self._json({'error': '没有这次运行'}, 404)
            start = int(q.get('from', ['0'])[0])
            with LOCK:
                out = {'lines': run.lines[start:], 'next': len(run.lines), 'done': run.rc is not None,
                       'rc': run.rc, 'stopped': run.stopped, 'files': run.files, 'elapsed': time.time() - run.start}
            return self._json(out)
        self._json({'error': 'not found'}, 404)

    def do_POST(self):
        url = urllib.parse.urlsplit(self.path)
        if not self._allowed():
            return self._json({'error': 'forbidden'}, 403)
        try:
            data = self._body()
        except Exception:
            return self._json({'error': '请求格式不对'}, 400)
        if url.path == '/api/lab/run':
            script, args = data.get('script', ''), data.get('args', [])
            if not (isinstance(script, str) and SCRIPT_RE.fullmatch(script) and (CODE / script).is_file()):
                return self._json({'error': f'只能运行 G1实战\\code\\ 里的课程脚本：{script}'}, 400)
            if not (isinstance(args, list) and len(args) <= 30 and all(isinstance(a, str) and ARG_RE.fullmatch(a) for a in args)):
                return self._json({'error': '参数里有不允许的字符'}, 400)
            with LOCK:
                busy = sum(r.running() for r in RUNS.values())
            if busy >= MAX_RUNNING:
                return self._json({'error': f'已经有 {busy} 个脚本在运行，先等它们结束或停止'}, 429)
            run = Run(script, args)
            RUNS[run.id] = run
            return self._json({'id': run.id})
        if url.path == '/api/lab/stream':
            return self._stream(data)
        if url.path == '/api/lab/stop':
            run = RUNS.get(data.get('id', ''))
            if run and run.running():
                run.stopped = True
                run.proc.kill()
            return self._json({'ok': True})
        self._json({'error': 'not found'}, 404)


    def _stream(self, data):
        with STREAM_LOCK:
            return self._stream_locked(data)

    def _stream_locked(self, data):
        with LOCK:
            old, STREAM['run'] = STREAM['run'], None
        if old and old.running():
            old.stopped = True
            old.proc.kill()
            old.proc.wait(10)
        if data.get('stop'):
            return self._json({'ok': True})
        script, args = data.get('script', ''), data.get('args', [])
        if not (isinstance(script, str) and SCRIPT_RE.fullmatch(script) and (CODE / script).is_file()):
            return self._json({'error': f'只能运行 G1实战\\code\\ 里的课程脚本：{script}'}, 400)
        if not (isinstance(args, list) and len(args) <= 30 and all(isinstance(a, str) and ARG_RE.fullmatch(a) for a in args)):
            return self._json({'error': '参数里有不允许的字符'}, 400)
        run = Run(script, [*args, '--web', str(STREAM_PORT)])
        RUNS[run.id] = run
        with LOCK:
            STREAM['run'] = run
        # 等脚本打印“网页查看器已就绪”（code/webview.py 在模型交给查看器之后才打印），
        # 太早让浏览器连上，可能停在黑屏
        deadline = time.time() + 180       # 有的脚本要先算一阵（例如第 8 课求 LQR）再出画面
        while time.time() < deadline and run.running():
            with LOCK:
                ready = STREAM_READY in run.lines
            if ready:
                return self._json({'id': run.id, 'url': f'http://127.0.0.1:{STREAM_PORT}/'})
            time.sleep(0.2)
        with LOCK:
            tail = run.lines[-8:]
        return self._json({'error': '实时画面没有启动起来', 'lines': tail, 'id': run.id}, 500)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


def main():
    servers = []
    for family, addr in ((socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')):
        try:
            cls = type('S', (Server,), {'address_family': family})
            servers.append(cls((addr, PORT), Handler))
        except OSError as e:
            if family == socket.AF_INET:
                sys.exit(f'端口 {PORT} 打不开（{e}）。可能已经有一个服务在运行：直接在浏览器打开 http://localhost:{PORT}/ 即可。')
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f'资料包已在 http://localhost:{PORT}/ 提供。关掉这个窗口即停止。', flush=True)
    v = versions()                           # 要 import 一次 mujoco，放在服务启动之后
    if v['mujoco']:
        print(f'实验台可用：Python {v["python"]}，MuJoCo {v["mujoco"]}（{v["executable"]}）', flush=True)
    else:
        print(f'提示：这个 Python（{v["executable"]}）里没有 mujoco，课文页的实验台不能运行脚本，网页照常可看。', flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for r in list(RUNS.values()):
            if r.running():
                r.proc.kill()


if __name__ == '__main__':
    main()
