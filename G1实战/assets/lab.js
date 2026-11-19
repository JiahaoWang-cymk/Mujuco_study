// G1 实战课文页的“实验台”：在页面里改参数、运行课程脚本、看输出和源码。
// 需要用“启动官方文档.bat”（serve.py）打开页面；直接双击 html 时只显示等价的命令行。
(function () {
  "use strict";
  const API = "/api/lab/";
  const labs = [...document.querySelectorAll(".lab[data-spec]")];
  if (!labs.length) return;

  const el = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const k of kids) if (k) e.append(k);
    return e;
  };
  const api = (path, body) => fetch(API + path, body === undefined
    ? { headers: { "X-Lab": "1" } }
    : { method: "POST", headers: { "X-Lab": "1", "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then(r => r.json().then(j => { if (!r.ok) throw new Error(j.error || r.status); return j; }));

  function offline(msg) {
    labs.forEach(lab => { const p = lab.querySelector(".lab-off"); if (p) p.innerHTML = msg; });
  }

  if (location.protocol === "file:") return;   // 保留静态说明
  api("ping").then(info => {
    if (!info.mujoco) {
      offline(`当前服务用的 Python（<code>${info.executable}</code>）里没有 mujoco，不能运行脚本。按第 0 课装好后，用装了 mujoco 的 Python 重新运行 <code>启动官方文档.bat</code>。`);
      return;
    }
    labs.forEach(lab => build(lab, info));
  }).catch(() => offline("这个页面不是由资料包的本地服务打开的，不能直接运行。双击资料包根目录的 <code>启动官方文档.bat</code>，再从它打开的页面进入本课；或者把上面的命令复制到终端，在 <code>G1实战</code> 目录下运行。"));

  // ---------------------------------------------------------------- 右侧的实时仿真画面
  // 点“运行”时用 --web 启动脚本，脚本自己仿真，画面由 MuJoCo Studio 的网页查看器输出，嵌在这里
  const viewer = document.querySelector(".viewer");
  let streamToken = 0, lastStream = null;
  const EMPTY = viewer?.querySelector(".viewer-empty")?.cloneNode(true);
  function clearHead() {
    viewer.querySelectorAll(".viewer-head a, .viewer-head button").forEach(e => e.remove());
  }
  function startStream(lab, spec, args, label) {
    if (!viewer) return;
    lastStream = [lab, spec, args, label];
    const frame = viewer.querySelector(".viewer-frame");
    const what = viewer.querySelector(".viewer-what");
    document.querySelectorAll(".lab.playing").forEach(l => l.classList.remove("playing"));
    lab.classList.add("playing");
    what.textContent = `${spec.title}：${label}`;
    what.title = what.textContent;
    clearHead();
    frame.replaceChildren(el("p", { class: "viewer-empty", text: "正在启动实时仿真…（载入模型要几秒）" }));
    const token = ++streamToken;               // 连点几次“运行”时，只认最后一次
    api("stream", { script: spec.script, args }).then(r => {
      if (token !== streamToken) return;
      const iframe = el("iframe", { src: r.url + "?t=" + Date.now(), title: "实时仿真画面" });
      // 网页查看器偶尔载入后画布不出图；尺寸一变它会重建画面，所以载入后轻轻抖动两次
      iframe.addEventListener("load", () => [1500, 4000].forEach(ms => setTimeout(() => {
        if (token !== streamToken) return;
        iframe.style.width = "calc(100% - 2px)";
        setTimeout(() => { iframe.style.width = ""; }, 200);
      }, ms)));
      frame.replaceChildren(iframe);
      viewer.querySelector(".viewer-head").append(
        el("button", { type: "button", text: "重置", title: "用同样的参数从 0 秒重新开始（画面黑屏或卡住时也可以点）",
          onclick: () => startStream(...lastStream) }),
        el("button", { type: "button", text: "关闭仿真", title: "停掉后台的仿真，清空画面", onclick: stopStream }),
        el("a", { href: r.url, target: "_blank", text: "新窗口打开" }));
    }).catch(err => {
      if (token !== streamToken) return;
      frame.replaceChildren(el("p", { class: "viewer-empty", text: "实时画面没能启动：" + err.message }));
      viewer.querySelector(".viewer-head").append(
        el("button", { type: "button", text: "重试", onclick: () => startStream(...lastStream) }));
    });
  }
  function stopStream() {
    ++streamToken;                             // 作废还没返回的启动请求
    api("stream", { stop: true }).catch(() => {});
    document.querySelectorAll(".lab.playing").forEach(l => l.classList.remove("playing"));
    viewer.querySelector(".viewer-what").textContent = "已关闭";
    clearHead();
    viewer.querySelector(".viewer-frame").replaceChildren(EMPTY.cloneNode(true));
  }
  if (viewer) {
    // 离开页面时停掉实时仿真
    addEventListener("pagehide", () => fetch(API + "stream", { method: "POST", keepalive: true,
      headers: { "X-Lab": "1", "Content-Type": "application/json" }, body: '{"stop":true}' }));
  }

  // ---------------------------------------------------------------- 源码
  const sources = {};
  function loadSource(script) {
    return sources[script] || (sources[script] = fetch("code/" + script, { cache: "no-store" }).then(r => r.text()));
  }
  function renderSource(text, focus) {
    const lines = text.replace(/\r/g, "").split("\n");
    let a = -1, b = -1;
    if (focus) {
      a = lines.findIndex(l => l.startsWith(`def ${focus}(`));
      if (a >= 0) {
        b = a + 1;
        while (b < lines.length && (lines[b] === "" || /^\s/.test(lines[b]))) b++;
        while (b > a && lines[b - 1].trim() === "") b--;
      }
    }
    const pre = el("pre", { class: "lab-src", tabindex: "0" });
    lines.forEach((l, i) => {
      const row = el("span", { class: "ln" + (i >= a && i < b ? " hl" : "") });
      row.append(el("i", { text: String(i + 1) }), l + "\n");
      pre.append(row);
    });
    return { pre, first: a };
  }

  // ---------------------------------------------------------------- 一个实验台
  function build(lab, info) {
    const spec = JSON.parse(lab.dataset.spec);
    const off = lab.querySelector(".lab-off");
    const cmd = lab.querySelector(".lab-cmd code");
    if (off) off.remove();

    const inputs = [];
    const form = el("div", { class: "lab-fields" });
    (spec.fields || []).forEach((f, i) => {
      const id = `lab-${spec.script}-${f.flag}-${Math.random().toString(36).slice(2, 7)}`;
      let input;
      if (f.type === "check") {
        input = el("input", { type: "checkbox", id });
        if (f.checked) input.checked = true;
        form.append(el("label", { class: "chk", for: id }, input, el("span", { text: f.label })));
      } else if (f.type === "select") {
        input = el("select", { id });
        for (const [v, t] of f.options) input.append(new Option(t, v, false, String(v) === String(f.value)));
        form.append(el("label", { class: "num", for: id }, el("span", { text: f.label }), input));
      } else {
        input = el("input", { type: "number", id, value: f.value ?? "", step: f.step || "any", placeholder: f.placeholder || "" });
        if (f.min !== undefined) input.min = f.min;
        if (f.max !== undefined) input.max = f.max;
        form.append(el("label", { class: "num", for: id }, el("span", { text: f.label }), input,
          f.unit ? el("small", { text: f.unit }) : null));
      }
      input.addEventListener("input", update);
      inputs.push([f, input]);
    });

    function args(forStream) {
      const out = forStream ? [...(spec.streamRun || [])] : [...(spec.run || [])];
      for (const [f, input] of inputs) {
        if (forStream ? f.onceOnly : f.streamOnly) continue;
        if (f.type === "check") { if (input.checked) out.push(f.flag); }
        else if (f.type === "select") out.push(f.flag, input.value);
        else if (input.value !== "" && String(input.value) !== String(f.omitIf ?? "\u0000")) out.push(f.flag, String(Number(input.value)));
      }
      return out;
    }
    function update() {
      cmd.textContent = ["python", "code/" + spec.script, ...args()].join(" ");
    }
    update();

    const runBtn = el("button", { type: "button", class: "primary", text: spec.runLabel || "运行" });
    const stopBtn = el("button", { type: "button", text: "停止", hidden: "" });
    const srcBtn = el("button", { type: "button", class: "ghost", text: "查看源码", "aria-expanded": "false" });
    const status = el("span", { class: "lab-status", role: "status" });
    const bar = el("div", { class: "lab-actions" }, runBtn, stopBtn, srcBtn, status);
    const log = el("pre", { class: "lab-log", hidden: "", "aria-live": "polite" });
    const results = el("div", { class: "lab-results" });
    const srcBox = el("div", { class: "lab-srcbox", hidden: "" });

    lab.querySelector(".lab-cmd").before(form);
    lab.querySelector(".lab-cmd").after(bar, srcBox, log, results);
    lab.classList.add("on");

    srcBtn.addEventListener("click", () => {
      const open = srcBox.hidden;
      srcBox.hidden = !open;
      srcBtn.setAttribute("aria-expanded", String(open));
      srcBtn.textContent = open ? "收起源码" : "查看源码";
      if (open && !srcBox.firstChild) {
        srcBox.append(el("p", { class: "lab-hint", text: "读取中…" }));
        loadSource(spec.script).then(text => {
          const { pre, first } = renderSource(text, spec.focus);
          srcBox.replaceChildren(
            el("p", { class: "lab-hint" }, `code/${spec.script}，共 ${text.split("\n").length} 行`,
              spec.focus ? `；高亮的是 ${spec.focus}()` : ""), pre);
          if (first >= 0) pre.scrollTop = Math.max(0, pre.children[first].offsetTop - 12);
        }).catch(() => srcBox.replaceChildren(el("p", { class: "lab-hint", text: "读不到源码文件。" })));
      }
    });

    let current = null;
    function label() {
      return inputs.filter(([f]) => !f.onceOnly).map(([f, input]) => f.type === "check"
        ? (input.checked ? f.label : "")
        : f.type === "select" ? `${f.label} ${input.selectedOptions[0]?.text ?? input.value}`
        : input.value === "" ? "" : `${f.label} ${Number(input.value)}${f.unit ? " " + f.unit : ""}`)
        .filter(Boolean).join("，") || "默认参数";
    }
    function start() {
      if (current) return;
      const a = args();
      if (spec.stream) startStream(lab, spec, args(true), label());
      log.hidden = false;
      log.textContent = "$ python code/" + spec.script + " " + a.join(" ") + "\n";
      results.replaceChildren();
      runBtn.disabled = true;
      stopBtn.hidden = false;
      status.className = "lab-status busy";
      status.textContent = "运行中…";
      api("run", { script: spec.script, args: a }).then(({ id }) => {
        current = id;
        let from = 0;
        const tick = () => api(`poll?id=${id}&from=${from}`).then(r => {
          if (r.lines.length) {
            const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
            log.append(r.lines.join("\n") + "\n");
            if (atBottom) log.scrollTop = log.scrollHeight;
          }
          from = r.next;
          if (!r.done) {
            status.textContent = `运行中… ${r.elapsed.toFixed(0)} s`;
            setTimeout(tick, 400);
            return;
          }
          finish(r);
        }).catch(err => finish({ rc: -1, files: [], elapsed: 0, error: err.message }));
        tick();
      }).catch(err => finish({ rc: -1, files: [], elapsed: 0, error: err.message }));
    }
    function finish(r) {
      current = null;
      runBtn.disabled = false;
      stopBtn.hidden = true;
      if (r.error) {
        status.className = "lab-status bad";
        status.textContent = "没能运行：" + r.error;
        return;
      }
      if (r.stopped) {
        status.className = "lab-status";
        status.textContent = "已停止";
      } else if (r.rc === 0) {
        status.className = "lab-status ok";
        status.textContent = `完成，用时 ${r.elapsed.toFixed(1)} s`;
      } else {
        status.className = "lab-status bad";
        status.textContent = `脚本出错（退出码 ${r.rc}），看上面的输出，对照本课“如果报错”`;
      }
      const want = spec.show || [];
      const files = r.files.filter(f => !want.length || want.includes(f));
      const stamp = Date.now();
      for (const f of files) {
        const url = f + "?t=" + stamp;
        if (/\.(png|jpe?g|gif|svg)$/i.test(f)) {
          results.append(el("figure", {}, el("a", { href: url, target: "_blank" }, el("img", { src: url, alt: f })),
            el("figcaption", { text: f })));
        } else if (/\.(mp4|webm)$/i.test(f)) {
          results.append(el("figure", {}, el("video", { src: url, controls: "", muted: "" }), el("figcaption", { text: f })));
        } else {
          results.append(el("p", { class: "lab-file" }, el("a", { href: url, target: "_blank", text: f })));
        }
      }
    }
    runBtn.addEventListener("click", () => start());
    stopBtn.addEventListener("click", () => { if (current) api("stop", { id: current }); });
  }
})();
