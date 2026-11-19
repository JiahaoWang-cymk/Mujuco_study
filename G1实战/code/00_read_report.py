"""第 0 课：读 pip 的安装报告，列出要下载的 wheel 清单。

先在能上网的机器上生成报告（只解析依赖，不下载、不安装，完整命令见第 0 课 0.3 节）：
    python -m pip install --dry-run --ignore-installed --report report.json --only-binary=:all: --platform win_amd64 --python-version 3.13 -r requirements.txt
再运行：
    python code/00_read_report.py report.json
"""
import json
import pathlib
import sys
from urllib.parse import unquote

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
items = report["install"]                      # pip 打算安装的每一个包

print(f"报告 {path.name}：共 {len(items)} 个包\n")
# 一个汉字在终端里占两格，所以表头的宽度要比下面的数据行少算几格才能对齐
print(f"{'包名':<18}{'版本':<12}{'Python 要求':<14}wheel 文件名")
for it in sorted(items, key=lambda x: x["metadata"]["name"].lower()):
    meta = it["metadata"]
    url = it["download_info"]["url"]
    filename = unquote(url.rsplit("/", 1)[-1])   # 把 %2B 还原成 +
    need = meta.get("requires_python") or "-"
    who = "（直接要求）" if it.get("requested") else ""
    print(f"{meta['name']:<20}{meta['version']:<14}{need:<16}{filename}{who}")

direct = sum(1 for it in items if it.get("requested"))
print(f"\n直接要求 {direct} 个，被依赖带进来 {len(items) - direct} 个。")
print("\n提示：最后一列文件名里的 cp313 / win_amd64 等标签，必须和目标机器一致。")
