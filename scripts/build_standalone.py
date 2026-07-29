"""docs/index.html + docs/data/*.json → 단일 HTML 파일.

fetch가 막힌 환경(file:// 로 직접 열기, CSP가 엄격한 호스팅)에서도 열리도록
모든 리포트 JSON을 <script> 안에 인라인한다. 비공개 저장소라 GitHub Pages를
못 쓰는 경우의 대안이자, 그냥 파일 하나로 공유하고 싶을 때 쓰는 경로다.

    python scripts/build_standalone.py [출력경로]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"
TEMPLATE = ROOT / "docs" / "index.html"

# </script> 가 문자열 안에 있으면 브라우저가 스크립트를 거기서 끊는다.
ESCAPES = {"</script": "<\\/script", "<!--": "<\\!--"}


def build(output: Path) -> Path:
    index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))

    reports = {}
    for entry in index.get("reports", []):
        path = DATA_DIR / f"{entry['date']}.json"
        if path.exists():
            reports[entry["date"]] = json.loads(path.read_text(encoding="utf-8"))

    blob = json.dumps({"index": index, "reports": reports}, ensure_ascii=False)
    for needle, replacement in ESCAPES.items():
        blob = blob.replace(needle, replacement)

    html = TEMPLATE.read_text(encoding="utf-8")
    injection = f"<script>window.__EMBEDDED__ = {blob};</script>\n<script>"

    marker = "<script>\nconst BLOCKS"
    if marker not in html:
        raise SystemExit("템플릿에서 스크립트 시작 지점을 못 찾았습니다.")
    html = html.replace(marker, injection + "\nconst BLOCKS", 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"{output} ({len(html) / 1024:.0f}KB, 리포트 {len(reports)}건)")
    return output


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "daily-trading.html"
    build(target)
