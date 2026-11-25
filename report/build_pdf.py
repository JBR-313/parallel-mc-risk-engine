"""Render report/FINAL_REPORT.md to report/FINAL_REPORT.pdf.

Markdown -> HTML (python-markdown, tables) -> PDF with headless Chromium.

  python report/build_pdf.py [--chrome /path/to/chrome]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Inter", "Noto Sans CJK KR", "Noto Sans KR", "Helvetica Neue", Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.5; color: #16202a; }
h1 { font-size: 22pt; margin: 0 0 4pt; color: #13202e; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; color: #13202e; border-bottom: 1.5pt solid #2a78d6; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; color: #13202e; }
h2, h3 { page-break-after: avoid; break-after: avoid; }
p { margin: 4pt 0; }a { color: #2a78d6; }
hr { border: none; border-top: 0.75pt solid #c9ced6; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 0.6pt solid #c9ced6; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eef2f7; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f2f4f7; padding: 0 2pt; }
pre { background: #f2f4f7; padding: 6pt 8pt; font-size: 8.5pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { width: 100%; margin: 6pt 0; }
ul { margin: 4pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }
"""


def find_chrome() -> str:
    for c in (shutil.which("chromium"), shutil.which("google-chrome"),
              *sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")))):
        if c:
            return c
    raise SystemExit("no Chromium found; pass --chrome")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome")
    a = ap.parse_args()
    src = HERE / "FINAL_REPORT.md"
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html = HERE / "FINAL_REPORT.html"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Final Report</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    pdf = HERE / "FINAL_REPORT.pdf"
    subprocess.run([a.chrome or find_chrome(), "--headless", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf.name}", html.as_uri()],
                   cwd=HERE, check=True, capture_output=True)
    html.unlink()
    print(f"wrote {pdf.relative_to(HERE.parent)} ({pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
