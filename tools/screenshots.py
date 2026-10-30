"""Screenshot every app section in both themes into docs/images/ (final polish).

    cd app && npm run build && cd ..
    uv run --with playwright==1.64.1 python tools/screenshots.py

Serves the built app (``app/dist``) with ``vite preview`` on 127.0.0.1, opens each hash route in
headless Chromium in the dark theme and the light ("corporate") theme, and writes
``docs/images/<section>-<theme>.png``. It also writes the three.js stills of the term-structure
ribbon (``hero-ribbon-<theme>.png``), which stand in for the Blender hero renders (Blender was
unreachable). Chromium: ``$DIAGRAM_KIT_CHROMIUM`` if set (the diagram render gate's browser), else
Playwright's own. Every request to a host other than 127.0.0.1 is aborted and counted; the run fails
if there was one. The app ships synthetic data only, so every image shows synthetic data.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
OUT = REPO / "docs" / "images"
PORT = 4317
BASE = f"http://127.0.0.1:{PORT}/"
SECTIONS = ("about", "architecture", "eda", "term-structure", "model-race", "evaluation", "backtest")
THEMES = ("dark", "corporate")
THEME_KEY = "ivf:implied-vol-forecasting-app:theme"  # app/src/lib/theme.ts themeKey(APP_SLUG)


def wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.3)
    raise SystemExit(f"vite preview did not answer on {BASE}")


def main() -> int:
    from playwright.sync_api import sync_playwright

    if not (APP / "dist" / "index.html").is_file():
        print("error: app/dist is missing; run `npm run build` in app/ first", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(["npx", "vite", "preview", "--host", "127.0.0.1", "--port", str(PORT), "--strictPort"],
                              cwd=APP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    blocked: list[str] = []
    written: list[Path] = []
    try:
        wait_for_server()
        exe = os.environ.get("DIAGRAM_KIT_CHROMIUM")
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=exe, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]) if exe else \
                p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            for theme in THEMES:
                context = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=1,
                                              color_scheme="dark" if theme == "dark" else "light")
                context.add_init_script(f"window.localStorage.setItem({THEME_KEY!r}, {theme!r})")

                def guard(route):
                    host = urlparse(route.request.url).hostname
                    if host in ("127.0.0.1", "localhost"):
                        route.continue_()
                    else:
                        blocked.append(route.request.url)
                        route.abort()

                context.route("**/*", guard)
                page = context.new_page()
                for section in SECTIONS:
                    page.goto(f"{BASE}#/{section}")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(2500 if section == "term-structure" else 1200)  # line draw-in and the WebGL frame
                    target = OUT / f"{section}-{theme}.png"
                    page.screenshot(path=str(target), full_page=True)
                    written.append(target)
                    if section == "term-structure" and page.locator(".ribbon-stage").count():
                        hero = OUT / f"hero-ribbon-{theme}.png"
                        page.locator(".ribbon-stage").screenshot(path=str(hero))
                        written.append(hero)
                context.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    for path in written:
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)")
    if blocked:
        print(f"error: {len(blocked)} request(s) to another host were blocked: {blocked[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
