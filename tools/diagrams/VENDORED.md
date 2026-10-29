# Vendored: vol-diagram-kit

- Source: a sibling checkout of the kit, https://github.com/rajib2k5/vol-diagram-kit (`diagram_kit/` and `build.py`)
  (amended M1, review B: this line held an absolute local path with the user name)
- Commit: `7493b4b` (full hash `7493b4b5222b7a72090345f3d1944c9fa274a332`)
- Vendored on: 2026-10-29

Do not edit here; change the kit and re-vendor.

Build and check (the `diagrams` dependency group in `pyproject.toml` supplies `pyyaml` and `resvg-py`):

```bash
uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out
uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out --check
```
