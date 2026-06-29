from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REQUIRED_FULL_PACK_ARTIFACTS = [
    "manifest.json",
    "metadata.json",
    "transcript.raw.json",
    "transcript.clean.json",
    "chunks.json",
    "context.md",
    "readable.md",
    "transcript.md",
]


def main() -> int:
    url = os.environ.get("VCTX_SMOKE_VIDEO_URL")
    if not url:
        print("Set VCTX_SMOKE_VIDEO_URL to a public video URL with subtitles.")
        return 2

    out = Path(os.environ.get("VCTX_SMOKE_OUT", ".tmp/smoke-url-subtitles"))
    if out.exists():
        shutil.rmtree(out)

    completed = subprocess.run(
        ["uv", "run", "vctx", "prepare", url, "--out", str(out)],
        text=True,
    )
    if completed.returncode != 0:
        return completed.returncode

    missing = [name for name in REQUIRED_FULL_PACK_ARTIFACTS if not (out / name).exists()]
    if missing:
        print(f"missing expected artifacts: {missing}")
        return 1

    print(f"ok: wrote full URL subtitle pack to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
