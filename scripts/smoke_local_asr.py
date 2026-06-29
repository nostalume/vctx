from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    media = os.environ.get("VCTX_ASR_SMOKE_MEDIA")
    if not media:
        print("Set VCTX_ASR_SMOKE_MEDIA to a local audio/video file to run the real ASR smoke.")
        return 2
    media_path = Path(media)
    if not media_path.exists():
        print(f"VCTX_ASR_SMOKE_MEDIA does not exist: {media_path}")
        return 2

    out_dir = Path(os.environ.get("VCTX_ASR_SMOKE_OUT", ".tmp/asr-real-smoke/out"))
    config_path = out_dir.parent / "vctx.toml"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
[transforms.asr]
instance = "local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model_policy = "tiny"
cache = "persistent"
""".strip(),
        encoding="utf-8",
    )

    executable = shutil.which("vctx")
    if executable is None:
        print(
            "Could not find vctx executable on PATH. "
            "Run this through `uv run python scripts/smoke_local_asr.py`."
        )
        return 2

    command = [
        executable,
        "prepare",
        str(media_path),
        "--out",
        str(out_dir),
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(command, text=True, check=False)  # noqa: S603
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
