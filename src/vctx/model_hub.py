from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PATTERNS = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "preprocessor_config.json",
]


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 2:
        return 2
    repo_id, destination = values
    try:
        hub = importlib.import_module("huggingface_hub")
        hub.snapshot_download(
            repo_id,
            local_dir=str(Path(destination)),
            allow_patterns=_PATTERNS,
            max_workers=16,
        )
    except Exception as exc:
        print(f"model download failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
