from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_PATTERNS = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "preprocessor_config.json",
]


def download_model(
    capability: str,
    model_id: str,
    destination: Path,
    *,
    cache_root: Path,
    conservative: bool,
    max_runtime: int,
) -> None:
    if capability == "asr":
        _download_asr(
            model_id,
            destination,
            conservative=conservative,
            max_runtime=max_runtime,
        )
        return
    _download_ocr(destination, cache_root)


def _download_asr(
    model_id: str,
    destination: Path,
    *,
    conservative: bool,
    max_runtime: int,
) -> None:
    from vctx.errors import DeadlineExceededError
    from vctx.supervise import run_supervised

    environment = os.environ.copy()
    if conservative:
        environment.pop("HF_XET_HIGH_PERFORMANCE", None)
    else:
        environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    repo_id = model_id if "/" in model_id else f"Systran/faster-whisper-{model_id}"
    command = [sys.executable, "-m", "vctx.model.download", repo_id, str(destination)]
    completed = run_supervised(command, b"", timeout_s=max_runtime, environment=environment)
    if completed.returncode == 124:
        raise DeadlineExceededError("model download exceeded --max-runtime")
    if completed.returncode:
        raise RuntimeError(
            f"Hugging Face model download failed with exit code {completed.returncode}"
        )


def _download_ocr(destination: Path, cache_root: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("rapidocr")
    module_file = module.__file__
    if module_file is None:
        raise ImportError("rapidocr has no filesystem package location")
    template = Path(module_file).with_name("config.yaml")
    config = template.read_text(encoding="utf-8").replace(
        "model_root_dir: null", f'model_root_dir: "{destination.as_posix()}"'
    )
    config_path = cache_root / "ocr" / "rapidocr" / "config.yaml"
    config_path.write_text(config, encoding="utf-8")
    module.download_models(config_path)


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
