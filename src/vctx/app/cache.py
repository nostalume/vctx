from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vctx.config import PrepareRequest, resolve_config
from vctx.errors import CacheError, ConfigError
from vctx.source.store import CacheInventory, PruneReceipt, SourceStore


def manage_cache(
    action: Literal["status", "prune"], *, config_path: Path | None, cache_dir: Path | None,
    age: str | None = None, all_records: bool = False, dry_run: bool = False
) -> CacheInventory | PruneReceipt:
    if age is not None and all_records:
        raise ConfigError("--age and --all are mutually exclusive")
    root = resolve_config(
        PrepareRequest(
            inputs=["cache-operation"], out_dir=Path("."), config_path=config_path,
            cache_dir=cache_dir,
        )
    ).cache.source_dir
    try:
        store = SourceStore(root)
        if action == "status":
            return store.inventory()
        before = datetime.now(UTC) - _age(age) if age is not None else None
        return store.prune(before=before, all_records=all_records, dry_run=dry_run)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise CacheError(f"source cache {action} failed: {exc}") from exc

def render_cache(report: BaseModel, *, json_output: bool) -> str:
    if json_output:
        return report.model_dump_json(indent=2) + "\n"
    lines = []
    for name, value in report.model_dump(mode="json").items():
        if name.endswith("bytes"):
            value = f"{value:,} B"
        elif isinstance(value, list):
            value = ", ".join(value) or "none"
        lines.append(f"{name.replace('_', ' ')}: {value}")
    return "\n".join(lines) + "\n"

def _age(value: str) -> timedelta:
    match = re.fullmatch(r"([1-9][0-9]*)([dhw])", value.casefold())
    if match is None:
        raise ConfigError("cache age must be a positive duration such as 30d, 12h, or 4w")
    amount = int(match.group(1))
    unit = {"h": "hours", "d": "days", "w": "weeks"}[match.group(2)]
    return timedelta(**{unit: amount})
