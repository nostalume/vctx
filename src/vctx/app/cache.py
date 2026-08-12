from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from vctx.config import ResolvedConfig
from vctx.errors import CacheError, ConfigError
from vctx.source.store import CacheInventory, PruneReceipt, SourceStore


@dataclass(frozen=True)
class Cache:
    root: Path

    @classmethod
    def open(cls, resolved: ResolvedConfig) -> Cache:
        return cls(resolved.cache.source_dir)

    def status(self) -> CacheInventory:
        return cache_status(self.root)

    def prune(
        self, *, age: str | None = None, all_records: bool = False, dry_run: bool = False
    ) -> PruneReceipt:
        return prune_cache(
            self.root, age=age, all_records=all_records, dry_run=dry_run
        )


def cache_status(root: Path) -> CacheInventory:
    try:
        return SourceStore(root).inventory()
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise CacheError(f"source cache status failed: {exc}") from exc


def prune_cache(
    root: Path, *, age: str | None = None, all_records: bool = False, dry_run: bool = False
) -> PruneReceipt:
    if age is not None and all_records:
        raise ConfigError("--age and --all are mutually exclusive")
    try:
        store = SourceStore(root)
        before = datetime.now(UTC) - _age(age) if age is not None else None
        return store.prune(before=before, all_records=all_records, dry_run=dry_run)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise CacheError(f"source cache prune failed: {exc}") from exc


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
