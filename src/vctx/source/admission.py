from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

from vctx.config import YtDlpSourceOptions
from vctx.errors import CacheError, UnsupportedSourceError
from vctx.net import NetRuntime
from vctx.source.local import LocalFileSourceAdapter
from vctx.source.session import ObservePermit, SourceSession
from vctx.source.store import SourceStore
from vctx.source.ytdlp import YtDlpSourceAdapter

SourceClaim = Literal["exact", "fallback", "unsupported"]
SourceSelection = Literal["local-file", "yt-dlp"]


class SourceAdapter(Protocol):
    name: str

    def claim(self, value: str) -> SourceClaim: ...

    def observe(
        self, value: str, *, permit: ObservePermit, options: YtDlpSourceOptions
    ) -> SourceSession: ...


def select_source(value: str) -> SourceSelection:
    if LocalFileSourceAdapter().claim(value) == "exact":
        return "local-file"
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return "yt-dlp"
    raise UnsupportedSourceError(f"unsupported source: {value}")


def open_source(
    selection: SourceSelection,
    value: str,
    *,
    permit: ObservePermit,
    options: YtDlpSourceOptions,
    net: NetRuntime,
    store: SourceStore | None = None,
) -> SourceSession:
    if selection == "local-file" and not Path(value).is_file():
        raise UnsupportedSourceError(f"local source does not exist: {value}")
    if permit.network == "denied" and store is not None and selection != "local-file":
        cached = store.get(value)
        if cached is not None:
            return cached
    adapter: SourceAdapter = (
        LocalFileSourceAdapter() if selection == "local-file" else YtDlpSourceAdapter(net=net)
    )
    session = adapter.observe(value, permit=permit, options=options)
    if store is None or session.record.metadata.source.kind != "url":
        return session
    try:
        return store.wrap(value, session)
    except CacheError:
        return session
