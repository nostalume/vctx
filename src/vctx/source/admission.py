from __future__ import annotations

from typing import Literal, Protocol

from vctx.config import YtDlpSourceOptions
from vctx.errors import CacheError, UnsupportedSourceError
from vctx.source.local import LocalFileSourceAdapter
from vctx.source.session import ObservePermit, SourceSession
from vctx.source.store import SourceStore
from vctx.source.ytdlp import YtDlpSourceAdapter

SourceClaim = Literal["exact", "fallback", "unsupported"]

class SourceAdapter(Protocol):
    name: str
    def claim(self, value: str) -> SourceClaim: ...
    def observe(
        self, value: str, *, permit: ObservePermit, options: YtDlpSourceOptions
    ) -> SourceSession: ...

def admit_source(
    value: str,
    *,
    permit: ObservePermit,
    options: YtDlpSourceOptions,
    store: SourceStore | None = None,
) -> SourceSession:
    adapters: list[SourceAdapter] = [LocalFileSourceAdapter(), YtDlpSourceAdapter()]
    exact = [adapter for adapter in adapters if adapter.claim(value) == "exact"]
    fallback = [adapter for adapter in adapters if adapter.claim(value) == "fallback"]
    candidates = exact or fallback
    if len(candidates) != 1:
        reason = "ambiguous" if candidates else "unsupported"
        raise UnsupportedSourceError(f"{reason} source: {value}")
    adapter = candidates[0]
    if permit.network == "denied" and store is not None and adapter.name != "local-file":
        cached = store.get(value)
        if cached is not None:
            return cached
    session = adapter.observe(value, permit=permit, options=options)
    if store is None or session.record.metadata.source.kind != "url":
        return session
    try:
        return store.wrap(value, session)
    except CacheError:
        return session
