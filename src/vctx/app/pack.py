from __future__ import annotations

import logging
from pathlib import Path

from vctx.app.prepare import SourcePrepared, prepare_source
from vctx.app.progress import phase
from vctx.app.result import PrepareResult, PrepareSummary
from vctx.app.run import RunRuntimes
from vctx.artifact.manifest import build_manifest
from vctx.artifact.publish import PackPublisher
from vctx.config import PrepareRequest, ResolvedConfig, load_resolved_config
from vctx.errors import ConfigError, OperationCancelledError, VctxError
from vctx.io import write_manifest
from vctx.util import vctx_version

logger = logging.getLogger(__name__)


def prepare_context_pack(request: PrepareRequest) -> PrepareResult:
    with phase(logger, "prepare.total"):
        resolved = load_resolved_config(request)
        _validate_cache_output(request.out_dir, resolved)
        with PackPublisher(request.out_dir) as publisher:
            previous = publisher.previous
            previous_by_id = {source.id: source for source in previous.sources} if previous else {}
            occupied = {source.key.casefold(): source.id for source in previous_by_id.values()}
            completed: set[str] = set()
            seen_inputs: set[str] = set()
            results: list[SourcePrepared] = []
            first_error: VctxError | None = None
            with RunRuntimes() as runtimes:
                for value in request.inputs:
                    if value in seen_inputs:
                        continue
                    seen_inputs.add(value)
                    source_request = request.model_copy(
                        update={"inputs": [value], "out_dir": publisher.stage}
                    )
                    try:
                        result = prepare_source(
                            source_request,
                            resolved,
                            occupied,
                            completed,
                            runtimes,
                            previous_by_id,
                            publisher.reset_lane,
                            publisher.rollback_lane,
                        )
                    except OperationCancelledError:
                        raise
                    except VctxError as exc:
                        first_error = first_error or exc
                        continue
                    if result is None:
                        continue
                    results.append(result)
                    first_error = first_error or result.error
            if not results:
                assert first_error is not None
                raise first_error
            merged = dict(previous_by_id)
            merged.update((result.source.id, result.source) for result in results)
            manifest = build_manifest(
                vctx_version(),
                list(merged.values()),
                incomplete=first_error is not None,
                previous=previous,
            )
            write_manifest(publisher.stage, manifest)
            publisher.commit(manifest)
        if first_error is not None:
            raise first_error
        artifacts = [artifact for result in results for artifact in result.artifacts]
        return PrepareResult(
            out_dir=request.out_dir,
            manifest=manifest,
            artifacts=artifacts,
            summary=PrepareSummary.from_run(
                request=request,
                resolved=resolved,
                manifest=manifest,
                artifacts=artifacts,
            ),
        )


def _validate_cache_output(out_dir: Path, resolved: ResolvedConfig) -> None:
    output = out_dir.resolve()
    for cache in (resolved.cache.source_dir.resolve(), resolved.cache.model_dir.resolve()):
        if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
            raise ConfigError(
                f"output and cache paths must not contain each other: {out_dir}, {cache}"
            )
