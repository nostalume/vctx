from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vctx.app.prepare import PreparePipeline, SourcePrepared
from vctx.app.progress import phase
from vctx.app.run import RunRuntimes, open_prepare_run
from vctx.artifact.bundle import write_manifest
from vctx.artifact.manifest import Manifest, ManifestEffect, RunFailure, build_manifest
from vctx.artifact.publish import PackPublisher, VerificationReport, verify_pack
from vctx.config import PrepareRequest, ResolvedConfig, load_resolved_config
from vctx.errors import ConfigError, OperationCancelledError, SourceConflictError, VctxError
from vctx.source.session import Revision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareResult:
    out_dir: Path
    manifest: Manifest
    target: str
    cache_dir: Path
    config_path: Path | None

    def render_cli(self) -> str:
        label = "partial context pack" if self.manifest.status == "partial" else "context pack"
        config = (
            str(self.config_path) if self.config_path is not None else "built-in defaults + CLI"
        )
        lines = [
            f"Wrote {label}: {self.out_dir}",
            f"Manifest: {self.out_dir / 'manifest.json'}",
            f"Target: {self.target}",
            f"Status: {self.manifest.status}",
            f"Output: {self.out_dir}",
            f"Cache: {self.cache_dir}",
            f"Config: {config}",
        ]
        artifacts = [
            f"{source.path}/{artifact.path}"
            for source in self.manifest.sources
            for artifact in source.artifacts
        ]
        routes = [
            _render_route(effect)
            for source in self.manifest.sources
            for effect in source.effects
            if effect.route is not None
        ]
        warnings = [
            omission
            for source in self.manifest.sources
            for outcome in source.outcomes
            for omission in outcome.omissions
        ]
        groups = (("Artifacts", artifacts), ("Routes", routes), ("Warnings", warnings))
        for heading, values in groups:
            if values:
                lines.append(f"{heading}:")
                lines.extend(f"  - {value}" for value in values)
        return "\n".join(lines) + "\n"


def prepare_context_pack(request: PrepareRequest) -> PrepareResult:
    with phase(logger, "prepare.total"):
        resolved = load_resolved_config(request)
        _validate_cache_output(request.out_dir, resolved)
        with PackPublisher(request.out_dir) as publisher:
            previous = publisher.previous
            previous_by_id = {source.id: source for source in previous.sources} if previous else {}
            occupied = {source.key.casefold(): source.id for source in previous_by_id.values()}
            completed: dict[str, Revision | None] = {}
            seen_inputs: set[str] = set()
            results: list[SourcePrepared] = []
            failures: list[RunFailure] = []
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
                        run = open_prepare_run(source_request, resolved, occupied, runtimes)
                        result = PreparePipeline(run).prepare(
                            completed,
                            previous_by_id,
                            publisher,
                        )
                    except OperationCancelledError:
                        raise
                    except SourceConflictError as exc:
                        first_error = first_error or exc
                        disputed = next(
                            (item for item in results if item.source.id == exc.source_id), None
                        )
                        if disputed is not None:
                            results.remove(disputed)
                            failures.append(_run_failure(disputed.input_value, exc))
                        publisher.reset_lane(exc.key)
                        failures.append(_run_failure(value, exc))
                        continue
                    except VctxError as exc:
                        first_error = first_error or exc
                        failures.append(_run_failure(value, exc))
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
                failures=failures,
                requested_target=resolved.target.value,
            )
            write_manifest(publisher.stage, manifest)
            publisher.commit(manifest)
        if first_error is not None:
            raise first_error
        return PrepareResult(
            out_dir=request.out_dir,
            manifest=manifest,
            target=resolved.target.value,
            cache_dir=resolved.cache.source_dir,
            config_path=resolved.config_file.path,
        )


def verify_context_pack(root: Path) -> VerificationReport:
    return verify_pack(root)


def _validate_cache_output(out_dir: Path, resolved: ResolvedConfig) -> None:
    output = out_dir.resolve()
    for cache in (resolved.cache.source_dir.resolve(), resolved.cache.model_dir.resolve()):
        if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
            raise ConfigError(
                f"output and cache paths must not contain each other: {out_dir}, {cache}"
            )


def _run_failure(value: str, error: VctxError) -> RunFailure:
    return RunFailure(
        input_sha256=hashlib.sha256(value.encode()).hexdigest(),
        category=type(error).__name__.removesuffix("Error").lower(),
        diagnostic="source admission failed",
    )


def _render_route(effect: ManifestEffect) -> str:
    parts = [f"{effect.operation}: {effect.route or effect.status}"]
    if effect.provider is not None:
        parts.append(f"provider={effect.provider}")
    if effect.model is not None:
        parts.append(f"model={effect.model}")
    if effect.uploaded:
        parts.append("upload=required")
    if effect.cost_may_apply:
        parts.append("cost=may-apply")
    parts.append(effect.diagnostic or effect.status)
    return "; ".join(parts)


def vctx_version() -> str:
    try:
        return version("vctx")
    except PackageNotFoundError:
        return "0.0.0"
