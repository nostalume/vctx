from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from platformdirs import user_config_path

from vctx.app.chunk import write_chunk_file
from vctx.app.doctor import doctor_report
from vctx.app.metadata import inspect_metadata, render_metadata_text
from vctx.app.prepare import PrepareRequest, prepare_context_pack
from vctx.app.render import RenderFormat, write_render_file
from vctx.config import WorkflowProfile
from vctx.errors import VctxError
from vctx.io import model_to_json

app = typer.Typer(no_args_is_help=True)


@app.command("prepare")
def prepare_command(
    input: str,
    out: Annotated[Path, typer.Option("--out", help="Output directory for durable artifacts.")],
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Allow writing into non-empty output directory.")
    ] = False,
    chunk_max_chars: Annotated[int, typer.Option("--chunk-max-chars")] = 6000,
    chunk_max_seconds: Annotated[int | None, typer.Option("--chunk-max-seconds")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    keep_temp: Annotated[bool, typer.Option("--keep-temp")] = False,
    workflow: Annotated[
        WorkflowProfile,
        typer.Option(
            "--workflow",
            help="Preparation workflow instance: default, transcript, visual, full, or metadata.",
        ),
    ] = WorkflowProfile.DEFAULT,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Use the offline workflow policy; network routes unavailable.",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Optional TOML config file for workflow defaults."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Emit INFO progress logs to stderr."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Emit DEBUG progress logs to stderr."),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Write prepare logs to this file."),
    ] = None,
) -> None:
    _configure_logging(verbose=verbose, debug=debug, log_file=log_file)
    try:
        result = prepare_context_pack(
            PrepareRequest(
                input=input,
                out_dir=out,
                overwrite=overwrite,
                chunk_max_chars=chunk_max_chars,
                chunk_max_seconds=chunk_max_seconds,
                cache_dir=cache_dir,
                keep_temp=keep_temp,
                workflow=workflow,
                offline=offline,
                config_path=_select_config_path(config),
            )
        )
    except VctxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc

    if result.manifest.status == "partial":
        typer.echo(f"Wrote partial context pack: {result.out_dir}")
    else:
        typer.echo(f"Wrote context pack: {result.out_dir}")
    typer.echo(f"Manifest: {result.out_dir / 'manifest.json'}")
    artifact_paths = {artifact.path for artifact in result.artifacts}
    if "metadata.json" in artifact_paths:
        typer.echo(f"Metadata: {result.out_dir / 'metadata.json'}")
    if "context.md" in artifact_paths:
        typer.echo(f"Context: {result.out_dir / 'context.md'}")
    if "readable.md" in artifact_paths:
        typer.echo(f"Readable: {result.out_dir / 'readable.md'}")
    for line in result.summary.render_cli_lines():
        typer.echo(line)


@app.command("metadata")
def metadata_command(
    input: str,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print normalized VideoMetadata JSON."),
    ] = False,
) -> None:
    try:
        metadata = inspect_metadata(input)
    except VctxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc

    if json_output:
        typer.echo(model_to_json(metadata), nl=False)
    else:
        typer.echo(render_metadata_text(metadata), nl=False)


@app.command("chunk")
def chunk_command(
    transcript: Path,
    out: Annotated[Path, typer.Option("--out", help="Output chunks JSON file.")],
    chunk_max_chars: Annotated[int, typer.Option("--chunk-max-chars")] = 6000,
    chunk_max_seconds: Annotated[int | None, typer.Option("--chunk-max-seconds")] = None,
) -> None:
    try:
        out_path = write_chunk_file(
            transcript,
            out,
            max_chars=chunk_max_chars,
            max_seconds=chunk_max_seconds,
        )
    except VctxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc

    typer.echo(f"Wrote chunks: {out_path}")


@app.command("render")
def render_command(
    metadata: Annotated[Path, typer.Option("--metadata", help="Input metadata JSON file.")],
    transcript: Annotated[
        Path,
        typer.Option("--transcript", help="Input transcript JSON file."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output Markdown file.")],
    format: Annotated[RenderFormat, typer.Option("--format", help="Render format.")],
    chunks: Annotated[Path | None, typer.Option("--chunks", help="Input chunks JSON file.")] = None,
) -> None:
    try:
        out_path = write_render_file(
            metadata_path=metadata,
            transcript_path=transcript,
            chunks_path=chunks,
            out_path=out,
            format=format,
        )
    except VctxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc

    typer.echo(f"Wrote render: {out_path}")


@app.command("doctor")
def doctor_command() -> None:
    typer.echo(doctor_report(), nl=False)


def main() -> None:
    app()


def _select_config_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    cwd = Path.cwd()
    for candidate in (cwd / "vctx.toml", cwd / ".vctx.toml"):
        if candidate.exists():
            return candidate
    env_config = os.environ.get("VCTX_CONFIG")
    if env_config:
        return Path(env_config)
    global_config = _global_config_path()
    return global_config if global_config.exists() else None


def _global_config_path() -> Path:
    return user_config_path("vctx", appauthor=False) / "config.toml"


def _configure_logging(*, verbose: bool, debug: bool, log_file: Path | None) -> None:
    logger = logging.getLogger("vctx")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if not verbose and not debug and log_file is None:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        return

    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    if verbose or debug:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False
