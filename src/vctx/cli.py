from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vctx.errors import VctxError
from vctx.options import MediaQuality, PrepareTarget

if TYPE_CHECKING:
    from vctx.app.auth import OpenRouterAuth
    from vctx.net import NetRuntime


class RenderFormat(StrEnum):
    CONTEXT = "context"
    READ = "read"
    TRANSCRIPT = "transcript"


app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
models_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
cache_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
auth_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
openrouter_auth_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
_AGENT_PROMPT = """# vctx agent context
- Prepare finite media or transcript sources into one durable pack.
- Read `manifest.json` first; it indexes independent source lanes and observed outcomes.
- For multiple sources, select a manifest source key when rendering one result.
- `context.md` is compact agent input; `read.md` is the readable report.
- Render canonical pack products, then verify the complete pack before trusting it.
- Treat listed artifacts as immutable; rerun prepare to publish a new generation.
- Reuse is automatic; request overwrite only to refresh or rebuild admitted work.
- Use `vctx doctor --json` to discover config, cache, and capability readiness.
- Use command-specific `--help` for exact syntax.
"""
app.add_typer(models_app, name="models")
app.add_typer(cache_app, name="cache")
app.add_typer(auth_app, name="auth")
auth_app.add_typer(openrouter_auth_app, name="openrouter")


def _openrouter_auth(net: NetRuntime | None = None) -> OpenRouterAuth:
    from vctx.app.auth import AuthError, OpenRouterAuth, system_keyring

    try:
        return OpenRouterAuth(keyring=system_keyring(), net=net)
    except AuthError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


@openrouter_auth_app.command("login")
def openrouter_login_command(
    headless: Annotated[bool, typer.Option("--headless")] = False,
) -> None:
    from vctx.app.auth import AuthError, desktop_login
    from vctx.net import HttpxNetRuntime

    with HttpxNetRuntime() as net:
        auth = _openrouter_auth(net)
        try:
            if headless:
                session = auth.begin("http://localhost")
                typer.echo(f"Open this URL:\n{session.authorization_url}")
                auth.finish(session, code=typer.prompt("Paste authorization code").strip())
            else:
                desktop_login(auth)
        except AuthError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
    typer.echo("OpenRouter authentication stored in the system keyring.")


@openrouter_auth_app.command("status")
def openrouter_status_command() -> None:
    status = _openrouter_auth().status()
    typer.echo("authenticated" if status.authenticated else "not authenticated")


@openrouter_auth_app.command("logout")
def openrouter_logout_command() -> None:
    _openrouter_auth().logout()
    typer.echo("OpenRouter authentication removed from the system keyring.")


@cache_app.command("status")
def cache_status_command(
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    report = _call(lambda: _cache(cache_dir, config).status())
    typer.echo(_render_cache(report, json_output), nl=False)


@cache_app.command("prune")
def cache_prune_command(
    age: Annotated[str | None, typer.Option("--age", help="Age such as 30d, 12h, or 4w.")] = None,
    all_records: Annotated[bool, typer.Option("--all", help="Prune every source record.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    report = _call(
        lambda: _cache(cache_dir, config).prune(age=age, all_records=all_records, dry_run=dry_run)
    )
    typer.echo(_render_cache(report, json_output), nl=False)


@models_app.command("pull")
def models_pull_command(
    capabilities: Annotated[list[str] | None, typer.Argument(help="asr/ocr; default: both")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    asr: Annotated[str | None, typer.Option("--asr")] = None,
    conservative: Annotated[
        bool, typer.Option("--conservative", help="Disable high-performance Xet mode.")
    ] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Revalidate and download again.")
    ] = False,
    max_runtime: Annotated[
        int, typer.Option("--max-runtime", min=1, max=86400, help="Hub child limit in seconds.")
    ] = 3600,
) -> None:
    from vctx.model_store import pull_models

    cache_root, asr_model_id = _models(cache_dir, config, asr)
    receipts = _call(
        lambda: pull_models(
            capabilities,
            cache_dir=cache_root,
            asr_model_id=asr_model_id,
            conservative=conservative,
            refresh=refresh,
            max_runtime=max_runtime,
        )
    )
    typer.echo(_render_models(receipts, json_output), nl=False)


@models_app.command("status")
def models_status_command(
    capabilities: Annotated[list[str] | None, typer.Argument(help="asr/ocr; default: both")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    asr: Annotated[str | None, typer.Option("--asr")] = None,
) -> None:
    from vctx.model_store import model_status

    cache_root, asr_model_id = _models(cache_dir, config, asr)
    receipts = _call(
        lambda: model_status(capabilities, cache_dir=cache_root, asr_model_id=asr_model_id)
    )
    typer.echo(_render_models(receipts, json_output), nl=False)


@models_app.command("verify")
def models_verify_command(
    capabilities: Annotated[list[str] | None, typer.Argument(help="asr/ocr; default: both")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    asr: Annotated[str | None, typer.Option("--asr")] = None,
) -> None:
    from vctx.model_store import verify_models

    cache_root, asr_model_id = _models(cache_dir, config, asr)
    receipts = _call(
        lambda: verify_models(capabilities, cache_dir=cache_root, asr_model_id=asr_model_id)
    )
    typer.echo(_render_models(receipts, json_output), nl=False)


@models_app.command("prune")
def models_prune_command(
    incomplete: Annotated[
        bool, typer.Option("--incomplete", help="Remove recoverable incomplete pulls.")
    ] = False,
    unreferenced: Annotated[
        bool, typer.Option("--unreferenced", help="Remove generations with no receipt.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    asr: Annotated[str | None, typer.Option("--asr")] = None,
) -> None:
    from vctx.model_store import prune_model_cache

    if not incomplete and not unreferenced:
        typer.echo("error: choose --incomplete and/or --unreferenced", err=True)
        raise typer.Exit(2)
    cache_root, _asr_model_id = _models(cache_dir, config, asr)
    report = _call(
        lambda: prune_model_cache(
            cache_root,
            incomplete=incomplete,
            unreferenced=unreferenced,
            dry_run=dry_run,
        )
    )
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    action = "would prune" if dry_run else "pruned"
    typer.echo(f"{action} {report.count} model path(s), {report.bytes} bytes")


@app.command("prepare")
def prepare_command(
    inputs: list[str],
    out: Annotated[Path, typer.Option("--out", help="Output directory for durable artifacts.")],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    chunk_max_chars: Annotated[int | None, typer.Option("--chunk-max-chars")] = None,
    chunk_max_seconds: Annotated[int | None, typer.Option("--chunk-max-seconds")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    media_quality: Annotated[MediaQuality | None, typer.Option("--media-quality")] = None,
    target: Annotated[PrepareTarget, typer.Option("--to")] = PrepareTarget.TRANSCRIPT,
    asr: Annotated[str | None, typer.Option("--asr", help="ASR selector.")] = None,
    ocr: Annotated[str | None, typer.Option("--ocr")] = None,
    vision: Annotated[str | None, typer.Option("--vision", help="Vision selector.")] = None,
    no_retain_media: Annotated[bool, typer.Option("--no-retain-media")] = False,
    offline: Annotated[bool | None, typer.Option("--offline", help="Deny network routes.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="TOML config file.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="INFO logs to stderr.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="DEBUG logs to stderr.")] = False,
    log_file: Annotated[Path | None, typer.Option("--log-file", help="Write logs to file.")] = None,
    profile_json: Annotated[
        Path | None, typer.Option("--profile-json", help="Write JSONL phase events.")
    ] = None,
    max_runtime: Annotated[
        int | None,
        typer.Option("--max-runtime", min=1, max=86400, help="Hard wall-clock limit in seconds."),
    ] = None,
    start: Annotated[
        float | None, typer.Option("--start", min=0, help="ASR start in seconds.")
    ] = None,
    end: Annotated[float | None, typer.Option("--end", min=0, help="ASR end in seconds.")] = None,
) -> None:
    from vctx.app.progress import configure_logging
    from vctx.config import PrepareRequest

    request = PrepareRequest(
        inputs=inputs,
        out_dir=out,
        overwrite=overwrite,
        chunk_max_chars=chunk_max_chars,
        chunk_max_seconds=chunk_max_seconds,
        cache_dir=cache_dir,
        media_quality=media_quality,
        target=target,
        asr_use=asr,
        ocr_use=ocr,
        vision_use=vision,
        retain_media=False if no_retain_media else None,
        offline=offline,
        config_path=config,
        start_seconds=start,
        end_seconds=end,
    )
    if max_runtime is not None:
        from vctx.prepare_worker import supervise_prepare

        outcome = supervise_prepare(
            request,
            timeout_s=max_runtime,
            verbose=verbose,
            debug=debug,
            log_file=log_file,
            profile_json=profile_json,
            stderr_sink=lambda block: typer.echo(
                block.decode(errors="replace"), err=True, nl=False
            ),
        )
        if outcome.returncode == 124:
            typer.echo("error: deadline_exceeded: prepare exceeded --max-runtime", err=True)
        if outcome.returncode:
            raise typer.Exit(outcome.returncode)
        typer.echo(outcome.stdout.decode(errors="replace"), nl=False)
        return
    from vctx.app.pack import prepare_context_pack

    configure_logging(
        verbose=verbose,
        debug=debug,
        log_file=log_file,
        profile_json=profile_json,
    )
    result = _call(lambda: prepare_context_pack(request))
    typer.echo(result.render_cli(), nl=False)


@app.command("metadata")
def metadata_command(
    input: str,
    json_output: Annotated[bool, typer.Option("--json", help="Print metadata JSON.")] = False,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    offline: Annotated[bool | None, typer.Option("--offline")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    from vctx.app.metadata import inspect_metadata, render_metadata_text
    from vctx.artifact.bundle import encode_json

    metadata = _call(
        lambda: inspect_metadata(
            input,
            config_path=config,
            cache_dir=cache_dir,
            offline=offline,
        )
    )

    if json_output:
        typer.echo(encode_json(metadata), nl=False)
    else:
        typer.echo(render_metadata_text(metadata), nl=False)


@app.command("render")
def render_command(
    pack: Annotated[Path, typer.Argument(help="Schema-3 context pack.")],
    format: Annotated[RenderFormat, typer.Option("--format", help="Render format.")],
    source: Annotated[str | None, typer.Option("--source", help="Source key.")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="External output file.")] = None,
) -> None:
    from vctx.app.render import render_pack

    result = _call(
        lambda: render_pack(
            pack,
            source_key=source,
            format=format.value,
            out=out,
        )
    )

    if result.path is None:
        assert result.content is not None
        typer.echo(result.content, nl=False)
    else:
        typer.echo(f"Wrote render: {result.path}")


@app.command("verify")
def verify_command(pack: Annotated[Path, typer.Argument(help="Schema-3 context pack.")]) -> None:
    from vctx.app.pack import verify_context_pack

    report = _call(lambda: verify_context_pack(pack))
    typer.echo(
        f"verified schema {report.manifest.schema_version} pack: "
        f"{len(report.manifest.sources)} source(s)"
    )


@app.command("doctor")
def doctor_command(
    target: Annotated[PrepareTarget, typer.Option("--to")] = PrepareTarget.TRANSCRIPT,
    asr: Annotated[str | None, typer.Option("--asr")] = None,
    ocr: Annotated[str | None, typer.Option("--ocr")] = None,
    vision: Annotated[str | None, typer.Option("--vision")] = None,
    offline: Annotated[bool | None, typer.Option("--offline")] = None,
    no_retain_media: Annotated[bool, typer.Option("--no-retain-media")] = False,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    from vctx.app.doctor import doctor_report

    typer.echo(
        doctor_report(
            config_path=config,
            cache_dir=cache_dir,
            target=target,
            asr=asr,
            ocr=ocr,
            vision=vision,
            offline=offline,
            retain_media=False if no_retain_media else None,
            json_output=json_output,
        ),
        nl=False,
    )


@app.command("prompt")
def prompt_command() -> None:
    typer.echo(_AGENT_PROMPT, nl=False)


def main() -> None:
    app()


def _resolved(cache_dir: Path | None, config: Path | None, *, asr: str | None = None):
    from vctx.config import PrepareRequest, load_resolved_config

    return load_resolved_config(
        PrepareRequest(
            inputs=["operation"],
            out_dir=Path("."),
            cache_dir=cache_dir,
            config_path=config,
            asr_use=asr,
        )
    )


def _cache(cache_dir: Path | None, config: Path | None):
    from vctx.app.cache import Cache

    return Cache.open(_resolved(cache_dir, config))


def _models(cache_dir: Path | None, config: Path | None, asr: str | None):
    from vctx.app.models import select_asr_model_id

    resolved = _resolved(cache_dir, config, asr=asr)
    return resolved.cache.model_dir, select_asr_model_id(resolved)


def _render_models(receipts: Any, json_output: bool) -> str:
    return import_module("vctx.app.models").render_model_receipts(receipts, json_output=json_output)


def _render_cache(report: Any, json_output: bool) -> str:
    return import_module("vctx.app.cache").render_cache(report, json_output=json_output)


def _call[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except VctxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc
