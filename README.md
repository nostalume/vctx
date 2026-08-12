# vctx

`vctx` is a CLI-first context compiler. It turns video URLs, local media, and
subtitle files into inspectable context packs for people and downstream agents.
It is not a chat app, RAG system, or hidden model workflow.

## Install

```console
uv tool install "vctx[full]"
```

Smaller profiles are available:

```console
uv tool install vctx
uv tool install "vctx[asr]"
uv tool install "vctx[visual]"
```

Visual processing uses PyAV in-process; no host FFmpeg executable is required.

## First run

```console
vctx models pull asr ocr
vctx prepare ./captions.srt --out ./pack
vctx verify ./pack
vctx render ./pack --format read
```

Prepare stops at transcripts by default. Request later products explicitly:

```console
vctx prepare VIDEO --out ./pack --to evidence
vctx prepare VIDEO --out ./pack --to summary
```

Start with `pack/manifest.json`. Each input has one independent source lane with
canonical JSON, readable projections, retained source material, and any captured
frames. Preparing into a verified pack reuses matching work; `--overwrite`
explicitly rebuilds requested lanes.

## Configuration

Copy [examples/vctx.toml](examples/vctx.toml), or configure a named compatible
AI endpoint with [examples/openai-compatible.toml](examples/openai-compatible.toml).

```console
vctx doctor --json
vctx prompt
```

Config selection is `--config`, then `./vctx.toml`, `VCTX_CONFIG`, the platform
global config, then built-ins. See [docs/api.md](docs/api.md) for the complete CLI,
configuration, path-resolution, cache, and artifact contracts.

## Development

```console
uv sync --extra full
uv run ruff check .
uv run ty check
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
