# vctx

`vctx` compiles video URLs, local video/audio, and SRT/VTT subtitles into a
durable context pack. A pack keeps canonical transcript, evidence, summary, and
provenance data beside readable Markdown so people and AI agents can inspect the
same source-grounded result.

It is a one-shot CLI, not a chat application, RAG database, or background
service. Video frames are decoded in-process with PyAV; no host `ffmpeg`
executable is required.

Subtitle-backed transcript preparation needs no configuration or AI account.
Evidence planning and summaries do require an admitted AI route: authenticate
once with `vctx auth openrouter login`, provide `OPENROUTER_API_KEY`, or configure
your own OpenAI-compatible endpoint. vctx never provides anonymous AI access.

## Installation

Python 3.14 or newer is required. The full profile includes local ASR, frame extraction,
and OCR:

```console
uv tool install "vctx[full]"
```

Smaller installs are available:

```console
uv tool install vctx             # subtitles, URL acquisition, compatible AI
uv tool install "vctx[asr]"      # core + faster-whisper
uv tool install "vctx[visual]"   # core + PyAV + RapidOCR
```

The equivalent pip command is `python -m pip install "vctx[full]"` inside a
Python 3.14 environment.

## Usage

Prepare local model assets once, compile a source, verify the resulting pack,
then render the view needed by a person or agent:

```console
vctx auth openrouter login
vctx models pull asr ocr
vctx prepare ./lecture.mp4 --out ./lecture-pack --to summary --source-assets complete --max-runtime 1800
vctx verify ./lecture-pack
vctx render ./lecture-pack --format read --out ./lecture.md
```

On Windows with an NVIDIA GPU, `vctx[full]` includes acceleration. For an ASR-only
install, use `vctx[asr-cuda]`. vctx selects admitted acceleration automatically and
falls back to CPU before output is emitted; it does not require `PATH` edits.

`prepare` defaults to `--to transcript`. `--to evidence` adds transcript-anchored
frame planning and observations; `--to summary` adds a citation-constrained
summary. The stages are monotonic, so a later target retains all safe earlier
products. Multiple inputs become independent source directories and are never
combined into one summary.

Source files live beside their products inside the output lane. The default
`--source-assets consumed` retains only assets needed by the requested work;
`--source-assets complete` retains every audio, video, or native-subtitle role
reported for the admitted source revision. A later complete request extends the
same verified output, fetching only missing roles while preserving transcript
quality and existing products.

For an agent-oriented view:

```console
vctx render ./lecture-pack --format context
vctx prompt
```

## Simple configuration

Create `vctx.toml` in the working directory:

```toml
[cache]
source_dir = ".cache/vctx/source"
model_dir = ".cache/vctx/models"

[transforms.asr]
quality = "balanced"

[evidence]
planner = "auto"
ocr = "auto"
vision = "auto"

[summary]
use = "auto"
language = "native"

[output]
projections = ["context", "read"]
```

For zero-TOML online planning and summaries, authenticate once with `vctx auth
openrouter login`; `auto` then admits the free zero-data-retention OpenRouter
route. `OPENROUTER_API_KEY` provides the same automatic route without keyring
login. Without either credential, `auto` does not make an AI call and the pack
records unavailable evidence/summary outcomes while retaining safe earlier
products. You may instead configure any suitable OpenAI-compatible `/v1`
endpoint. Secrets stay in an environment variable or system keyring.

Inspect the effective setup without downloading or creating anything:

```console
vctx doctor --to summary --json
```

More runnable configurations are under [docs/examples](docs/examples/README.md). The
complete command behavior, every configuration field, path precedence, pack
layout, and exit status are documented in [docs/api.md](docs/api.md).

## Workflow

```text
INPUT...
  -> admit and acquire each source
  -> transcript -> evidence -> summary
  -> canonical schema-5 JSON + selected Markdown projections
  -> atomic PACK publication
  -> verify PACK
  -> render context | read | transcript
```

The pack is the integration boundary. Begin with `manifest.json`; it records
source identities, revisions, artifacts, product outcomes, provider/model
effects, omissions, upload/cost facts, and integrity digests. Re-running
`prepare` reuses matching verified lanes. Use `--overwrite` only when you intend
to refresh or rebuild them.

## License

MIT License. See [LICENSE](LICENSE).
