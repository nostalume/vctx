# vctx

`vctx` prepares clean context packs from video URLs, local media, and transcript files.

It is for people and agents who want source-grounded video context without a chat app, RAG stack, or hidden model workflow.

## Install

From PyPI with uv:

```bash
uv tool install "vctx[full]"
```

`[full]` is the recommended install for normal users; it includes local ASR and visual/OCR extras. Visual installs include PyAV, RapidOCR, and ONNX Runtime. Smaller installs are available when you only need part of the stack:

```bash
uv tool install vctx            # minimal transcript/URL workflows
uv tool install "vctx[asr]"     # minimal + local faster-whisper ASR
uv tool install "vctx[visual]"  # minimal + PyAV + local OCR/visual extras
```

Tier-1 installation coverage is Windows x64, Linux x64, and macOS Apple
Silicon on Python 3.12–3.14. The CI matrix installs the built wheel as core,
ASR, visual, full, and the simultaneous `[asr,visual]` selection on those
environments. Visual frame production is in-process through PyAV; no host media
executable is required.

Then run:

```bash
vctx prepare INPUT... --out DIR
```

Prepare local models explicitly; normal `prepare` never downloads them:

```bash
vctx models pull asr ocr
vctx models status asr ocr
vctx models verify asr ocr --json
```

Inspect or explicitly prune the independent source cache:

```bash
vctx cache status
vctx cache prune --dry-run --age 30d
vctx cache prune --age 30d
```

Plain `cache prune` removes only orphan blobs and temporary files. Use `--all`
to retire every cached source record. Source-cache commands never touch models.

For one-off use without installing the tool globally:

```bash
uvx vctx prepare INPUT... --out DIR
uvx --from "vctx[full]" vctx prepare INPUT... --out DIR
```

## Install for development

```bash
uv sync
```

Optional local media/model extras:

```bash
uv sync --extra asr
uv sync --extra visual
uv sync --extra full
```

## Essential API

### Prepare a context pack

```bash
uv run vctx prepare INPUT... --out DIR
```

Examples:

```bash
uv run vctx prepare ./captions.srt --out ./out/captions
uv run vctx prepare ./lecture.vtt --out ./out/lecture
uv run vctx prepare ./part-1.vtt ./part-2.vtt --out ./out/course
uv run vctx prepare "https://www.ted.com/talks/terry_moore_how_to_tie_your_shoes" --workflow visual --out ./out/ted
```

### Useful options

```text
--workflow default|transcript|visual|full|metadata
--media-quality auto|fast|balanced|high
--asr auto|none|instance:NAME|local:MODEL
--ocr auto|none
--vision auto|none|instance:NAME
--no-retain-media
--config PATH
--offline
--overwrite
--chunk-max-chars INT
--chunk-max-seconds INT
--cache-dir PATH
--verbose
--debug
```

Each URL operation performs one source observation. `metadata` and `prepare`
share config, offline policy, source options, and sanitized source identity.

### Inspect output

Start with:

```text
DIR/manifest.json
DIR/<source-key>/read.md
DIR/<source-key>/context.md
```

Each input owns one independent `DIR/<source-key>/` lane. Required source media
is retained flat inside that lane and indexed under the same source in the root
manifest. Multiple inputs are never summarized together. Use
`--no-retain-media` only when pack size matters more than self-containment.

Preparing into an existing verified pack performs a source-identity upsert. New
sources are added, unchanged revisions are reused, and changed revisions replace
only their lane. `--overwrite` forces requested lanes to rebuild. The complete
pack is verified and swapped as one filesystem generation; unknown or corrupt
output contents are always refused.

`--offline` admits local inputs and verified cached URL observations/subtitles.
A miss or unverified asset fails before `yt-dlp`, network construction, or output
pack creation.

Inspect the resolved product policy without network access:

```bash
uv run vctx doctor --workflow visual --offline --json
```

Core artifacts:

```text
metadata.json
transcript.json
chunks.json
context.md
read.md
```

Optional visual artifacts:

```text
evidence.json         captures with typed OCR/VLM observations
frames/frame-*.png    display-corrected captured frames
```

Optional planning artifact:

```text
evidence-plan.json
```

## Visual workflow

Visual runs use a language-neutral, transcript-anchored evidence plan. They fetch video only when the validated plan requests frames.

```text
canonical transcript windows
  -> validated claims, relations, and frame requests
  -> frame sampling
  -> OCR and/or VLM description when available
  -> durable captures with typed processor outcomes
  -> evidence.json
```

For the free/ZDR OpenRouter auto route, run `vctx auth openrouter login` (or add `--headless`). It provisions the reserved `keyring:openrouter` slot. Named OpenAI-compatible endpoints accept exactly `env:NAME` or `keyring:NAME`; these references only locate secrets and never select an endpoint or request policy. Secrets are never written to config, logs, or artifacts.

### Minimal config selector examples

```toml
[transforms.asr]
use = "instance:local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model = "small"
device = "auto"
compute = "auto"

[evidence]
planner = "auto"  # or "instance:my-planner"
vision = "auto"   # or "instance:my-vlm"
ocr = "auto"

[instances.ai.my-vlm]
base_url = "https://provider.example/v1"
model = "vision-model"
credential = "env:MY_AI_KEY"
```

ASR uses `transforms.asr.use`; evidence capabilities use the terse `[evidence]` selectors. Named providers live under `[instances.asr.*]` and `[instances.ai.*]`.

## What vctx is not

- not an AI chat app
- not a video Q&A system
- not a knowledge base
- not a vector/RAG framework
- not a web backend
- not a hidden paid model caller

## Developer docs

- [`docs/api.md`](docs/api.md) — architecture, CLI, config, and artifacts.
- [`docs/AGENTS.md`](docs/AGENTS.md) — project goal, stack, and principles.

## License

MIT. See [`LICENSE`](LICENSE).
