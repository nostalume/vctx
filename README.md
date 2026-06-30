# vctx

`vctx` prepares clean context packs from video URLs, local media, and transcript files.

It is for people and agents who want source-grounded video context without a chat app, RAG stack, or hidden model workflow.

## Install

From PyPI with uv:

```bash
uv tool install "vctx[full]"
```

`[full]` is the recommended install for normal users; it includes local ASR and visual/OCR extras. Smaller installs are available when you only need part of the stack:

```bash
uv tool install vctx            # minimal transcript/URL workflows
uv tool install "vctx[asr]"     # minimal + local faster-whisper ASR
uv tool install "vctx[visual]"  # minimal + local OCR/visual extras
```

Then run:

```bash
vctx prepare INPUT --out DIR
```

For one-off use without installing the tool globally:

```bash
uvx vctx prepare INPUT --out DIR
uvx --from "vctx[full]" vctx prepare INPUT --out DIR
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
uv run vctx prepare INPUT --out DIR
```

Examples:

```bash
uv run vctx prepare ./captions.srt --out ./out/captions
uv run vctx prepare ./lecture.vtt --out ./out/lecture
uv run vctx prepare "https://www.ted.com/talks/terry_moore_how_to_tie_your_shoes" --workflow visual --out ./out/ted
```

### Useful options

```text
--workflow default|transcript|visual|full|metadata
--config PATH
--offline
--overwrite
--chunk-max-chars INT
--chunk-max-seconds INT
--cache-dir PATH
--verbose
--debug
```

### Inspect output

Start with:

```text
DIR/manifest.json
DIR/readable.md
DIR/context.md
```

Core artifacts:

```text
metadata.json
transcript.raw.json
transcript.clean.json
chunks.json
transcript.md
```

Optional visual artifacts:

```text
visual_records.json   OCR/VLM/capture evidence
visual_scores.json    visual satisfaction diagnostics
visual/frames/*.png   captured frames
```

Optional flow artifact:

```text
knowledge_flow.json
```

## Visual workflow

Visual runs use transcript-anchored motives. They fetch video only when useful visual evidence is planned.

```text
transcript cues
  -> visual motives
  -> frame sampling
  -> OCR and/or VLM description when available
  -> capture records
  -> visual_records.json + visual_scores.json
```

Use `OPENROUTER_API_KEY` only when selecting OpenRouter-backed VLM/text routes. Secrets are read from environment or configured `.env` files and are not written to artifacts.

### Minimal config selector examples

```toml
[transforms.asr]
use = "instance:local-default"

[instances.asr.local-default]
type = "local-faster-whisper"
model_policy = "auto"

[transforms.visual_context]
use = "auto"  # or "instance:my-vlm" / "openrouter:<model-id>"
```

Transform config uses one selector field, `use`. Do not combine old-style `route`, `instance`, and `model` fields; named providers live under `[instances.asr.*]` and `[instances.vision.*]`.

## What vctx is not

- not an AI chat app
- not a video Q&A system
- not a knowledge base
- not a vector/RAG framework
- not a web backend
- not a hidden paid model caller

## Developer docs

- [`docs/api.md`](docs/api.md) — CLI/config/artifacts.
- [`docs/architecture.md`](docs/architecture.md) — boundaries.
- [`docs/graph/README.md`](docs/graph/README.md) — module/API graphs.
- [`docs/development.md`](docs/development.md) — develop/test/integration workflow.

## License

MIT. See [`LICENSE`](LICENSE).
