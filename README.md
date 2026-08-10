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
Silicon on Python 3.12–3.14. The CI matrix verifies each published profile on
those environments. PyAV is packaged now; the visual runtime’s final migration
away from its current host-ffmpeg frame adapter is tracked separately.

Then run:

```bash
vctx prepare INPUT --out DIR
```

Prepare local models explicitly; normal `prepare` never downloads them:

```bash
vctx models pull asr ocr
vctx models status asr ocr
vctx models verify asr ocr --json
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
--asr auto|none|instance:NAME|local:MODEL
--ocr auto|none
--vision auto|none|instance:NAME|openrouter:MODEL
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

### Inspect output

Start with:

```text
DIR/manifest.json
DIR/readable.md
DIR/context.md
```

Required source media is copied into `DIR/media/` and indexed in the manifest
by default, including local inputs. Use `--no-retain-media` only when pack size
matters more than self-containment.

`--offline` admits local inputs and verified local assets. URL-source caching is
not implemented yet, so an offline URL fails before `yt-dlp` runs or an output
pack is created.

Inspect the resolved product policy without network access:

```bash
uv run vctx doctor --workflow visual --offline --json
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

- [`docs/api.md`](docs/api.md) — architecture, CLI, config, and artifacts.
- [`docs/AGENTS.md`](docs/AGENTS.md) — project goal, stack, and principles.

## License

MIT. See [`LICENSE`](LICENSE).
