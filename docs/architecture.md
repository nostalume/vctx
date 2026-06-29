# Architecture

`vctx` is a one-shot context compiler.

```text
CLI request
  -> app workflow
  -> source adapters
  -> normalized transcript/media models
  -> bounded transforms
  -> render bundle
  -> artifacts + manifest
```

## Boundaries

| Layer | Owns | Must not own |
| --- | --- | --- |
| `cli` | flags, request construction, summary printing | provider calls, workflow policy |
| `app` | workflow order, config resolution, manifest steps | provider payload shapes, Markdown rendering internals |
| `sources` | local/URL metadata, subtitles, media fetch | chunking, rendering, transforms |
| `transcript` / `subtitles` / `chunking` | deterministic text normalization and chunks | provider calls, file output policy |
| `transforms` | ASR/OCR/VLM/text-model product transforms | final rendering, source acquisition |
| `render` | Markdown/JSON projections from typed products | source/model/network access |
| `models` | artifact/domain schemas | app, cli, providers |
| `net` | HTTP transport runtime | product semantics |

## Dependency direction

```text
cli -> app -> sources/transforms/render/io -> models
                 transforms/provider leaves -> net
```

Forbidden:

```text
models -> app/cli/render/transforms/sources
render -> sources/transforms/net
sources -> render/transforms
pure transforms -> provider clients/network
AiRoute -> HTTP session/client/semaphore
```

## AI and network split

```text
transform layer      decides product need: ASR, OCR, VLM, text supplement
AI route layer       resolves serializable route metadata
network runtime      executes HTTP/API transport
provider adapter     normalizes response into typed internal product
```

`AiRoute` is metadata. It is not a runtime manager.

## Visual contract

Visual evidence is motive-led, not score-led.

```text
transcript
  -> deterministic visual cases
  -> bounded uncertain-segment LLM supplement when configured
  -> VisualOperationMotive values
  -> executable VisualAction recipe
  -> frame/OCR/VLM/capture records
  -> score + satisfaction diagnostics
```

Rules:

- No transcript motive, no visual fetch.
- Capture is source evidence.
- OCR is for visible text/formula/table/code.
- VLM is for environment/action/explanation.
- `visual_records.json` is evidence only.
- `visual_scores.json` is diagnostics only.
- Missed satisfaction becomes a manifest warning.

## Artifact boundary

The output directory is the stable integration surface. Python internals may move; these contracts should not drift casually:

```text
manifest.json
metadata.json
transcript.raw.json
transcript.clean.json
chunks.json
knowledge_flow.json
visual_records.json
visual_scores.json
context.md
readable.md
transcript.md
visual/frames/*.png
```

## Current graph authority

Implementation-facing details live in `docs/graph/`:

- `graph/README.md` — module family index.
- `graph/app.md` — app/config workflow graph.
- `graph/model-transforms.md` — transform APIs and visual pipeline.
- `graph/ai.md` — AI route metadata.
- `graph/net.md` — network runtime.
- `graph/models.md` — schema ownership.
