# Project context

## Product

`vctx` compiles source media/transcripts into inspectable context packs.

```text
input
  -> metadata
  -> transcript / ASR fallback
  -> optional visual evidence
  -> chunks
  -> JSON + Markdown artifacts
  -> manifest
```

The product is the output directory. Downstream agents consume the artifacts.

## Non-goals

Do not build these into `vctx`:

- chat UI
- Q&A system
- RAG/vector database
- personal knowledge base
- cross-video memory
- desktop/web backend
- hidden paid/cloud model calls

## Rules

- CLI first: `vctx prepare INPUT --out DIR`.
- Deterministic source data first.
- Model calls are source-preparation transforms, not assistant behavior.
- Every route, warning, and artifact is visible in `manifest.json`.
- JSON artifacts are canonical; Markdown is projection.
- Keep default install small; use optional extras for heavy tools.
- Provider payloads stop at adapters.
- Internal boundaries use typed models, not opaque dict probing.

## Current artifact families

```text
manifest.json          run audit and artifact index
metadata.json          source metadata
transcript.*           raw/clean/human transcript
chunks.json            chunked source text
knowledge_flow.json    evidence-linked flow graph
visual_records.json    OCR/VLM/capture evidence records
visual_scores.json     visual satisfaction diagnostics
visual/frames/*.png    frame artifacts
context.md             agent context projection
readable.md            human projection
```

## Docs spine

```text
docs/context.md                 this file: intent and rules
docs/architecture.md            durable boundaries
docs/api.md                     CLI/config/artifact contract
docs/graph/README.md            graph doc index and module correspondence
docs/graph/app.md               app/config workflow graph
docs/graph/model-transforms.md  transform API graph
docs/development.md             developer workflow and tests
```

Historical plans stay in `docs/report/` and are not current authority.

## Quality bar

Done means:

- focused tests cover the behavior
- `ruff`, `ty`, and `pytest` pass
- artifacts are readable and machine-readable
- manifest tells the truth
- real side-effect integration is run when touching provider/media behavior
