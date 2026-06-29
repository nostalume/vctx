# Models Module Graph

## Purpose

Define stable internal data shapes and artifact schemas.

Models are data contracts only. They validate and serialize owned payloads; they do not perform side effects, provider calls, routing, rendering, network access, or source acquisition.

## Dependencies

```text
models -> pydantic -> stdlib types
```

Models are leaves.

Forbidden:

```text
models -> app
models -> cli
models -> sources
models -> transforms
models -> render
models -> io
models -> net
```

## Current API graph

```text
vctx.models
  SourceRef                     # package-level shared source reference

vctx.models.metadata
  VideoMetadata

vctx.models.media
  MediaAsset

vctx.models.artifacts
  ArtifactKind
  Artifact
  ArtifactBundle

vctx.models.knowledge_flow
  KnowledgeFlowNodeKind
  KnowledgeFlowEdgeKind
  KnowledgeFlowNode
  KnowledgeFlowEdge
  KnowledgeFlow
  KnowledgeFlowSupplementEvidence
  KnowledgeFlowSupplement

vctx.models.visual
  EvidenceKind
  Evidence
  FrameAsset
  VisualUncertainty
  VisualRecord
  VisualRecordSet
  VisualEvidenceScore
  EssentialCaseType
  EssentialCaseAction
  EssentialVisualCase
  EssentialCaseSupplementEvidence
  EssentialCaseSupplement

vctx.models.manifest
  CapabilityName
  SelectedRoute
  TransformEvidence
  ArtifactRef
  ManifestStep
  Manifest
  ManifestBuilder
```

## Reduced / deleted model satellites

These paths are intentionally deleted and guarded by `scripts/check_module_layout.py`:

```text
src/vctx/models/llm_products.py   # vague product bucket; moved to domain model modules
src/vctx/models/transcript.py     # one-owner contracts moved to transcript.py
src/vctx/models/chunks.py         # one-owner contracts moved to chunking.py
src/vctx/models/common.py         # SourceRef moved to models/__init__.py
```

## Ownership rules

### Package-level shared API

`SourceRef` lives in `vctx.models` because it is shared by metadata/media/source adapters and has no narrower owner.

```text
from vctx.models import SourceRef
```

Do not recreate `models/common.py` for one shared class.

### Domain modules

Domain model modules own contracts that are reused across app/transforms/render/io.

Examples:

```text
knowledge_flow.py owns KnowledgeFlow and KnowledgeFlowSupplement
visual.py owns visual records, visual evidence, essential cases, and visual supplement contracts
manifest.py owns manifest/evidence route contracts
```

### One-owner contracts do not belong in models/

If a contract is only owned by a single flat processing module, place it with that owner:

```text
transcript.py owns Transcript, TranscriptSegment, TranscriptPayload
chunking.py owns ChunkOptions, TranscriptChunk, ChunkSet
```

## Atomic isolation

Models validate and serialize data only. They contain no side effects and no provider calls.

## Verification

- JSON serialization is stable enough for artifacts.
- Provider-specific raw payloads are not normal fields.
- Timestamps and provenance survive transformations.
- Deleted satellite modules remain absent via `scripts/check_module_layout.py`.
