# AI / Model Route API Graph

## Purpose

Provide typed model-mediated route metadata for transform workflows.

The AI/model route layer is not the transform layer and not the network runtime layer.

```text
transform layer
  owns product workflow semantics:
    ASR transcript fallback
    visual evidence acquisition
    essential visual case planning
    knowledge-flow extraction

AI/model route layer
  owns model/provider dispatch metadata:
    task identity
    model reference parsing
    route availability
    credential/upload/cost metadata
    OpenRouter/free/configured/local route selection

network runtime layer
  owns API-call mechanics:
    request/response transport
    sessions/connections
    timeout/retry/backoff
    provider/task concurrency limits
```

Transforms ask for behavior. The AI/model route layer resolves typed route metadata. Provider adapters delegate HTTP mechanics to the injected network runtime.

## Dependency direction

Allowed:

```text
transforms.ai -> transforms.model_resolution
transforms.model_resolution -> net
provider adapters -> net runtime protocol
app -> transforms.ai/model_resolution through config-driven route construction
```

Forbidden:

```text
AiRoute -> session/client object
ai/model route layer -> render
ai/model route layer -> chunking
ai/model route layer -> sources
ai/model route layer -> final artifact writer
net -> ai/transforms/app/sources
models -> ai/transforms/net
```

## Current module layout

```text
src/vctx/transforms/ai_routes.py
  type AiTaskKind
  type AiSelectedRoute
  AiRoute
  AiResult[T]
  ai_route_from_model_route(...)
  model_capability_for_ai_task(...)

src/vctx/transforms/model_resolution.py
  ModelCapability
  ModelRef
  ResolvedModelRoute
  OpenRouterModelArchitecture
  OpenRouterModelPricing
  OpenRouterModel
  OpenRouterRegistryPayload
  parse_model_ref(...)
  resolve_model_ref(...)
  choose_openrouter_model(...)
  load_openrouter_models(...)
```

Deleted/reduced:

```text
src/vctx/transforms/openrouter_registry.py
```

OpenRouter registry loading now lives in `model_resolution.py` because it is owned by model resolution and only one provider registry exists.

## Public contracts

### `AiTaskKind`

Current task names are behavior names, not provider names:

```python
type AiTaskKind = Literal[
    "asr_transcription",
    "ocr_text",
    "vision_description",
    "essential_case_extraction",
    "knowledge_flow_extraction",
]
```

Deleted route ghosts:

```text
transcript_cleanup
chapter_suggestion
```

Cleanup and chapters are rejected product slices; they are no longer active AI task names.

### `AiRoute`

`AiRoute` is serializable route metadata:

```python
class AiRoute(BaseModel):
    task: AiTaskKind
    selected: AiSelectedRoute
    provider: str
    provider_id: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    cost: str = "unknown"
    upload: str = "none"
    available: bool = False
    reason: str
    warnings: list[str] = Field(default_factory=list)
```

It must not hold:

```text
HTTP clients
sessions
semaphores
runtime objects
secret values
```

### `AiResult[T]`

Python 3.12 generic result wrapper:

```python
class AiResult[T](BaseModel):
    value: T
    route: AiRoute
    warnings: list[str] = Field(default_factory=list)
```

## Model ref resolution

Model refs use prefix-style routing:

```text
none
local:<path-or-id>
openrouter:<model-id>
auto
configured provider refs through policy/provider config
```

Resolution produces:

```text
ResolvedModelRoute
  ref
  provider
  model
  base_url
  api_key_env
  cost
  upload
  available
  reason
  warnings
```

## Implemented consumers

```text
ASR route planning/execution
visual OCR/VLM route discovery
text LLM knowledge-flow supplement
text LLM essential-case supplement
OpenRouter registry model selection
```

## Reduction plan

Priority 1 is complete:

```text
cleanup/chapter task/model capability names were removed from active APIs
```

Priority 2:

```text
keep AiRoute as metadata while reducing visual planning action structures separately
```

Rejected:

```text
Do not recreate transforms/ai/ package tree.
Do not make AiRoute own NetRuntime or HTTP clients.
Do not split openrouter_registry.py back out until another provider registry exists.
```
