# Network Runtime API Graph

## Purpose

Normalize API-call workflow behind an injected network runtime.

The network runtime is not the AI typed layer and not the transform layer.

```text
transform layer
  owns product workflow semantics

AI typed layer
  owns task/route/result contracts and provider dispatch policy

network runtime layer
  owns HTTP/session/concurrency/retry/timeout execution
```

The goal is to replace scattered synchronous `urlopen(...)` leaves with one injectable runtime boundary that can start synchronous and later gain async/multi-connection execution without changing transform product APIs.

## Current network call shape

Direct HTTP transport is centralized in:

```text
src/vctx/net.py
```

Source/model adapters build `NetRequest` values and call an injected `NetRuntime`.
No transform/source leaf should call `urlopen(...)` directly.

## Dependency direction

Target dependency tree:

```text
app
  -> net
      -> stdlib/http client implementation
  -> transforms
      -> ai
          -> net runtime protocol/value inputs
      -> transform product modules
  -> sources
      -> net runtime protocol/value inputs
```

Allowed:

```text
app -> net
app -> transforms
app -> sources
transforms.ai -> net protocol/types
transforms provider adapters -> net protocol/types
sources provider adapters -> net protocol/types
net -> stdlib/httpx/aiohttp implementation, later
net -> errors/models only if needed for typed errors
```

Forbidden:

```text
net -> app
net -> transforms
net -> sources
net -> render/io/chunking
AiRoute -> session/client object
models -> net
render -> net
```

`net` is a runtime/service boundary. It must not know whether a request is ASR, VLM, subtitle fetch, or OpenRouter registry beyond typed request metadata.

## Layer separation

### Transform layer

Transforms ask for products:

```text
missing transcript -> ASR transcript
selected frame -> visual description
URL subtitle playlist -> VTT text
```

Transforms should not own HTTP session lifecycle, retry policy, or connection pooling.

### AI typed layer

AI owns route/task identity:

```text
AiRoute(task="vision_description", provider="openrouter", model="...")
AiRoute(task="asr_transcription", provider="alias", provider_id="online-test")
AiResult[T](value, route, warnings)
```

AI should not own long-lived HTTP sessions directly. It passes request intent to a network runtime.

### Network runtime layer

Network runtime owns API-call mechanics:

```text
timeouts
connection reuse
sync/async backend
retry/backoff
per-host/per-provider concurrency
request/response byte transport
secret-safe error text
```

It returns typed low-level responses, not domain products.

## Public contracts target

### NetRequest

Planned typed request object:

```python
class NetRequest(BaseModel):
    method: Literal["GET", "POST"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes | None = None
    timeout_s: float
    purpose: NetPurpose
    provider_id: str | None = None
```

`headers` may contain an Authorization header at adapter runtime, but tests and manifest must never print secrets.

### NetResponse

```python
class NetResponse(BaseModel):
    url: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes
```

Adapters parse `body` into provider/domain responses. `NetResponse` does not expose ASR/VLM/OpenRouter-specific schemas.

### NetRuntime

Initial sync protocol:

```python
class NetRuntime(Protocol):
    def request(self, request: NetRequest) -> NetResponse: ...
```

Later async protocol:

```python
class AsyncNetRuntime(Protocol):
    async def request(self, request: NetRequest) -> NetResponse: ...
    async def request_many(self, requests: Sequence[NetRequest]) -> list[NetResponse]: ...
```

Do not make `AiRoute` implement `request` or hold sessions.

### NetRuntimeConfig

```python
class NetRuntimeConfig(BaseModel):
    timeout_s: float = 120
    max_connections: int = 8
    per_provider_concurrency: int = 2
    retry_count: int = 1
    backoff_s: float = 0.5
```

This belongs to runtime/app configuration later, not to `AiRoute`.

## Normalized API-call workflow

All online provider calls should follow one shape:

```text
domain adapter
  -> build NetRequest from typed input + route/provider config
  -> net.request(...)
  -> parse NetResponse into provider response model
  -> normalize provider response into internal product
```

Examples:

```text
OpenAiCompatibleVisionAdapter.describe(frame)
  -> NetRequest(POST, base_url, image payload, timeout, purpose="vision_description")
  -> NetResponse
  -> VisualRecord text

OpenAiCompatibleAsrAdapter.transcribe(media)
  -> NetRequest(POST, base_url, multipart audio, timeout, purpose="asr_transcription")
  -> NetResponse
  -> TranscriptPayload

load_openrouter_models(cache_root, runtime)
  -> NetRequest(GET, https://openrouter.ai/api/v1/models, purpose="model_registry")
  -> NetResponse
  -> list[OpenRouterModel]

_read_subtitle_text(url, runtime)
  -> NetRequest(GET, url, purpose="subtitle_fetch")
  -> NetResponse
  -> VTT/m3u8 parsing
```

## Async/multi-connection migration rule

Do not begin with broad async conversion.

First normalize call sites behind sync `NetRuntime`. Then add async/multi-request APIs at the runtime boundary.

Migration order:

```text
1. Introduce sync NetRuntime contracts and default urllib implementation.
2. Move one leaf call, preferably OpenRouter registry fetch, behind NetRuntime.
3. Move VLM describe leaf behind NetRuntime.
4. Add batch shape for visual descriptions: describe_many(frame_inputs).
5. Add HttpxNetRuntime/request_many implementation.
6. Wire VLM describe_many through request_many with bounded concurrency.
7. Add retry/backoff only if real endpoint behavior requires it.
```

This avoids contaminating transform/product APIs with async before the boundary is stable.

## Efficiency target

The performance problem is not `AiRoute`; it is leaf API execution.

Current issue:

```text
synchronous urlopen per call
no shared connection/session
no provider-level semaphore
no request batching
no normalized retry/backoff
```

Target:

```text
one injected runtime per prepare run
shared connection/session later
bounded request_many for frame/VLM calls
provider/task concurrency limits
secret-safe retry/error behavior
```

## Test strategy

Default tests must not perform live network calls.

Use adapter-boundary tests:

```text
real app/transform path
patched NetRuntime returning deterministic NetResponse
assert manifest/artifacts/internal products
```

Use opt-in real network smoke/integration only for fixed public sources or explicitly configured providers.

## Current status

Implemented:

```text
src/vctx/net.py
  NetRequest
  NetResponse
  NetRetryPolicy
  NetRuntime protocol
  BatchNetRuntime protocol
  UrllibNetRuntime(request with GET retry)
  HttpxNetRuntime(request + request_many with GET retry)

src/vctx/transforms/model_resolution.py
  load_openrouter_models(..., net=NetRuntime | None)
  OpenRouter registry GET uses NetRequest(purpose="model_registry")

src/vctx/transforms/visual_vlm.py
  OpenAiCompatibleVisionAdapter(..., net=NetRuntime | None)
  describe(frame) POST uses NetRequest(purpose="vision_description")
  describe_many(frames) builds request batches
  describe_many uses BatchNetRuntime.request_many when available

src/vctx/transforms/visual_execute.py
  describe actions call adapter.describe_many(frames) once per action

src/vctx/transforms/asr.py
  OpenAiCompatibleAsrAdapter(..., net=NetRuntime | None)
  online ASR POST uses NetRequest(purpose="asr_transcription")

src/vctx/sources/ytdlp_source.py
  YtDlpSourceAdapter(net_factory=Callable[[], NetRuntime] | None)
  subtitle and HLS VTT GETs use NetRequest(purpose="subtitle_fetch")
```

Still direct synchronous leaves:

```text
none outside src/vctx/net.py
```

Current network stage: direct HTTP is centralized in `src/vctx/net.py`; source/model adapters use `NetRuntime`; GET requests have Tenacity retry/backoff; POST VLM/ASR requests remain non-retried by default because of billing/idempotency risk.
