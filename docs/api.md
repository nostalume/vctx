# vctx API

This document defines the initial public API for `vctx`: the CLI commands, artifact files, JSON shapes, and user/agent interaction contract.

`vctx` is a CLI-first tool. Python internals may change, but the CLI behavior and artifact shapes should be treated as the stable integration surface.

## CLI principles

- Non-interactive by default.
- No embedded chat or Q&A.
- Deterministic transcript/subtitle acquisition is tried before model routes.
- Optional model/tool branches are explicit in config, workflow policy, and `manifest.json` evidence.
- All durable output goes to the explicit `--out` directory.
- stdout is for final machine/human-consumable result lines.
- stderr is for progress, warnings, and errors.
- `manifest.json` is the first artifact a downstream agent should inspect.

## How `vctx prepare` decides what to do

`vctx prepare INPUT --out DIR` is a branch workflow, not a provider menu. It resolves the requested workflow and config, then runs only the branches that are useful and available for the input.

```text
1. Detect the input source.
2. Extract normalized metadata.
3. Try deterministic transcript acquisition first:
   - local `.srt` / `.vtt` / transcript JSON
   - URL subtitles/captions through `yt-dlp`
4. If transcript text is still missing and the workflow allows it, use the selected ASR fallback instance.
5. Normalize transcript segments and build chunks.
6. If the workflow asks for visual evidence and video media is available:
   - extract frames with `ffmpeg`
   - run local RapidOCR when `rapidocr` is installed
   - optionally run an OpenAI-compatible/OpenRouter VLM description route
   - preserve frame captures as artifacts
7. Extract deterministic knowledge flow from transcript and kept visual evidence.
8. Optionally merge configured text-model supplements when enabled.
9. Write JSON/Markdown artifacts and `manifest.json`.
```

Workflow presets decide which branches are allowed:

| Workflow | Transcript branch | Visual branch | Knowledge-flow branch | Typical outputs |
| --- | --- | --- | --- | --- |
| `default` | deterministic transcript; ASR only if configured and needed | auto/optional | deterministic, optional configured supplement | manifest, metadata, transcript/chunks/context/readable |
| `transcript` | transcript-focused; ASR only if configured and needed | off | off | transcript/chunks/context/readable |
| `visual` | transcript plus visual evidence when video media exists | on; requires `ffmpeg` for frames | deterministic/auto | visual records and frame artifacts when captured |
| `full` | transcript + visual + configured supplements | on; requires `ffmpeg` for frames | on when configured | all applicable artifacts |
| `metadata` | metadata only | off | off | `metadata.json` + `manifest.json(status=partial)` |

Configuration answers two questions:

1. Which workflow/default policy should this run use?
   - `runtime.workflow`
   - `source.*`
   - `output.*`
2. If a workflow branch needs a model/tool, which implementation is selected?
   - `transforms.asr.instance` -> `[instances.asr.<name>]`, or omit for `auto`
   - `transforms.visual_context.instance` -> `[instances.vision.<name>]`, or use `model = "auto"` / `openrouter:<model-id>`
   - `transforms.knowledge_flow.model` for the current text-model supplement path

Current config precedence is high to low:

```text
CLI/request values
  -> explicit --config file
  -> local project config: ./vctx.toml, then ./.vctx.toml
  -> env-selected config: VCTX_CONFIG
  -> global user config: platform user config dir / vctx/config.toml
  -> built-in defaults
```

`vctx` selects one config file by this precedence. It does not merge multiple config files. `.env` files are used only when listed in `runtime.env_files` and only for credential lookup by a selected provider.

## Required tools and optional extras

Base transcript workflows do not require ASR, OCR, VLM, or `ffmpeg`. Extra tools are only needed when the selected workflow reaches the corresponding branch.

| Tool/package | Needed for | Install / availability |
| --- | --- | --- |
| `yt-dlp` Python package | URL metadata and subtitles; URL media download when ASR/visual media is needed | Project dependency. `vctx doctor` reports availability. |
| `ffmpeg` executable | Visual/full workflows that extract video frames; not needed for transcript-only or metadata workflows | Install from your OS package manager or <https://ffmpeg.org/> and ensure `ffmpeg` is on `PATH`. `vctx doctor` checks it. |
| `rapidocr` + `onnxruntime` Python packages | Local OCR over extracted frames | Install the visual extra, for example `uv sync --extra visual` or package equivalent. If absent, local OCR action is unavailable. |
| `faster_whisper` Python package | Local ASR through `type = "local-faster-whisper"` | Install the ASR extra, for example `uv sync --extra asr` or package equivalent. |
| `vctx[full]` optional extra | Installs all local heavy feature extras currently declared by the project | Use `uv sync --extra full` when you want ASR + visual/OCR support in one environment. Default installs stay small. |
| `OPENROUTER_API_KEY` | OpenRouter registry-backed VLM/text routes | Store in shell env or a file listed by `runtime.env_files`; config stores only the env-var name. |

## Current model/tool semantics

### ASR

`transforms.asr` is a transcript fallback policy. It runs only when deterministic transcript acquisition fails and media is available. It does not run when a transcript already exists.

Automatic local/default route:

```toml
[transforms.asr]
# instance omitted => auto policy
```

Named ASR instance:

```toml
[transforms.asr]
instance = "local-default"  # arbitrary name

[instances.asr.local-default]
type = "local-faster-whisper"
model_policy = "auto"
cache = "persistent"
```

Online ASR instance:

```toml
[transforms.asr]
instance = "openai-whisper"

[instances.asr.openai-whisper]
type = "openai-compatible-audio"
base_url = "https://api.openai.com/v1/audio/transcriptions"
api_key_env = "OPENAI_API_KEY"
model = "whisper-1"
```

`local-default`, `local-model`, and `openai-whisper` are example names, not magic built-ins. The user can name an instance anything and select it with `transforms.asr.instance`.

Current ASR instance types:

| Type | Behavior |
| --- | --- |
| `local-faster-whisper` | Runs local `faster_whisper`. `model_policy = "auto"` currently executes faster-whisper model id `base`. Managed model ids use `runtime.cache_dir/models/faster-whisper`; explicit local paths do not use managed download/cache. |
| `openai-compatible-audio` | Sends multipart audio/media to `base_url` with `model` and credential from `api_key_env`; manifest records upload/cost evidence automatically. |

### Visual frames, OCR, and VLM descriptions

Visual/full workflows need `ffmpeg` to extract frame images from video media. Transcript-only and metadata workflows do not need `ffmpeg`.

Local OCR is available only when `rapidocr` is importable. Its provider id is `rapidocr`. `vctx` does not currently expose a separate OCR model selector; RapidOCR model/cache behavior belongs to that package.

Visual descriptions use OpenRouter model resolution or a named vision instance.

Automatic/pinned OpenRouter route:

```toml
[transforms.visual_context]
model = "auto"                  # select a free capable OpenRouter VLM when possible
# model = "openrouter:<model>"   # pin a specific OpenRouter VLM
```

Named vision instance:

```toml
[transforms.visual_context]
instance = "my-vlm"

[instances.vision.my-vlm]
type = "openai-compatible-vision"
base_url = "https://example.invalid/v1/chat/completions"
api_key_env = "MY_VLM_API_KEY"
model = "my-vision-model"
```

`model = "auto"` may fetch/cache OpenRouter registry metadata when network/upload are allowed and `OPENROUTER_API_KEY` is present. It selects the highest-ranked free capable model from vctx's curated capability ranking, then context length and stable id order. Registry metadata filters capability/cost; it does not prove objective model quality.

### Knowledge-flow and text-model supplements

Deterministic knowledge-flow extraction does not need a model. Current LLM supplement routing is controlled by `transforms.knowledge_flow`; this also gates LLM essential visual case extraction today. That coupling is current behavior, not the ideal long-term config split.

## Commands

### `vctx prepare`

Prepare a complete context pack from a URL or local transcript/media input.

```bash
vctx prepare INPUT --out DIR [OPTIONS]
```

Examples:

```bash
vctx prepare "https://www.youtube.com/watch?v=abc123" --out ./out/abc123
vctx prepare ./lecture.vtt --out ./out/lecture
vctx prepare ./captions.srt --out ./out/captions --overwrite
```

Inputs:

| Argument | Description |
| --- | --- |
| `INPUT` | URL or local path. Initially URL via `yt-dlp`, `.vtt`, `.srt`, transcript JSON, or supported local media when ASR/visual branches need media. |

Options:

| Option | Default | Description |
| --- | --- | --- |
| `--out DIR` | required | Output directory for durable artifacts. |
| `--overwrite` | unset | Allow reusing an existing output directory. |
| `--chunk-max-chars INT` | `6000` | Maximum approximate characters per chunk before flushing. |
| `--chunk-max-seconds INT` | unset | Optional maximum chunk duration. |
| `--cache-dir DIR` | platform cache dir | Override cache location. |
| `--keep-temp` | unset | Preserve temporary downloads/intermediate files. |
| `--workflow NAME` | `default` | Select a preparation workflow: `default`, `transcript`, `visual`, `full`, or `metadata`. |
| `--offline` | unset | Use offline policy; network/model-service routes are unavailable. |
| `--config PATH` | unset | Optional TOML config file. Missing fields keep built-in defaults; CLI/request values override config fields. |

Default transcript-bearing output files:

```text
DIR/
  manifest.json
  metadata.json
  transcript.raw.json
  transcript.clean.json
  transcript.md
  chunks.json
  context.md
  readable.md
```

Visual/full or supplement branches may additionally write:

```text
DIR/
  visual_records.json
  visual_scores.json
  visual/frames/*.png
  knowledge_flow.json
```

Artifact orthogonality:

```text
manifest.json          run audit and artifact index; inspect first
metadata.json          source metadata
transcript.raw.json    source/ASR transcript before normalization
transcript.clean.json  normalized transcript used by transforms
chunks.json            context-window chunks
knowledge_flow.json    canonical evidence-linked flow graph
visual_records.json    canonical OCR/VLM/capture evidence records
visual_scores.json     visual satisfaction diagnostics
visual/frames/*.png    frame artifacts referenced by visual records
context.md             AI-agent context injection projection
readable.md            human inspection projection
transcript.md          human timestamped transcript projection
```

`context.md` and `readable.md` intentionally overlap in source material but serve different consumers. JSON artifacts are canonical machine records; Markdown files are projections.

Current artifact contract:

| Artifact | When written | Purpose |
| --- | --- | --- |
| `manifest.json` | every successful or partial prepare | Route, warning, evidence, and artifact index. Start here for automation. |
| `metadata.json` | every successful or partial prepare | Normalized input/source metadata. |
| `transcript.raw.json` | transcript-bearing workflows | Source transcript/ASR payload before normalization. |
| `transcript.clean.json` | transcript-bearing workflows | Normalized transcript segments. |
| `chunks.json` | transcript-bearing workflows | Chunked transcript for downstream context windows. |
| `transcript.md` | `transcript` format enabled | Human-readable transcript projection. |
| `context.md` | `context` format enabled | Agent-oriented context injection artifact; includes visual records and knowledge-flow summary when available. |
| `readable.md` | `readable` format enabled | Human-readable inspection artifact; includes knowledge-flow summary when available. |
| `visual_records.json` | visual/full workflow with captured visual evidence | Canonical OCR/VLM/capture evidence records only; no satisfaction diagnostics. |
| `visual_scores.json` | visual/full workflow with checked visual motives | Satisfaction diagnostics for required visual operations; missed checks are also manifest warnings. |
| frame image files | visual/full workflow with capture records | PNG frame artifacts referenced from visual records and manifest. |
| `knowledge_flow.json` | transcript or kept visual evidence yields flow edges | Canonical evidence-linked flow nodes/edges from transcript and kept visual records. |

Current MVP stage:

```text
auditable context pack
  -> deterministic transcript/prose flow extraction
  -> motive-led visual evidence when useful
  -> visual_records.json evidence
  -> visual_scores.json diagnostics
  -> evidence-linked knowledge_flow.json
  -> rendered context/readable projections
```

Parked until a clear ergonomic consumer exists: chapters, rich graph semantics, and broad claim-validation subsystems.

Successful stdout:

```text
Wrote context pack: DIR
Manifest: DIR/manifest.json
Context: DIR/context.md
Readable: DIR/readable.md
```

Warnings stderr example:

```text
warning: official subtitles not found; used automatic subtitles for language en
```

Failure stderr example:

```text
error: no transcript found for input; no transcript fallback route is configured. Provide a transcript file, configure an ASR instance, or use --workflow metadata for metadata-only output.
```

### Config file contract

Config is optional and exists to provide workflow defaults and advanced model/tool credentials without turning the CLI into a provider menu.

Precedence, high to low:

```text
CLI/request values
  -> explicit --config file
  -> local project config: ./vctx.toml, then ./.vctx.toml
  -> env-selected config: VCTX_CONFIG
  -> global user config: platform user config dir / vctx/config.toml
  -> built-in defaults
```

`vctx` selects one config file by this precedence; it does not merge multiple config files. Missing fields are not errors. They resolve to built-in defaults or `auto` policy. Secrets are never stored directly; config references environment variable names. `.env` files are optional convenience inputs for those environment variables when listed by `runtime.env_files`.

Downloaded URL media is an output-owned artifact. ASR/visual media downloads write final reusable files under `DIR/media/` plus a small `*.vctx-media.json` sidecar. Temporary/partial yt-dlp files use `runtime.cache_dir/tmp/yt-dlp`. A repeated run into the same output directory reuses matching sidecar+media unless `--overwrite` is used.

Example:

```toml
[runtime]
workflow = "transcript"          # default | transcript | visual | full | metadata
cache_dir = ".cache/vctx"        # optional; relative paths resolve from this config file
env_files = [".env"]             # optional; loaded only for provider credentials
keep_temp = false

[source.yt_dlp]
session = "browser:chrome"       # optional: none | browser:<name> | cookies-file:<path>
network = "proxy:socks5://127.0.0.1:1080" # optional: direct | proxy:<url>
playlist = "items:1"             # optional: default | items:<yt-dlp item spec>
media_profile = "balanced"       # fast | balanced | high; visual quality/speed preset

[output]
formats = ["json", "context", "readable", "transcript"]
chunk_max_chars = 6000
chunk_max_seconds = 900

[transforms.asr]
instance = "local-default"       # arbitrary example name selected below

[instances.asr.local-default]
type = "local-faster-whisper"
model_policy = "auto"            # currently executes faster-whisper model id "base"
cache = "persistent"             # managed weights under runtime.cache_dir/models/faster-whisper

[instances.asr.local-model]
type = "local-faster-whisper"
model = "D:/models/faster-whisper-tiny"  # explicit path => no managed cache/download

[instances.asr.openai-whisper]
type = "openai-compatible-audio"
base_url = "https://api.openai.com/v1/audio/transcriptions"
api_key_env = "OPENAI_API_KEY"   # value can come from shell env or runtime.env_files
model = "whisper-1"

[transforms.visual_context]
model = "auto"  # cached/fetched OpenRouter registry selects a free capable VLM when OPENROUTER_API_KEY exists
# or choose a named vision instance:
# instance = "my-vlm"

[instances.vision.my-vlm]
type = "openai-compatible-vision"
base_url = "https://example.invalid/v1/chat/completions"
api_key_env = "MY_VLM_API_KEY"
model = "my-vision-model"

[transforms.knowledge_flow]
model = "auto"  # current text-model supplement path; deterministic extraction works without this
```

Policy fields such as `route`, `allow_upload`, and `preferred_provider` are advanced routing controls. Normal public config should prefer `auto`, a decisive model reference, or a named `[instances.asr.*]` / `[instances.vision.*]` entry. Local vs online is inferred from the model or instance shape where supported:

```text
openrouter:<model-id>  -> remote OpenRouter route, OPENROUTER_API_KEY credential name, upload as required by capability
local:<path-or-id>     -> local route, no upload, explicit paths are config-relative and disable managed download/cache
hf:<repo-id>           -> managed local cache route when a compatible runtime exists
alias:<name>           -> reserved model-ref shape; named runtime instances use [instances.<capability>.<name>]
none                  -> disable that model-mediated transform
auto                  -> choose the highest-ranked free compatible OpenRouter route for the capability when credentials/network/upload allow it
```

Field semantics:

| Field | Semantics |
| --- | --- |
| `runtime.workflow` | Default workflow profile when CLI `--workflow` is not supplied. |
| `runtime.cache_dir` | Persistent tool/model/media cache. Defaults to the platform user cache directory, e.g. Windows `C:\Users\<user>\AppData\Local\vctx\Cache`. Relative config values resolve from the config file directory; CLI `--cache-dir` values stay relative to the caller CWD. |
| `runtime.env_files` | Optional dotenv files to consult during credential resolution. Relative config values resolve from the config file directory. Secrets are not copied into manifests/config dumps. |
| `source.yt_dlp.session` | Optional source session access as `none`, `browser:<name>`, or `cookies-file:<path>`. Omit unless yt-dlp needs login/session cookies. |
| `source.yt_dlp.network` | Optional source network route as `direct` or `proxy:<url>`. Omit for direct network. |
| `source.yt_dlp.playlist` | Optional playlist/multipart selector as `default` or `items:<spec>`. Omit unless the source URL resolves to the wrong playlist item. |
| `source.yt_dlp.media_profile` | Visual media quality/speed preset: `fast`, `balanced`, or `high`. ASR always uses audio-only demand automatically. |
| `output.formats` | Default render/artifact formats for `prepare`; the prepare CLI does not expose a `--format` flag. |
| `transforms.asr.instance` | Optional ASR fallback instance selector from `[instances.asr.<name>]`; omit for auto. Runs only if deterministic transcript acquisition fails and media is available. |
| `transforms.visual_context.model` | Visual-description model reference. Current `auto`/`openrouter:<model-id>` paths use OpenRouter when credentials and policy allow. |
| `transforms.visual_context.instance` | Optional vision instance selector from `[instances.vision.<name>]`; use when you want a named OpenAI-compatible VLM endpoint. |
| `transforms.knowledge_flow.model` | Current text-model supplement selector. Deterministic knowledge-flow extraction does not need a model. |
| `instances.asr.<name>.type` | ASR implementation type: `local-faster-whisper` or `openai-compatible-audio`. The `<name>` is arbitrary. |
| `instances.asr.<name>.model_policy` | Local faster-whisper managed model policy. Current `auto` executes model id `base`. |
| `instances.asr.<name>.model` | Either a model id such as `tiny`/`base` for managed persistent cache, or an explicit local model path. A local path uses local files only. |
| `instances.asr.<name>.cache` | `persistent` stores managed faster-whisper weights under `runtime.cache_dir/models/faster-whisper`; `disabled` requires an explicit local model path. |
| `instances.asr.<name>.api_key_env` | Environment variable containing an ASR credential. The config stores only the variable name. |
| `instances.vision.<name>.type` | Vision implementation type. Current implemented value: `openai-compatible-vision`, using chat-completions style image messages. |
| `instances.vision.<name>.base_url` | VLM chat-completions endpoint. |
| `instances.vision.<name>.api_key_env` | Environment variable containing the VLM credential; values can come from shell env or `runtime.env_files`. |
| `instances.vision.<name>.model` | VLM model id sent to the endpoint. |

Configured online ASR/VLM routes are selected only when a named online instance is selected, required credentials are present, and the manifest can record upload/cost evidence.

### Auto-adaptive transformations

Model transformations are capability-level defaults, not provider menus. The normal API should avoid asking users to choose `local` vs `online` vs provider names.

Default routing semantics:

| Route | Meaning |
| --- | --- |
| deterministic | Use source data such as local transcript files, official subtitles, automatic subtitles, or deterministic extraction. |
| local | Use local installed tools/packages such as faster-whisper or RapidOCR. |
| free-online | Use a free online route when policy, credentials, upload behavior, and capability checks allow it. |
| configured-online | Use the configured provider/instance selected by config. |
| unavailable | Fail clearly or write a partial manifest, depending on request policy. |

API graph for model transformations:

```text
prepare INPUT
  -> deterministic acquisition
       -> platform metadata
       -> official/manual subtitles
       -> automatic subtitles
  -> if transcript unavailable and workflow allows transcript fallback:
       -> select configured ASR instance
       -> local/configured-online ASR
       -> timestamped transcript
  -> deterministic transcript normalization
  -> if visual-context workflow enables visual context:
       -> derive transcript-anchored essential visual cases
       -> derive VisualOperationMotive values
       -> discover executable visual actions
       -> plan sample/OCR/describe/capture recipe
       -> extract frame images with ffmpeg
       -> run local RapidOCR if planned and installed
       -> run configured/OpenRouter VLM descriptions if planned
       -> write visual_records.json evidence + visual_scores.json diagnostics + frame artifacts
  -> deterministic knowledge-flow extraction from transcript and kept visual evidence
  -> optional text-model supplement when transforms.knowledge_flow selects an executable route
  -> chunk/render/write artifacts
  -> manifest records every route and provider actually used
```

The CLI should not expose raw provider menus for normal usage. Prefix-style model/resource references such as `openrouter:<model-id>`, `local:<path>`, `hf:<repo-id>`, `alias:<name>`, `auto`, and `none` are decisive field shapes where implemented: they infer route behavior instead of requiring separate provider/base-url/api-key/cost/upload fields. If two implementations can serve the same capability, `vctx` should choose the best project default and record the actual choice in `manifest.json`.

### `vctx metadata`

Print normalized metadata for an input without preparing a full context pack.

```bash
vctx metadata INPUT [--json]
```

Purpose:

- cheap source inspection
- agent preflight
- debugging extractor behavior

Output:

- human-readable text by default
- `VideoMetadata` JSON with `--json`

### `vctx chunk`

Chunk an existing transcript artifact.

```bash
vctx chunk transcript.clean.json --out chunks.json [--chunk-max-chars 6000]
```

Purpose:

- re-chunk without re-downloading source
- test chunking strategies
- support agent workflows with custom transcript acquisition

Input:

- `transcript.clean.json` or compatible `Transcript` JSON

Output:

- `chunks.json`

### `vctx render`

Render Markdown from existing artifacts.

```bash
vctx render --metadata metadata.json --chunks chunks.json --out context.md --format context
vctx render --metadata metadata.json --transcript transcript.clean.json --out readable.md --format readable
```

Purpose:

- regenerate Markdown after renderer changes
- support external pipelines that already have JSON artifacts

Formats:

```text
context
readable
transcript
```

### `vctx doctor`

Inspect local environment.

```bash
vctx doctor
```

Checks:

- Python version
- package versions
- `yt-dlp` import
- cache directory writability
- optional `ffmpeg` availability
- optional ASR dependencies when installed

No network calls by default.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Generic runtime failure. |
| `2` | Invalid command usage or options. |
| `3` | Unsupported input/source. |
| `4` | Transcript unavailable. |
| `5` | Output directory or filesystem error. |

## Artifact contract

### `manifest.json`

The run ledger and discovery document.

Downstream agents should read this first.

Shape:

```json
{
  "schema_version": "0.1",
  "tool": "vctx",
  "tool_version": "0.1.0",
  "status": "ok",
  "input": "https://www.youtube.com/watch?v=abc123",
  "created_at": "2026-06-07T12:00:00Z",
  "artifacts": [
    {
      "kind": "metadata",
      "path": "metadata.json",
      "media_type": "application/json"
    },
    {
      "kind": "context",
      "path": "context.md",
      "media_type": "text/markdown"
    }
  ],
  "steps": [
    {
      "name": "source.detect",
      "status": "ok",
      "detail": "yt-dlp"
    },
    {
      "name": "transcript.extract",
      "status": "ok",
      "detail": "official_subtitles:en:vtt"
    },
    {
      "name": "transform.asr",
      "status": "skipped",
      "detail": "transcript already available"
    }
  ],
  "warnings": [],
  "transform_evidence": [
    {
      "capability": "asr",
      "selected_route": "skipped",
      "deterministic": true,
      "reason": "transcript already available"
    }
  ]
}
```

Fields:

| Field | Description |
| --- | --- |
| `schema_version` | Artifact schema version. |
| `tool_version` | Installed `vctx` version. |
| `status` | `ok`, `partial`, or `error`. |
| `input` | Original user input string. |
| `artifacts` | Files written relative to output directory. |
| `steps` | Ordered pipeline steps with compact status/details. |
| `warnings` | Recoverable issues. |
| `transform_evidence` | Structured route evidence for model-mediated capabilities, including selected route, provider/model id, upload/cost flags, deterministic flag, and reason. Secrets are never recorded. |

### `metadata.json`

Normalized source metadata.

Shape:

```json
{
  "id": "youtube__abc123",
  "source_type": "youtube",
  "source": {
    "kind": "url",
    "value": "https://www.youtube.com/watch?v=abc123"
  },
  "title": "Example Video",
  "uploader": "Example Channel",
  "duration_seconds": 1234.5,
  "webpage_url": "https://www.youtube.com/watch?v=abc123",
  "language": "en",
  "extractor": "youtube",
  "raw_provider": "yt-dlp"
}
```

### `transcript.raw.json`

Transcript as parsed from source with minimal cleanup.

Shape:

```json
{
  "video_id": "youtube__abc123",
  "provenance": {
    "method": "official_subtitles",
    "language": "en",
    "format": "vtt",
    "provider": "yt-dlp"
  },
  "segments": [
    {
      "id": "seg_000001",
      "start": 0.0,
      "end": 4.2,
      "text": "Welcome to this video.",
      "source_id": "caption-1"
    }
  ]
}
```

### `transcript.clean.json`

Same shape as `transcript.raw.json`, but after deterministic normalization:

- empty segments removed
- whitespace normalized
- simple subtitle markup removed
- segment IDs reassigned if necessary
- chronological order enforced

No summarization or semantic rewriting.

### `chunks.json`

Chunked transcript for agent processing.

Shape:

```json
{
  "video_id": "youtube__abc123",
  "strategy": "chars-v1",
  "chunks": [
    {
      "id": "chunk_0001",
      "start": 0.0,
      "end": 305.2,
      "text": "Welcome to this video...",
      "segment_ids": ["seg_000001", "seg_000002"],
      "char_count": 5840,
      "approx_token_count": 1460
    }
  ]
}
```

### `context.md`

Agent-optimized Markdown.

Characteristics:

- compact metadata
- clear usage note
- chunk tags with IDs and timestamps
- source text preserved

Shape:

```markdown
# Agent Context Pack

## Metadata

- Title: Example Video
- URL: https://www.youtube.com/watch?v=abc123
- Duration: 00:20:34
- Transcript source: official_subtitles / en / vtt

## Usage

The chunks below are timestamped source text extracted from the video.
Preserve timestamps when citing claims.

## Chunks

<chunk id="chunk_0001" start="00:00:00" end="00:05:05">
Welcome to this video...
</chunk>
```

### `readable.md`

Human-readable transcript pack.

Characteristics:

- pleasant Markdown
- time-range headings
- no XML-like chunk tags
- no generated summary by default

Shape:

```markdown
# Example Video

Source: https://www.youtube.com/watch?v=abc123  
Duration: 00:20:34  
Transcript source: official_subtitles / en

## 00:00:00–00:05:05

Welcome to this video...
```

### `transcript.md`

Timestamped cleaned transcript.

Shape:

```markdown
# Transcript — Example Video

[00:00:00–00:00:04] Welcome to this video.
[00:00:04–00:00:09] Today we will discuss...
```

## Interaction with downstream AI agents

Recommended agent flow:

1. Run `vctx prepare INPUT --out DIR`.
2. Read `DIR/manifest.json`.
3. If `status` is `ok` or `partial`, select artifact:
   - use `context.md` for context injection
   - use `chunks.json` for programmatic chunk-by-chunk processing
   - use `readable.md` for human-facing source review
4. The agent performs summarization, knowledge-flow extraction, Q&A, or memory updates outside `vctx`.

Example agent prompt wrapper:

```text
The following is a context pack generated by vctx from a video source.
Use it as source material. Preserve timestamps when citing claims.
Do not assume content not present in the context.

<video_context>
... context.md ...
</video_context>
```

## Stability policy

For early versions, treat these as semi-stable:

- CLI command names
- output file names
- `manifest.json` discovery fields
- `metadata.json`, `transcript.clean.json`, and `chunks.json` top-level fields

Internal Python module paths are not stable until implementation matures.
