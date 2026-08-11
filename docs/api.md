# vctx API

This document defines the initial public API for `vctx`: the CLI commands, artifact files, JSON shapes, and user/agent interaction contract.

`vctx` is a CLI-first tool. Python internals may change, but the CLI behavior and artifact shapes should be treated as the stable integration surface.

## Architecture contract

```text
CLI request
  -> app workflow and source admission
  -> source adapters
  -> normalized transcript/media models
  -> bounded transforms
  -> render bundle
  -> artifacts + manifest
```

| Layer | Owns | Must not own |
| --- | --- | --- |
| `cli` | flags, request construction, summary printing | provider calls, workflow policy |
| `app` | effect shells, workflow order, resolved policy, scoped runtime lifetime, manifest steps | provider payloads, rendering internals |
| `source` | local/URL metadata, subtitles, media acquisition | chunking, rendering, transforms |
| `transcript` | deterministic parsing, normalization, and chunks | provider calls, output policy |
| `asr` | local faster-whisper planning, admission, and execution | final rendering, source acquisition |
| `visual` | evidence planning, PyAV frames, OCR, VLM, and evidence models | source acquisition, workflow composition |
| `render` | Markdown/JSON projections from typed products | source, model, or network access |
| `net` | scoped HTTPX client, caller-declared Tenacity retry execution, attempt evidence | product semantics, inferred retry policy |

Dependencies point inward toward typed models:

```text
cli -> app -> source/asr/visual -> artifact/render/io
            ai/provider transport -> net
```

Forbidden dependencies are `render ->` acquisition or network, `source ->`
capability execution/render, and capability models -> app composers. Route
identity, credential location, provider receipt identity, and request policy are
independent values.

`app/run.py` owns one persistent, deterministically closed HTTP runtime per
prepare command. Source subtitle/HLS fetches and OpenAI-compatible calls receive
that runtime explicitly. Metadata and OpenRouter login own shorter scoped
runtimes. yt-dlp's internal provider transport remains yt-dlp-owned; all other
vctx HTTP passes through `net.py`. `urllib.parse` is used only for pure URL
parsing.

Callers declare retry policy on each request. Subtitle/HLS GET permits bounded
connection, timeout, `429`, and selected `5xx` retries. AI POST permits one retry
for pre-response connection failure, `429`, or selected `5xx`, preserves the
idempotency key, and never retries an ambiguous response timeout. OpenRouter
OAuth exchange is single-attempt. Responses and typed failures report actual
attempt counts; retries are never stacked.

Configuration file/platform reads (`load_config`) are separate from deterministic
policy resolution (`resolve_config`). Source syntax selection is separate from
effectful source opening. AI selection produces a secret-free `AiRoute`; one
effectful credential read produces a process-local, non-serializable binding.
Diagnostics and artifacts expose only the credential locator/source.

Visual evidence is transcript-led: a validated language-neutral evidence plan
selects bounded capture/OCR/VLM actions. `evidence.json` stores captures and typed
observations; processor admission and failures are recorded in the manifest. No
validated frame request means no visual fetch. Missed satisfaction is a manifest
warning.

The output directory is the stable integration boundary. Internal modules may
move, but CLI behavior, exit status, manifest discovery, and documented artifact
schemas require explicit compatibility treatment.

## CLI principles

- Non-interactive by default.
- No embedded chat or Q&A.
- Deterministic transcript/subtitle acquisition is tried before model routes.
- Optional model/tool branches are explicit in config, workflow policy, and `manifest.json` evidence.
- All durable output goes to the explicit `--out` directory.
- stdout is for final machine/human-consumable result lines.
- stderr is for progress, warnings, and errors.
- `manifest.json` is the first artifact a downstream agent should inspect.
- Local model acquisition occurs only through `vctx models pull`; `prepare`,
  `models status`, and `models verify` are network-free for local models.

## Managed local models

`vctx models pull [asr] [ocr]` prepares selected local models under the runtime
cache and writes integrity receipts. `status` reports receipt presence without
reading model contents; `verify` hashes cached files and reports `ready`,
`missing`, or `corrupt`. All commands support `--json`, `--cache-dir`,
`--config`, and the ASR selector `--asr`.

Receipts expose capability, provider, model ID, cache-relative path, byte size,
package version, and integrity. They never include credentials or absolute host
paths. License data is omitted unless an upstream source supplies an
authoritative identifier.

## Source-cache lifecycle

```bash
vctx cache status [--json] [--cache-dir DIR] [--config PATH]
vctx cache prune [--dry-run] [--age 30d | --all] [--json]
                 [--cache-dir DIR] [--config PATH]
```

`status` is read-only and reports `records`, `assets`, `blobs`, `temporary`, and
`bytes`; a missing store is an empty inventory and is not created. `prune`
without an age/all selector removes only validated orphan blobs and temporary
files. `--age` accepts positive hours, days, or weeks such as `12h`, `30d`, or
`4w`; it uses best-effort last use and falls back to acquisition time. `--all`
is the only way to select every verified source record.

Prune JSON reports `dry_run`, `examined`, `selected`, `removed`,
`reclaimed_bytes`, and `failures`. Dry-run uses the same selection as a real
prune but removes nothing. Catalog authority is retired transactionally before
unreferenced blobs are deleted; shared blobs survive, and failed file deletion
is reported for a later retry. Invalid catalogs, links, blob names, and busy
writers refuse unsafe mutation. `--cache-dir` selects its `source/` child and
overrides config; `[cache].source_dir` otherwise resolves relative to the config
file. Neither command inspects or mutates `[cache].model_dir`.

## Stage-2 acceptance matrix

| Area | Accepted behavior | Observable boundary |
| --- | --- | --- |
| Local source | Local transcript/media is admitted without network access. | One independent source lane is published. |
| URL source | Online prepare observes a URL once and may seed verified source-cache records. | Provider effects and freshness are recorded under that source only. |
| Offline URL | A verified cache hit is usable; a miss, corrupt record, or unverified asset fails closed. | No provider runtime or output pack is created on admission failure. |
| Transcript route | Official/manual subtitles win; ASR runs only when transcript text is absent and policy permits it. | Route evidence and retained inputs stay in the owning lane. |
| Visual route | Visual work runs only when requested and useful; frames feed RapidOCR and optional VLM routes. | Captures, scores, warnings, and model evidence stay in the owning lane. |
| Retention | Required media/subtitles are retained by default; explicit omission is supported. | Each asset has integrity metadata or an omission receipt. |
| Batch result | Inputs execute independently in CLI order; duplicate identities collapse. | Successes survive sibling failures and the root status becomes `partial` or `error` as applicable. |
| Existing pack | Add, reuse, replace, and forced rebuild are source-identity upserts. | Unrequested lanes remain byte-identical; unknown/corrupt packs are refused. |
| Publication | Pack replacement uses staging, verification, backup, rename, and recovery. | Failure or cancellation preserves the last verified generation. |
| Metadata | `metadata` shares source admission, config, cache, and offline policy with `prepare`. | It reports sanitized source facts without creating a pack. |
| Cache operations | `status` is read-only; `prune` selects verified records/orphans and never model files. | Missing stores stay absent; dry-run and real selection agree. |

The fixed-source network integration check is opt-in because it depends on a
stable public source and real network access. All other rows are mandatory,
network-free acceptance behavior.

## How `vctx prepare` decides what to do

`vctx prepare INPUT... --out DIR` runs the same branch workflow independently
for one or more explicit inputs. It never combines their content.

```text
1. Admit one source session per distinct input identity and observe each URL at most once.
2. Project sanitized normalized metadata from that observation.
3. Try deterministic transcript acquisition first:
   - local `.srt` / `.vtt` / transcript JSON
   - URL subtitles/captions through `yt-dlp`
4. If transcript text is still missing and the workflow allows it, use the selected ASR fallback instance.
5. Normalize transcript segments and build chunks.
6. If the workflow asks for visual evidence and video media is available:
   - extract display-corrected frames in-process with PyAV
   - run local RapidOCR when `rapidocr` is installed
   - optionally run an OpenAI-compatible/OpenRouter VLM description route
   - preserve frame captures as artifacts
7. Validate language-neutral claims, relations, and transcript-anchored frame requests.
8. Optionally merge configured text-model supplements when enabled.
9. Write JSON/Markdown artifacts and `manifest.json`.
```

Workflow presets decide which branches are allowed:

| Workflow | Transcript branch | Visual branch | Knowledge-flow branch | Typical outputs |
| --- | --- | --- | --- | --- |
| `default` | deterministic transcript; ASR only if configured and needed | auto/optional | deterministic, optional configured supplement | manifest, metadata, transcript/chunks/context/readable |
| `transcript` | transcript-focused; ASR only if configured and needed | off | off | transcript/chunks/context/readable |
| `visual` | transcript plus visual evidence when video media exists | on; PyAV frame production | deterministic/auto | visual records and frame artifacts when captured |
| `full` | transcript + visual + configured supplements | on; PyAV frame production | on when configured | all applicable artifacts |
| `metadata` | metadata only | off | off | `metadata.json` + `manifest.json(status=partial)` |

Configuration answers two questions:

1. Which workflow/default policy should this run use?
   - `runtime.workflow`
   - `source.*`
   - `output.*`
2. If a workflow branch needs a model/tool, which implementation is selected?
   - `transforms.asr.use = "instance:<name>"` -> `[instances.asr.<name>]`, or use `auto`
   - `evidence.planner` -> `auto`, `none`, or `instance:<name>`
   - `evidence.vision` -> `auto`, `none`, or `instance:<name>`
   - `evidence.ocr` -> `auto` or `none`

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

Base transcript workflows do not require ASR, OCR, VLM, or frame decoding. Extra packages are only needed when the selected workflow reaches the corresponding branch.

| Tool/package | Needed for | Install / availability |
| --- | --- | --- |
| `yt-dlp` Python package | URL metadata and subtitles; URL media download when ASR/visual media is needed | Project dependency. `vctx doctor` reports availability. |
| `av` (PyAV 18) + Pillow | In-process video decode, display orientation, scaling, and PNG frame publication | Installed by `vctx[visual]` and `vctx[full]`; no host media executable is required. |
| `rapidocr` + `onnxruntime` Python packages | Local OCR over extracted frames | Install the visual extra, for example `uv sync --extra visual` or package equivalent. If absent, local OCR action is unavailable. |
| `faster_whisper` Python package | Local ASR through `type = "local-faster-whisper"` | Install the ASR extra, for example `uv sync --extra asr` or package equivalent. |
| `vctx[full]` optional extra | Installs all local heavy feature extras currently declared by the project | Use `uv sync --extra full` when you want ASR + visual/OCR support in one environment. Default installs stay small. |
| `OPENROUTER_API_KEY` | Optional OpenRouter free/ZDR auto route | Prefer `vctx auth openrouter login`; shell env or a listed `runtime.env_files` file is also accepted. |

Published extras are flat PEP 621 dependency lists. `full` is mechanically
verified as the normalized union of `asr` and `visual`; selecting `[asr,visual]`
simultaneously is verified as the equivalent independent composition. CI builds
the wheel once and runs all profile smokes against that artifact rather than the
source tree.

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
use = "instance:local-default"  # arbitrary name

[instances.asr.local-default]
type = "local-faster-whisper"
model = "small"
device = "auto"
compute = "auto"
cache = "persistent"
```

`local-default` and `local-model` are example names, not magic built-ins. The user can name a local instance anything and select it with `transforms.asr.use = "instance:<name>"`.

Current ASR instance types:

| Type | Behavior |
| --- | --- |
| `local-faster-whisper` | Runs local multilingual faster-whisper `small` by default. Managed model ids use `cache.model_dir`; `path:<local-path>` uses local files only. Execution never downloads models. |

Local ASR always transcribes in the detected language. It uses Silero VAD with
threshold `0.5`, minimum silence `2000 ms`, and speech padding `400 ms`. An empty
VAD pass is confirmed once without VAD; only two successful empty passes produce
`no_speech`. Segment timestamps are admitted at millisecond precision; negative,
non-finite, or unordered anchors are rejected. Runtime loads only prepared
CTranslate2 directories and never pulls a model.

### Visual frames, OCR, and VLM descriptions

Visual/full workflows use one PyAV decoder per source. Targets resolve to the frame displayed at the requested media time, apply display orientation, never upscale, and publish PNG captures under `frames/` with a 2560-pixel maximum long edge. OCR, VLM, and the human-visible artifact consume the same pixels.

Local OCR is available only when `rapidocr` is importable. Its provider id is `rapidocr`. `vctx` does not currently expose a separate OCR model selector; RapidOCR model/cache behavior belongs to that package.

Text and image inference share one OpenAI-compatible AI instance contract.

Automatic OpenRouter-free route:

```toml
[evidence]
planner = "auto"                # authenticated OpenRouter free/ZDR router
vision = "auto"
ocr = "auto"
```

Named AI instance:

```toml
[evidence]
vision = "instance:my-vlm"

[instances.ai.my-vlm]
base_url = "https://example.invalid/v1"
credential = "env:MY_VLM_API_KEY"
model = "my-vision-model"
format = "auto"                 # schema, json, prompt, or negotiated auto
```

`use = "auto"` uses `openrouter/free` only when network use is allowed and an OpenRouter credential is present in the system keyring or `OPENROUTER_API_KEY`. Every request requires zero-data-retention and parameter support. It never selects a paid route.

Credentials are exactly `env:NAME` or `keyring:NAME`. `keyring:openrouter` is a
storage locator and may be used by any explicitly configured HTTPS instance; it
does not activate OpenRouter fields or bind a host. Only the built-in auto route
selects the canonical OpenRouter endpoint and free/ZDR request policy.

Authenticate without putting a secret in config:

```bash
vctx auth openrouter login             # desktop loopback PKCE
vctx auth openrouter login --headless  # paste-code PKCE
vctx auth openrouter status
vctx auth openrouter logout
```

### Evidence planning

`evidence.planner` selects one language-neutral structured planning call over deterministic transcript windows. The model may propose claims, relations, and segment anchors, but code validates ownership and derives all frame timestamps. Missing or failed planning produces a transcript-only partial lane without keyword fallback.

## Commands

### `vctx prepare`

Prepare a complete context pack from a URL or local transcript/media input.

```bash
vctx prepare INPUT... --out DIR [OPTIONS]
```

Examples:

```bash
vctx prepare "https://www.youtube.com/watch?v=abc123" --out ./out/abc123
vctx prepare ./lecture.vtt --out ./out/lecture
vctx prepare ./part-1.vtt ./part-2.vtt --out ./out/course
```

Inputs:

| Argument | Description |
| --- | --- |
| `INPUT...` | One or more explicit URL/local paths. Each locator must select one finite item; playlist expansion is not implicit. |

Inputs run in CLI order. Repeated stable identities collapse to one lane. A
controlled failure does not erase successful sibling lanes; the command exits
nonzero and the root manifest is `partial`. Source effects, assets, warnings,
and transform evidence never cross lane boundaries.

Options:

| Option | Default | Description |
| --- | --- | --- |
| `--out DIR` | required | Output directory for durable artifacts. |
| `--overwrite` | unset | Force requested lanes to rebuild when upserting a verified schema-2 pack. |
| `--chunk-max-chars INT` | `6000` | Maximum approximate characters per chunk before flushing. |
| `--chunk-max-seconds INT` | unset | Optional maximum chunk duration. |
| `--cache-dir DIR` | platform cache dir | Override the base containing `source/` and `models/`. |
| `--media-quality QUALITY` | `auto` | Source-media intent: `auto`, `fast`, `balanced`, or `high`. |
| `--keep-temp` | unset | Preserve temporary downloads/intermediate files. |
| `--workflow NAME` | `default` | Select a preparation workflow: `default`, `transcript`, `visual`, `full`, or `metadata`. |
| `--asr SELECTOR` | config/workflow default | ASR selector: `auto`, `none`, `instance:<name>`, or `local:<model>`. |
| `--ocr SELECTOR` | config/workflow default | Frame OCR selector: `auto` or `none`. |
| `--vision SELECTOR` | config/workflow default | Vision-description selector: `auto`, `none`, or `instance:<name>`. |
| `--no-retain-media` | unset | Omit required source media from the pack and record that omission in the manifest. |
| `--offline` | unset | Use offline policy; network/model-service routes are unavailable. |
| `--config PATH` | unset | Optional TOML config file. Missing fields keep built-in defaults; CLI/request values override config fields. |

Offline admits local inputs and verified cached URL observations/subtitles. A
miss exits with `offline URL cache miss` before invoking `yt-dlp`, constructing a
network runtime, or creating an output pack. Source records live in
`cache.source_dir/index.sqlite3`; verified content-addressed bytes live under
`cache.source_dir/blobs/`. Raw locators, credentials, and signed subtitle URLs
are not persisted.

An absent or empty output creates a pack. An existing output is accepted only
when its schema-2 manifest, owned lane set, retained assets, artifact sizes,
and hashes verify. Prepare then upserts by stable source identity: a new source
adds a lane, a matching revision reuses it, and a changed revision replaces only
that lane. Unrequested lanes stay byte-identical. Publication uses sibling
staging plus backup/rename/restore, so failure or cancellation cannot expose a
mixed generation. `--overwrite` never bypasses ownership or integrity checks.

Default transcript-bearing output files:

```text
DIR/
  manifest.json
  youtube-abc123/
    metadata.json
    subtitle.en.vtt
    transcript.json
    chunks.json
    context.md
    read.md
```

Visual/full or supplement branches may additionally write:

```text
DIR/<source-key>/
  evidence.json
  frames/frame-*.png
  evidence-plan.json
```

Artifact orthogonality:

```text
manifest.json          pack audit and source-lane index; inspect first
metadata.json          source metadata
subtitle.<lang>.<ext>  original native subtitle bytes when retained
transcript.json        canonical normalized transcript used by transforms
chunks.json            context-window chunks
evidence-plan.json     validated claims, relations, frame targets, omissions, and receipts
evidence.json          canonical captures with typed OCR/VLM observations
frames/frame-*.png     frame artifacts referenced by evidence captures
context.md             AI-agent context injection projection
read.md                human inspection projection
```

`context.md` and `read.md` intentionally overlap in source material but serve different consumers. JSON artifacts are canonical machine records; Markdown files are projections.

Current artifact contract:

| Artifact | When written | Purpose |
| --- | --- | --- |
| `manifest.json` | every successful or partial prepare | Route, warning, evidence, and artifact index. Start here for automation. |
| `metadata.json` | every successful or partial prepare | Normalized input/source metadata. |
| `subtitle.<language>.<ext>` | native subtitle retained | Original acquired subtitle bytes for inspection and reparsing. |
| `transcript.json` | transcript-bearing workflows | Canonical parsed and deterministically normalized transcript segments. |
| `chunks.json` | transcript-bearing workflows | Chunked transcript for downstream context windows. |
| `context.md` | `context` format enabled | Agent-oriented context artifact; includes visual evidence and validated plan claims when available. |
| `read.md` | `readable` format enabled | Human-readable inspection artifact; includes validated plan claims when available. |
| `evidence.json` | visual/full workflow with captured visual evidence | Canonical captures and typed `ok`, `empty`, `unavailable`, or `failed` processor observations. |
| frame image files | visual/full workflow with captures | PNG frame artifacts referenced from evidence and manifest. |
| `evidence-plan.json` | evidence planning ran, including a valid empty result | Canonical validated claims, relations, derived frame targets, omissions, and per-window receipts. |

Current MVP stage:

```text
auditable context pack
  -> deterministic transcript/prose flow extraction
  -> motive-led visual evidence when useful
  -> evidence.json
  -> validated evidence-plan.json
  -> rendered context/readable projections
```

Parked until a clear ergonomic consumer exists: chapters, rich graph semantics, and broad claim-validation subsystems.

Successful stdout:

```text
Wrote context pack: DIR
Manifest: DIR/manifest.json
Artifacts:
  - SOURCE-KEY/context.md
  - SOURCE-KEY/read.md
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

`--workflow` supplies defaults only. Explicit `--asr`, `--ocr`, and `--vision`
selectors override their own capability independently; none of them enables a
different capability. CLI selectors override the chosen config file.

Required source bytes are retained by default inside their flat
`DIR/<source-key>/` lane. Fixed names are `subtitle.<language>.<ext>`,
`audio.<ext>`, `video.<ext>`, and `input.<ext>`. The corresponding
`manifest.sources[].assets` records purpose, selected policy, size, and SHA-256
integrity independently from that source's rendered `artifacts`.
`--no-retain-media` records explicit omissions. Temporary yt-dlp files remain
in the source cache.

Example:

```toml
[runtime]
workflow = "transcript"          # default | transcript | visual | full | metadata
env_files = [".env"]             # optional; loaded only for provider credentials
keep_temp = false

[cache]
source_dir = ".cache/vctx/source" # optional; config-relative
model_dir = ".cache/vctx/models"  # optional; config-relative

[source]
media_quality = "auto"            # auto | fast | balanced | high

[source.yt_dlp]
session = "browser:chrome"       # optional: none | browser:<name> | cookies-file:<path>
network = "proxy:socks5://127.0.0.1:1080" # optional: direct | proxy:<url>
playlist = "items:1"             # optional: default | items:<yt-dlp item spec>

[output]
formats = ["json", "context", "readable", "transcript"]
chunk_max_chars = 6000
chunk_max_seconds = 900

[transforms.asr]
use = "instance:local-default"   # arbitrary example name selected below

[instances.asr.local-default]
type = "local-faster-whisper"
model = "small"                  # built-in default
device = "auto"                 # auto, cpu, or cuda
compute = "auto"
cache = "persistent"             # managed weights under cache.model_dir

[instances.asr.local-model]
type = "local-faster-whisper"
model = "path:D:/models/faster-whisper-tiny"  # explicit path => no managed cache/download

[evidence]
planner = "auto"
vision = "auto"  # authenticated OpenRouter free/ZDR route when available
ocr = "auto"
# or choose a named vision instance:
# use = "instance:my-vlm"

[instances.ai.my-vlm]
base_url = "https://example.invalid/v1"
credential = "env:MY_VLM_API_KEY"
model = "my-vision-model"

```

Transform selector field `use` is the public model/tool selection surface. Runtime network/upload constraints come from execution context, not user config. Normal public config should choose exactly one selector value; separate `route`/`instance`/`model` transform fields are not part of the current config surface.

```text
auto                  -> let vctx choose the implemented default for that transform
none                  -> disable that transform branch
instance:<name>       -> select [instances.asr.<name>] or [instances.ai.<name>]
path:<local-path>     -> local model/resource path; config-relative where supported
local:<path-or-id>    -> local model id/path for local-capable transforms
hf:<repo-id>          -> managed local cache route when a compatible runtime exists
```

Field semantics:

| Field | Semantics |
| --- | --- |
| `runtime.workflow` | Default workflow profile when CLI `--workflow` is not supplied. |
| `cache.source_dir` / `cache.model_dir` | Independent persistent source and model stores. Config-relative paths resolve from the config file; one-off `--cache-dir` overrides both with `source/` and `models/` children. |
| `runtime.env_files` | Optional dotenv files to consult during credential resolution. Relative config values resolve from the config file directory. Secrets are not copied into manifests/config dumps. |
| `source.yt_dlp.session` | Optional source session access as `none`, `browser:<name>`, or `cookies-file:<path>`. Omit unless yt-dlp needs login/session cookies. |
| `source.yt_dlp.network` | Optional source network route as `direct` or `proxy:<url>`. Omit for direct network. |
| `source.yt_dlp.playlist` | Optional playlist/multipart selector as `default` or `items:<spec>`. Omit unless the source URL resolves to the wrong playlist item. |
| `source.media_quality` | Generic media intent: `auto`, `fast`, `balanced`, or `high`. ASR still selects audio for its purpose. |
| `output.formats` | Default render/artifact formats for `prepare`; the prepare CLI does not expose a `--format` flag. |
| `output.retain_media` | Retain required URL/local media as manifest-listed pack artifacts; defaults to `true`. |
| `transforms.asr.use` | ASR fallback selector: `auto`, `none`, `instance:<name>`, `local:<model-or-path>`, or `path:<local-path>`. Runs only if deterministic transcript acquisition fails and media is available. |
| `evidence.ocr` | Local frame-OCR selector: `auto` or `none`. |
| `evidence.vision` | Visual-description selector: `auto`, `none`, or `instance:<name>`. |
| `evidence.planner` | Language-neutral evidence-plan selector: `auto`, `none`, or `instance:<name>`. |
| `instances.ai.<name>` | Provider-neutral OpenAI-compatible `/v1` root and model, reusable by text or image tasks. |
| `instances.asr.<name>.type` | Local ASR implementation type: `local-faster-whisper`. The `<name>` is arbitrary. |
| `instances.asr.<name>.model` | Model id such as `small`, or `path:<local-path>` for an immutable local CTranslate2 model. Omission selects multilingual `small`. |
| `instances.asr.<name>.device` | `auto`, `cpu`, or `cuda`; auto may fall back once to CPU during initialization. |
| `instances.asr.<name>.compute` | Faster-whisper compute type; defaults to `auto`. |
| `instances.asr.<name>.cache` | `persistent` stores managed faster-whisper weights under `cache.model_dir`; `disabled` requires `path:<local-path>`. |
| `instances.ai.<name>.base_url` | OpenAI-compatible `/v1` root; vctx appends `/chat/completions`. Remote cleartext HTTP is rejected unless `insecure = true`. |
| `instances.ai.<name>.credential` | Exactly `env:NAME` or `keyring:NAME`; remote instances require a locator and secrets are never persisted. Names carry no endpoint or policy semantics. |
| `instances.ai.<name>.model` | Model id sent to the endpoint. |
| `instances.ai.<name>.format` | `auto`, `schema`, `json`, or `prompt`; auto falls back only when the endpoint explicitly rejects a format. |

Configured VLM routes are selected only when a named AI instance is selected, its credential is present, and the manifest can record upload/cost evidence. ASR execution is local-only.

### Auto-adaptive transformations

Model transformations are capability-level defaults, not provider menus. The normal API should avoid asking users to choose `local` vs `online` vs provider names.

Manifest route evidence semantics:

| Manifest route | Meaning |
| --- | --- |
| deterministic | Use source data such as local transcript files, official subtitles, automatic subtitles, or deterministic extraction. |
| local | Use local installed tools/packages such as faster-whisper or RapidOCR. |
| free-online | Use a free online route when policy, credentials, upload behavior, and capability checks allow it. |
| configured-online | Use the configured provider/instance selected by config. |
| unavailable | Fail clearly or write a partial manifest, depending on request policy. |

API graph for model transformations:

```text
prepare INPUT...
  -> deterministic acquisition
       -> platform metadata
       -> official/manual subtitles
       -> automatic subtitles
  -> if transcript unavailable and workflow allows transcript fallback:
       -> select prepared local faster-whisper instance
       -> local ASR
       -> timestamped transcript
  -> deterministic transcript normalization
  -> if visual-context workflow enables visual context:
       -> derive transcript-anchored essential visual cases
       -> derive VisualOperationMotive values
       -> discover executable visual actions
       -> plan sample/OCR/describe/capture recipe
        -> extract display-corrected frame images with PyAV
       -> run local RapidOCR if planned and installed
       -> run configured/OpenRouter VLM descriptions if planned
       -> write evidence.json + frames/ artifacts
  -> validate and merge language-neutral evidence-plan windows
  -> chunk/render/write artifacts
  -> manifest records every route and provider actually used
```

The CLI does not expose raw provider menus for normal usage. Selector values such as `local:<path>`, `hf:<repo-id>`, `instance:<name>`, `auto`, and `none` infer route behavior without separate transform route/model fields. OpenRouter is only the authenticated free/ZDR auto recipe; other compatible providers use named AI instances. The manifest records the configured and actual model choice.

### `vctx metadata`

Print normalized metadata for an input without preparing a full context pack.

```bash
vctx metadata INPUT [--json] [--config PATH] [--cache-dir DIR] [--offline]
```

Purpose:

- cheap source inspection
- agent preflight
- debugging extractor behavior
- the same source admission/config policy as `prepare`, without creating a pack

Output:

- human-readable text by default
- `VideoMetadata` JSON with `--json`

### `vctx chunk`

Chunk an existing transcript artifact.

```bash
vctx chunk transcript.json --out chunks.json [--chunk-max-chars 6000]
```

Purpose:

- re-chunk without re-downloading source
- test chunking strategies
- support agent workflows with custom transcript acquisition

Input:

- `transcript.json` or compatible `Transcript` JSON

Output:

- `chunks.json`

### `vctx render`

Render Markdown from existing artifacts.

```bash
vctx render --metadata metadata.json --chunks chunks.json --out context.md --format context
vctx render --metadata metadata.json --transcript transcript.json --out read.md --format readable
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
vctx doctor --workflow visual --offline --no-retain-media
vctx doctor --json
```

The report resolves the same user-facing policy as `prepare` and shows:

- Python version
- installed distribution profile (`core`, `asr`, `visual`, or `full`)
- selected workflow, offline state, and media-retention policy
- ASR, OCR, and vision selectors with readiness
- `yt-dlp` import
- source-cache presence without creating or writing it
- installed distribution profile, including in-process PyAV visual readiness

`doctor` is network-free. `--json` produces the same facts for automation and
never includes credentials.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Generic runtime failure. |
| `2` | Invalid command usage or options. |
| `3` | Unsupported input/source. |
| `4` | Transcript unavailable. |
| `5` | Output, cache integrity/schema, or filesystem error. |
| `6` | Offline policy rejected a source that is not available in verified local cache. |
| `7` | Source provider failed or requires explicit refresh. |
| `130` | Operation cancelled; the active temporary effect was cleaned. |

## Artifact contract

### `manifest.json`

The run ledger and discovery document.

Downstream agents should read this first.

Shape:

```json
{
  "schema_version": "2",
  "tool": "vctx",
  "tool_version": "0.1.0",
  "pack_id": "3db85608-8840-4dc7-afc2-9af6e4300f76",
  "updated_run_id": "83d86ad6-fc75-4c60-ab9a-427611297b0f",
  "status": "ok",
  "created_at": "2026-06-07T12:00:00Z",
  "updated_at": "2026-06-07T12:00:00Z",
  "sources": [
    {
      "id": "youtube__abc123",
      "key": "youtube-abc123",
      "path": "youtube-abc123",
      "kind": "url",
      "revision": {"kind": "observed", "value": "sha256-fingerprint"},
      "freshness": "observed-online",
      "observed_at": "2026-06-07T11:59:58Z",
      "title": "Example",
      "duration_seconds": 120.0,
      "status": "ok",
      "artifacts": [
        {"kind": "metadata", "path": "metadata.json", "media_type": "application/json", "bytes": 420, "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
        {"kind": "context", "path": "context.md", "media_type": "text/markdown", "bytes": 2048, "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"}
      ],
      "effects": [
        {"operation": "observe", "status": "succeeded", "attempts": 1}
      ],
      "assets": [],
      "steps": [],
      "warnings": [],
      "transform_evidence": []
    }
  ]
}
```

Fields:

| Field | Description |
| --- | --- |
| `schema_version` | Artifact schema version. |
| `tool_version` | Installed `vctx` version. |
| `pack_id` | Stable UUID for the pack identity. |
| `updated_run_id` | UUID for the publication attempt that produced this manifest. |
| `status` | `ok`, `partial`, or `error`. |
| `sources` | Independent source entries. Each owns stable identity/key/path, revision, status, artifacts, effects, retained assets, steps, warnings, and transform evidence. Paths inside an entry are relative to its lane. |

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

### `transcript.json`

Canonical transcript after deterministic parsing and normalization.

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

Normalization guarantees:

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

### `read.md`

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

1. Run `vctx prepare INPUT... --out DIR`.
2. Read `DIR/manifest.json`.
3. For each usable `sources[]` entry, enter its `path` and select artifacts:
   - use `<path>/context.md` for context injection
   - use `<path>/chunks.json` for programmatic chunk-by-chunk processing
   - use `<path>/read.md` for human-facing source review
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
- `metadata.json`, `transcript.json`, and `chunks.json` top-level fields

Internal Python module paths are not stable until implementation matures.
