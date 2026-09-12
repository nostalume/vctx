# vctx CLI and artifact contract

`vctx` is a one-shot context compiler. Its stable integration surfaces are the
installed CLI and schema-5 output pack. Python modules are internal.

## Workflow

```text
source admission -> transcript -> evidence -> summary
                         |            |          |
                         +------ canonical products
                                      |
                              context/read projections
```

`prepare` runs each source independently. It records selected routes, effects,
warnings, omissions, and artifacts in `manifest.json`. Rendering reads verified
canonical products and never acquires sources or calls providers.

The built-in configuration is zero-TOML, not anonymous AI access. It can prepare
a subtitle-backed transcript without credentials. Evidence planning and summary
with `auto` require `OPENROUTER_API_KEY`, a prior `auth openrouter login`, or an
explicit OpenAI-compatible instance. Missing AI authentication produces no AI
request; the manifest records unavailable downstream products and preserves safe
earlier products.

## Installation profiles

```console
uv tool install vctx
uv tool install "vctx[asr]"
uv tool install "vctx[visual]"
uv tool install "vctx[full]"
```

| Profile | Capability |
| --- | --- |
| core | Local/URL subtitles, source cache, compatible AI routes |
| asr | Core plus local faster-whisper |
| visual | Core plus PyAV frames and RapidOCR |
| full | ASR (including Windows CUDA libraries) and visual capabilities |

PyAV decodes video in-process. No host FFmpeg executable is required. Normal
`prepare` never downloads models; use `models pull` explicitly.
Use the `asr-cuda` extra on Windows for project-local CUDA 12/cuDNN 9 libraries. The
runtime discovers them without mutating `PATH`; `doctor --json` reports their state.

## Migrating from 0.3 to 0.4

Upgrade a uv tool installation with `uv tool upgrade vctx`. Version 0.4 writes
schema 5 and reads immutable schemas 3, 4, and 5. Schema-5 packs place
`manifest.json` and direct source lanes at the root, with role-named source files
beside canonical products. Consumers must follow manifest-relative paths instead
of reconstructing schema-3 paths. Version 0.3 cannot read schema 5; preserve an old
pack copy before downgrading.

For normal retention configuration, replace `output.retain_media` with:

```toml
[output]
source_assets = "consumed" # or "complete"
```

`consumed` preserves files acquired by product work. `complete` explicitly admits
the additional network and storage cost of every audio, video, or native-subtitle
role reported for the source revision. The hidden `--no-retain-media` option and
`output.retain_media = false` remain a deprecated compatibility opt-out.

Named ASR instances may still select a model, but `device` and `compute` are no
longer accepted configuration fields. Use `transforms.asr.quality` for the
user-visible speed/quality choice. Runtime device, compute mode, threads, and
batching are selected automatically and recorded in transcript provenance.

A same-revision prepare may extend an existing verified pack and fetch only
missing roles. A changed source revision requires explicit `--overwrite`; corrupt,
linked, or ambiguous output is refused rather than trusted as an upgrade base.

## Commands

Use command-specific `--help` for the exact option grammar.

Closed CLI string values are complete here: `--to` accepts `transcript`,
`evidence`, or `summary`; `--asr-quality` accepts `fast`, `balanced`, or `accurate`;
`--media-quality` accepts `auto`, `fast`, `balanced`,
or `high`; `render --format` accepts `context`, `read`, or `transcript`; and model
capabilities are `asr` and `ocr`. Capability selectors have their own complete
grammar under [Selectors](#selectors).

### Prepare

```console
vctx prepare INPUT... --out PACK [OPTIONS]
```

Important options:

| Option | Meaning |
| --- | --- |
| `--to transcript|evidence|summary` | Highest requested product; transcript is default |
| `--asr`, `--ocr`, `--vision` | Override one capability selector |
| `--asr-quality fast|balanced|accurate` | Transcript speed/quality intent; balanced is default |
| `--media-quality auto|fast|balanced|high` | URL visual media policy |
| `--source-assets consumed|complete` | Retain used roles or every reported source role |
| `--max-runtime SECONDS` | Hard 1..86400 second wall-clock limit; expiry exits 124 |
| `--profile-json FILE` | Stream bounded JSONL phase events to a diagnostic file |
| `--start SECONDS`, `--end SECONDS` | Transcribe only the selected absolute interval |
| `--offline` | Deny network routes |
| `--overwrite` | Refresh/rebuild rather than reuse admitted work |
| `--cache-dir DIR` | One-run base for `source/` and `models/` |
| `--config FILE` | Select one TOML file explicitly |
| `--chunk-max-chars`, `--chunk-max-seconds` | Chunk limits |
| `--verbose`, `--debug`, `--log-file` | Diagnostics |

Examples:

```console
vctx prepare captions.srt --out pack
vctx prepare lecture.mp4 --out pack --to evidence
vctx prepare URL --out pack --to summary --max-runtime 1800 --config docs/examples/local-full.toml
vctx prepare part-1.vtt part-2.vtt --out course
```

The default `consumed` scope retains source media/subtitles acquired by product
work. `complete` acquires every audio, video, or native-subtitle capability
reported for the admitted finite revision; one combined representation covers
audio and video. Multiple inputs get independent lanes and are never combined
into one summary. Updating a verified pack adds new lanes, reuses satisfied
matching revisions, and fetches only missing roles when raising a lane to
complete. A changed revision conflicts unless `--overwrite` is explicit.
Publication swaps one complete filesystem generation. Unknown or corrupt
existing output is refused, including with `--overwrite`.

When `--max-runtime` expires, vctx terminates the worker process tree and prints
the stable `deadline_exceeded` category. A prior complete pack is preserved; an
interrupted private stage is never published.

Expected negative outcomes may still publish useful partial products. The
manifest status and per-product outcomes distinguish ready, partial, unavailable,
and failed work.

### Metadata

```console
vctx metadata INPUT [--json] [--offline] [--config FILE] [--cache-dir DIR]
```

Uses the same source admission, sanitization, config, and cache policy as
`prepare`, but creates no pack.

### Render

```console
vctx render PACK --format context|read|transcript [--source KEY] [--out FILE]
```

Without `--out`, content goes to stdout. A multi-source pack requires
`--source`. In-pack projections use relative links stable under pack relocation.
An external output file remains outside the immutable pack.

### Verify

```console
vctx verify PACK
```

Hashes every listed byte, rejects unlisted files/links, validates known typed
products, and checks cross-product citations. Opening a pack for targeted render
verifies only the required canonical dependency closure.

### Doctor and prompt

```console
vctx doctor [--to TARGET] [--offline] [--json]
vctx prompt
```

Doctor is read-only. It reports selected config origin/path, resolved source and
model cache paths, installed profile, target, retention, and capability readiness.
It does not create caches, call the network, pull models, or reveal secrets.

`prompt` prints a static, terse operating contract for agents. Exact syntax stays
in `--help`; observed run facts stay in `manifest.json`.

### Models

```console
vctx models pull [asr] [ocr] [--refresh] [--max-runtime SECONDS] [--json]
vctx models status [asr] [ocr] [--json]
vctx models verify [asr] [ocr] [--json]
vctx models prune [--incomplete] [--unreferenced] [--dry-run] [--json]
```

All accept `--config`, `--cache-dir`, and `--asr`. Pull reuses an exact ready
model without network or content reads; `--refresh` explicitly downloads again,
and `--max-runtime` bounds the Hub child. Status reads per-model receipts; verify
streams model contents and refreshes trusted file facts. Prune removes only
inactive incomplete workspaces or immutable generations not referenced by any
valid receipt; `--dry-run` reports the same admitted targets without deletion.

### Source cache

```console
vctx cache status [--json]
vctx cache prune [--dry-run] [--age 30d | --all] [--json]
```

Both accept `--config` and `--cache-dir`. Plain prune removes orphan blobs and
temporary files. `--age` retires old records; `--all` retires every record.
Source-cache commands never touch model storage.

Local, cached, and remote subtitle inputs are limited to 8 MiB of encoded text;
the source boundary refuses larger input before decoding it.

### OpenRouter authentication

```console
vctx auth openrouter login [--headless]
vctx auth openrouter status
vctx auth openrouter logout
```

This owns the reserved `keyring:openrouter` credential for the automatic
OpenRouter recipe. A named compatible endpoint uses its own `env:NAME` or
`keyring:NAME`; a credential reference never selects an endpoint.

The automatic route checks `OPENROUTER_API_KEY` first and then
`keyring:openrouter`. When admitted, it selects OpenRouter's free route with the
zero-data-retention and required-parameter policy. Login is therefore one-time
authentication for zero-TOML AI use, not a credential-free service. `logout`
removes only the reserved OpenRouter keyring entry; it does not alter environment
variables or named endpoint credentials.

## Configuration

Exactly one file is selected; files are never merged and parent directories are
not searched:

```text
explicit --config
  > ./vctx.toml
  > VCTX_CONFIG
  > platform user config directory / vctx/config.toml
  > built-in defaults
```

Request/CLI fields override the selected file, which overrides built-ins.
`.vctx.toml`, `VCTX_CACHE_DIR`, and arbitrary environment field overlays are not
supported.

### Path resolution

| Path source | Base |
| --- | --- |
| Relative CLI path (`--config`, `--cache-dir`, output) | Current working directory |
| Relative selected-file value | Selected config file directory |
| Platform default cache/config | Platformdirs location |

`--cache-dir CACHE` supplies `CACHE/source` and `CACHE/models`. Without it,
`cache.source_dir` and `cache.model_dir` independently override their defaults.

### Field reference

Every section and field is optional. Unknown sections, fields, enum values, and
references to missing instances are errors. The runnable files under
[`docs/examples/`](examples/README.md) exercise the same strict loader as the CLI.
Tables list every closed string value. A field explicitly described as an open
string is not an enum and is validated by its owning provider or adapter.

#### Runtime, cache, and source

| Field | Type/default | Behavior |
| --- | --- | --- |
| `runtime.offline` | boolean, `false` | Denies source and AI network routes. Cached and local work may continue. CLI `--offline` enables it for one invocation. |
| `runtime.env_files` | path list, `[]` | Loads credential variables from these files. Relative paths use the config directory. Values already present in the process environment take precedence. |
| `cache.source_dir` | path, platform default | Stores remote-source metadata and content-addressed blobs. |
| `cache.model_dir` | path, platform default | Stores explicitly pulled ASR/OCR models and integrity receipts. |
| `source.media_quality` | `auto`, `fast`, `balanced`, or `high`; `auto` | Selects URL video up to 720p, 480p, 720p, or 1080p respectively. It is a preference, not a byte limit; `auto` may fall back to `fast` when cache space is insufficient. |
| `source.yt_dlp.session` | `none`, `browser:NAME`, or `cookies-file:PATH`; `none` | Selects no credentials, reads cookies from a supported browser, or reads an explicit cookie file. These are all accepted forms. |
| `source.yt_dlp.network` | `direct` or `proxy:URL`; `direct` | Selects direct source access or routes yt-dlp operations through the given proxy. |
| `source.yt_dlp.playlist` | `default` or `items:SPEC`; `default` | Uses provider-default playlist behavior or an yt-dlp item selection such as `items:1-3,7`. |
| `source.yt_dlp.subtitle_languages` | string list, `[]` | Ordered subtitle-language preference. Empty uses provider/default selection. |

`--cache-dir CACHE` overrides both cache fields as `CACHE/source` and
`CACHE/models`. It does not alter the persistent config.

#### Pipeline policy and output

| Field | Type/default | Behavior |
| --- | --- | --- |
| `transforms.asr.use` | selector, `auto` | Chooses speech recognition when a usable native subtitle is unavailable. ASR is eligible for every target. |
| `transforms.asr.quality` | `fast`, `balanced`, or `accurate`; `balanced` | Chooses an already prepared managed model. It never downloads one. |
| `transforms.asr.enabled` | strict boolean, inferred | Advanced explicit gate. `false` forces `use = "none"`; `true` cannot be combined with `none`. |
| `evidence.planner.use` | selector, `auto` | Chooses the AI transcript-to-frame-request planner. Eligible for evidence and summary targets. |
| `evidence.planner.enabled` | strict boolean, inferred | Explicitly gates the planner. |
| `evidence.ocr.use` | selector, `auto` | Chooses frame OCR. Eligible for evidence and summary targets. |
| `evidence.ocr.enabled` | strict boolean, inferred | Explicitly gates OCR. |
| `evidence.vision.use` | selector, `auto` | Chooses AI visual description. Eligible for evidence and summary targets. |
| `evidence.vision.enabled` | strict boolean, inferred | Explicitly gates vision description. |
| `summary.use` | selector, `auto` | Chooses the AI summarizer. Eligible only for the summary target. |
| `summary.language` | string, `native` | Requested summary language. `native` means the dominant transcript language. |
| `output.projections` | set of `context`, `read`; both | Markdown projections published in every source lane. Canonical JSON remains authoritative. |
| `output.chunk_max_chars` | integer, `6000` | Maximum transcript characters per canonical chunk. |
| `output.chunk_max_seconds` | integer or omitted | Optional maximum time span per chunk. Omission disables the time limit. |
| `output.source_assets` | `consumed` or `complete`; `consumed` | Retains only assets acquired by requested products, or acquires every source capability reported for the admitted revision. |

Each evidence policy accepts a terse string, for example `ocr = "none"`, or an
explicit table exposing its `enabled` and `use` fields:

```toml
[evidence.vision]
enabled = true
use = "instance:compatible"
```

The target is the upper pipeline boundary: `transcript` disables evidence and
summary, `evidence` enables the three evidence policies, and `summary` enables
all stages. A specific `none` remains disabled even when its target is enabled.

#### Selectors

| Selector | Meaning |
| --- | --- |
| `auto` | Resolve an admitted route from installed/local state and configured authentication. It never pulls a model. |
| `none` | Explicitly disable the capability and record the resulting omission. |
| `instance:NAME` | Use `[instances.asr.NAME]` for ASR or `[instances.ai.NAME]` for planner, vision, and summary. |
| `local:ID` | Select a named local faster-whisper model; `models pull asr` manages this form. |
| `path:PATH` | Select an existing local ASR model path. Relative paths in config use the config directory. |
| `hf:REPO` | Pass an explicit Hugging Face ASR model reference; it is not managed by `models pull`. |

`--asr`, `--ocr`, and `--vision` use the same selector grammar and override the
selected file. Planner and summary remain config-controlled.

#### Expert ASR compatibility

```toml
[transforms.asr]
use = "path:C:/models/faster-whisper-custom"
```

Named instances and explicit model references remain an expert compatibility path.
Execution device, compute mode, threading, and batching are selected internally
and recorded in transcript provenance; they are not configuration surfaces.

#### OpenAI-compatible AI instances

```toml
[instances.ai.compatible]
base_url = "https://provider.example/v1"
model = "model-name"
credential = "env:VCTX_AI_API_KEY"
format = "auto"
timeout_s = 120
insecure = false
```

| Field | Type/default | Behavior |
| --- | --- | --- |
| `base_url` | required URL | Absolute OpenAI-compatible `/v1` root without query or fragment. vctx calls `/chat/completions`. |
| `model` | required open string | Provider-defined model identifier sent unchanged with every request; it is not a vctx enum. |
| `credential` | `env:NAME` or `keyring:NAME` | Resolves a secret at call time. Required for non-loopback endpoints; never selects the endpoint itself. |
| `format` | `auto`, `schema`, `json`, or `prompt`; `auto` | Structured-output strategy. `auto` negotiates/falls back; fixed modes require that provider behavior. |
| `timeout_s` | integer `1..900`, `120` | Request timeout. Vision calls allow at least 180 seconds. |
| `insecure` | boolean, `false` | Must be true to admit cleartext HTTP away from loopback. HTTPS and loopback HTTP need no exception. |

The built-in OpenRouter recipe is separate from named instances. `vctx auth
openrouter login` stores the reserved `keyring:openrouter` secret; `auto` first
considers `OPENROUTER_API_KEY`, then that keyring entry, and applies the free,
zero-data-retention route policy. Use a separately named credential for any
other base URL, even when the underlying account is also OpenRouter.

Secret values never belong in TOML, logs, manifests, doctor, or prompt output.
Evidence planning preserves source language, treats transcript text as untrusted
data, anchors claims to supplied segment IDs, and requests frames only when they
provide materially useful visible evidence.

## Storage

### Cache

```text
cache base/
  source/
    index.sqlite3
    blobs/<sha256>
    tmp/
      asr/                    # disposable bounded interval inputs
  models/
```

SQLite maps sanitized locator identities and exact media request profiles to
content digests. Blob names own cached bytes. Local inputs bypass the source
cache. The final retained media file is an independent portable copy: cache and
output never share mutable file identity. No silent eviction occurs.

### Pack

```text
PACK/
  manifest.json
  <source-key>/
    audio.<ext>            # acquired audio-only representation
    video.<ext>            # acquired video-only representation
    media.<ext>            # one combined audio/video representation
    subtitle.<lang>.<ext>  # acquired native subtitle
    metadata.json
    transcript.json
    chunks.json
    evidence-plan.json       # when planned
    evidence.json            # when produced
    summary.json             # when produced
    context.md               # selected projection
    read.md                  # selected projection
    frames/frame-0001.png    # when captured
```

Only `manifest.json` and its direct source lanes are at the pack root.
`manifest.json` is the sole source index; each source path is its stable key.
Every artifact reference is relative, portable, size/digest indexed, and owned by
one source. Repeated prepares aggregate independent lanes, not their content.

`transcript.json`, `chunks.json`, `evidence-plan.json`, `evidence.json`, and
`summary.json` are canonical typed products. Markdown is reproducible projection.
Summary citations resolve to transcript segments and evidence captures; captures
resolve to admitted plan requests and listed frame files.

Closed manifest string values are:

| Field | Values |
| --- | --- |
| `schema_version` | `5` for new packs; immutable schemas `3` and `4` remain readable |
| `tool` | `vctx` |
| manifest/source `status` | `ok`, `partial`, `error` |
| source `kind` | `url`, `file` |
| source `freshness` | `immutable`, `observed-online`, `unverified-offline` |
| source `asset_scope` | `omitted`, `consumed`, `complete` |
| source `source_capabilities` | a bounded set of `audio`, `video`, `subtitle`, or unknown |
| outcome `status` | `ready`, `partial`, `unavailable` |

Artifact `kind`, product name, requested target, effect operation/status, model,
route, and provider are bounded open strings rather than enums. Their observed
values remain inspectable without making provider extensions a schema change.

## Failure and effect policy

| Code | Meaning |
| --- | --- |
| 0 | Requested operation completed |
| 2 | CLI/config usage error |
| 3 | Batch/source conflict or partial failure |
| 4 | Transcript/source unavailable |
| 5 | Output, cache, integrity, or filesystem failure |
| 6 | Offline policy rejected an uncached source |
| 7 | Provider/model capability unavailable or failed |
| 130 | Cancellation |

Network/model/source effects are explicit and recorded. Cache publication may
fail soft when source work can continue; required output retention may not.
Cancellation and publication failure preserve the prior complete pack. Recovery
uses manifest/run identity rather than directory naming hints.

## Stability

Stable surfaces are command behavior, config grammar/precedence, schema-3/4/5 pack
layout, canonical product schemas, relative artifact references, and exit
categories. Internal Python ownership and human-readable prose may evolve.
