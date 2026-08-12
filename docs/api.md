# vctx CLI and artifact contract

`vctx` is a one-shot context compiler. Its stable integration surfaces are the
installed CLI and schema-3 output pack. Python modules are internal.

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
| full | Normalized union of ASR and visual |

PyAV decodes video in-process. No host FFmpeg executable is required. Normal
`prepare` never downloads models; use `models pull` explicitly.

## Commands

Use command-specific `--help` for the exact option grammar.

### Prepare

```console
vctx prepare INPUT... --out PACK [OPTIONS]
```

Important options:

| Option | Meaning |
| --- | --- |
| `--to transcript|evidence|summary` | Highest requested product; transcript is default |
| `--asr`, `--ocr`, `--vision` | Override one capability selector |
| `--media-quality auto|fast|balanced|high` | URL visual media policy |
| `--no-retain-media` | Make output depend on external/cache media |
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
vctx prepare URL --out pack --to summary --config examples/vctx.toml
vctx prepare part-1.vtt part-2.vtt --out course
```

The default retains source media/subtitles when available. Multiple inputs get
independent lanes and are never combined into one summary. Updating a verified
pack adds new lanes, reuses satisfied matching revisions, and replaces only
changed/upgraded lanes. Publication swaps one complete filesystem generation.
Unknown or corrupt existing output is refused, including with `--overwrite`.

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
vctx models pull [asr] [ocr] [--json]
vctx models status [asr] [ocr] [--json]
vctx models verify [asr] [ocr] [--json]
```

All accept `--config`, `--cache-dir`, and `--asr`. Pull is the only normal model
download path. Status reads receipts; verify hashes prepared model contents.

### Source cache

```console
vctx cache status [--json]
vctx cache prune [--dry-run] [--age 30d | --all] [--json]
```

Both accept `--config` and `--cache-dir`. Plain prune removes orphan blobs and
temporary files. `--age` retires old records; `--all` retires every record.
Source-cache commands never touch model storage.

### OpenRouter authentication

```console
vctx auth openrouter login [--headless]
vctx auth openrouter status
vctx auth openrouter logout
```

This owns the reserved `keyring:openrouter` credential for the automatic
OpenRouter recipe. A named compatible endpoint uses its own `env:NAME` or
`keyring:NAME`; a credential reference never selects an endpoint.

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

### Current TOML surface

See the strict runnable files under `examples/`. Unknown fields are errors.

```toml
[runtime]
offline = false
env_files = [".env"]

[cache]
source_dir = ".cache/vctx/source"
model_dir = ".cache/vctx/models"

[source]
media_quality = "auto"

[source.yt_dlp]
session = "none"              # browser:NAME or cookies-file:PATH
network = "direct"            # or proxy:URL
playlist = "default"          # or items:SPEC
subtitle_languages = ["en"]

[transforms.asr]
use = "instance:local-default"

[evidence]
planner = "auto"
ocr = "auto"
vision = "auto"

[summary]
use = "auto"
language = "native"

[output]
projections = ["context", "read"]
chunk_max_chars = 6000
chunk_max_seconds = 900
retain_media = true

[instances.asr.local-default]
type = "local-faster-whisper"
model = "small"
device = "auto"
compute = "auto"
cache = "persistent"

[instances.ai.compatible]
base_url = "https://provider.example/v1"
model = "model-name"
credential = "env:VCTX_AI_API_KEY"
format = "auto"
timeout_s = 120
```

Selectors are `auto`, `none`, `instance:NAME`, or capability-supported model
references such as `path:...`, `local:...`, and `hf:...`. Named AI endpoints are
OpenAI-compatible `/v1` roots; vctx calls `/chat/completions`. Remote HTTP is
rejected unless explicitly admitted as insecure. Secret values never belong in
TOML, logs, manifests, doctor, or prompt output.

`language = "native"` means the dominant transcript language. Evidence planning
preserves source language, treats transcript text as untrusted data, anchors all
claims to supplied segment IDs, and requests frames only for materially useful
visible evidence.

## Storage

### Cache

```text
cache base/
  source/
    index.sqlite3
    blobs/<sha256>
    tmp/
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
    metadata.json
    transcript.json
    chunks.json
    evidence-plan.json       # when planned
    evidence.json            # when produced
    summary.json             # when produced
    context.md               # selected projection
    read.md                  # selected projection
    subtitle.<lang>.<ext>    # retained when available
    media.<ext>              # retained when available
    frames/
      frame-0001.png         # when captured
```

Only `manifest.json` is at the pack root. Each source lane is a direct child.
Every artifact reference is relative, portable, size/digest indexed, and owned by
one source. Repeated prepares aggregate independent lanes, not their content.

`transcript.json`, `chunks.json`, `evidence-plan.json`, `evidence.json`, and
`summary.json` are canonical typed products. Markdown is reproducible projection.
Summary citations resolve to transcript segments and evidence captures; captures
resolve to admitted plan requests and listed frame files.

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

Stable surfaces are command behavior, config grammar/precedence, schema-3 pack
layout, canonical product schemas, relative artifact references, and exit
categories. Internal Python ownership and human-readable prose may evolve.
