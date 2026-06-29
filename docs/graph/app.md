# App graph

## Owned modules

```text
vctx.cli
vctx.config
vctx.app.prepare
vctx.app.result
vctx.app.credentials
vctx.app.metadata
vctx.app.chunk
vctx.app.render
vctx.app.doctor
```

## Purpose

Own CLI intent, config resolution, workflow order, manifest truth.

No provider payload normalization. No Markdown rendering internals.

## `prepare` call graph

```text
cli.prepare_command
  -> PrepareRequest
  -> prepare_context_pack(request)
      -> resolve_config(request)
      -> validate output policy
      -> detect source adapter
      -> metadata
      -> deterministic transcript payload
      -> media + ASR fallback if transcript missing and allowed
      -> parse + normalize transcript
      -> chunk transcript
      -> visual cases / plan / execute when useful
      -> knowledge flow
      -> render_artifact_bundle(...)
      -> write artifacts
      -> write manifest
      -> PrepareResult
```

## Config shape

```text
PrepareRequest
  input
  out_dir
  overwrite
  workflow
  offline
  config_path
  cache_dir
  chunk budgets
```

```text
ResolvedConfig
  runtime
  source
  transforms
  providers
  instances
  output
```

## Config file selection

```text
request/CLI
  -> --config
  -> ./vctx.toml
  -> ./.vctx.toml
  -> VCTX_CONFIG
  -> user config
  -> defaults
```

Only one config file is selected.

## Route policy

```text
missing/null -> default
false        -> disabled
auto/default -> inspect source + installed tools + policy
explicit     -> use or fail clearly
```

## Manifest rule

Every branch records a step:

```text
source.detect
metadata.extract
transcript.extract
transform.asr
transcript.parse
transcript.normalize
chunk
transform.visual_plan
source.media
transform.visual_satisfaction
transform.visual_capture
```

Optional skipped/unavailable paths are explicit; secrets are never recorded.
