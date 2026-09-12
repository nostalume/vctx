# Changelog

This file records user-visible changes to vctx. The current CLI and pack contract
is documented in [docs/api.md](docs/api.md).

## 0.4.1 — 2026-09-12

### Fixed

- This is the first published 0.4 release and includes every 0.4.0 change below.
  The v0.4.0 tag uploaded no package; publication now uses the pinned uv Trusted
  Publishing path already proven by v0.3.0.

## 0.4.0 — 2026-09-12

### Added

- Managed ASR and OCR model lifecycle commands with explicit pull, status,
  verification, refresh, deadline, and safe-prune behavior.
- Automatic GPU-first faster-whisper execution with bounded CPU fallback. The
  `asr-cuda` extra supplies project-local CUDA libraries on Windows.
- Transcript quality intent (`fast`, `balanced`, or `accurate`), persistent exact
  ASR reuse, bounded media intervals, and execution profiling receipts.
- Anonymous first-party Bilibili BV admission, resumable source acquisition, and
  hard wall-clock bounds for preparation.
- `consumed` and `complete` source-asset scopes. A complete same-revision request
  fetches only missing roles and preserves existing products.

### Changed

- New packs use schema 5. `manifest.json` and direct source lanes are at the pack
  root; retained `audio`, `video`, `media`, and `subtitle` files live beside their
  source's products. Schemas 3 and 4 remain readable.
- ASR device, compute mode, threads, and batching are selected internally and
  recorded as run facts rather than exposed as normal configuration.
- Source and model caches now preserve recoverable partial acquisition state and
  publish only verified complete objects.
- Preparation refuses changed source revisions unless `--overwrite` is explicit;
  corrupt or ambiguous output is refused even with overwrite.

### Migrating from 0.3

1. Upgrade the tool:

   ```console
   uv tool upgrade vctx
   ```

2. Replace normal `output.retain_media` configuration with an explicit intent:

   ```toml
   [output]
   source_assets = "consumed" # or "complete"
   ```

   The hidden `--no-retain-media` option and `output.retain_media = false` remain
   accepted only as a deprecated compatibility opt-out.

3. Remove `device` and `compute` from named ASR instances. Use
   `transforms.asr.quality` for the user-visible speed/quality choice; vctx chooses
   runtime mechanics and reports the result through `doctor` and pack provenance.

4. Read paths from `manifest.json`. Do not assume schema-3 lane paths when
   consuming newly written packs. vctx 0.4 reads existing schema-3/4 packs and
   replaces them only through a verified prepare generation; vctx 0.3 does not
   understand schema 5, so keep an old pack copy if you must downgrade.

5. Re-running a complete request against the same admitted revision is monotonic.
   To accept a changed remote revision, pass `--overwrite` explicitly.
