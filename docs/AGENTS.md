# vctx

## Project goal

`vctx` is a one-shot, CLI-first context compiler. It converts source media and
transcripts into inspectable context packs for people and downstream agents.
The output directory is the product and stable integration surface; canonical
JSON artifacts are projected into readable Markdown.

It is not a chat UI, Q&A system, RAG/vector database, personal knowledge base,
cross-video memory, or web/desktop backend.

## Tech stack

- Python 3.14 package and Typer CLI, built with Hatchling and managed with uv.
- Typed domain and artifact schemas with Pydantic; HTTP through httpx; URL
  source/subtitle/media acquisition through yt-dlp.
- Deterministic text processing for subtitle parsing and chunking.
- Optional local transforms: faster-whisper for ASR; RapidOCR plus ONNX Runtime
  for frame OCR. Optional OpenAI-compatible/OpenRouter routes provide text and
  vision transforms when explicitly configured.
- Visual frame extraction currently uses the host `ffmpeg` executable.
- Quality gate: Ruff, ty, pytest, behavior-suite LOC budget, then distribution build.

## Principles

- Keep the CLI first and the base install small; make heavy/local capabilities
  optional and explicit.
- Prefer deterministic source data. Model calls are source-preparation
  transforms, never hidden assistant behavior.
- Make every selected route, warning, effect, and artifact visible in
  `manifest.json`.
- Keep provider payloads at adapters and use typed internal models across
  boundaries.
- Preserve dependency direction: `cli -> app -> source/transforms/render/io ->
  models`; render does not acquire sources or call providers, and models do not
  depend on higher layers.
- The visual path is motive-led: acquire media only for transcript-anchored
  visual evidence; store evidence in `visual_records.json` and satisfaction
  diagnostics in `visual_scores.json`.
