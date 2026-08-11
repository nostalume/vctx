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
- Visual frame extraction uses PyAV 18 and Pillow in-process; no host media
  executable is required.
- Quality gate: Ruff, ty, pytest, behavior-suite LOC budget, then distribution build.

## Principles

- Keep the CLI first and the base install small; make heavy/local capabilities
  optional and explicit.
- Prefer deterministic source data. Model calls are source-preparation
  transforms, never hidden assistant behavior.
- Make every selected route, warning, effect, and artifact visible in
  `manifest.json`.
- Keep deterministic selection separate from effects. Command-scoped runtimes
  own HTTP lifetime; callers declare retry semantics and `net.py` executes them.
- Keep provider payloads inside `ai.py`; credential locators, instance identity,
  and request policy are independent typed values.
- Preserve dependency direction: `cli -> app -> affiliated source/asr/visual
  capabilities -> artifact/render/io`; render does not acquire sources or call
  providers, and capability modules do not depend on app composers.
- The visual path is plan-led: a language-neutral AI result may request only
  transcript-anchored frames; code validates anchors and derives timestamps
  before media acquisition.
