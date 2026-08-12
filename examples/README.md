# vctx configuration examples

Copy `vctx.toml` into a workspace for local ASR and automatic visual evidence.
Paths in a selected config file are relative to that file. `--cache-dir` is a
one-run base override, resolved from the current directory, and supplies both
`source/` and `models/`.

`openai-compatible.toml` shows a named OpenAI-compatible endpoint. Set its
credential in the environment; never put secret values in TOML.

Inspect the selected file, resolved cache paths, and capability readiness with:

```console
vctx doctor --config examples/vctx.toml --to evidence --json
```
