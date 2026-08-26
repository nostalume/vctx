# Configuration examples

Every TOML file here passes the same strict config loader as the CLI. Copy one
to `./vctx.toml`, pass it with `--config`, or copy only the sections you need.
Files are selected one at a time and are never merged.

| File | Purpose |
| --- | --- |
| `minimal.toml` | Zero-TOML defaults; immediately useful for subtitle transcripts |
| `subtitles-only.toml` | Core install with model-backed transforms disabled |
| `local-full.toml` | Local faster-whisper ASR, PyAV frames, and RapidOCR |
| `openrouter.toml` | Automatic free/ZDR OpenRouter route after login |
| `openai-compatible.toml` | Named OpenAI-compatible text and vision endpoint |
| `private-source.toml` | Browser cookies, proxy, playlist slice, and subtitle priority |

```console
vctx doctor --config docs/examples/local-full.toml --to evidence --json
vctx models pull asr ocr --config docs/examples/local-full.toml
vctx prepare VIDEO --out pack --to summary --config docs/examples/local-full.toml
```

Relative paths inside a config are resolved from that config's directory. Never
put secret values in TOML: use `env:NAME`, `keyring:NAME`, or `vctx auth
openrouter login`.

`minimal.toml` does not supply AI credentials. To use it with `--to summary`,
first run `vctx auth openrouter login` or set `OPENROUTER_API_KEY`. The
`openrouter.toml` file makes the automatic AI intent visible, but authentication
still remains outside TOML.
