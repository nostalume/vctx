# Graph docs

These files map docs to the live `src/vctx` module layout.

| Source family | Current modules | Graph doc |
| --- | --- | --- |
| CLI/app workflow | `cli.py`, `app/*`, `config.py` | `app.md` |
| Domain schemas | `models/*`, `transcript.py`, `chunking.py` | `models.md` |
| Source adapters | `sources/*`, `subtitles.py` | this index + `app.md` |
| Transforms | `transforms/*` | `model-transforms.md` |
| AI route metadata | `transforms/ai_routes.py`, `model_resolution.py` | `ai.md` |
| Network runtime | `net.py` | `net.md` |
| Rendering/artifacts | `render/*`, `io.py` | this index + `api.md` |

## Dependency spine

```text
cli
  -> app
      -> sources / transforms / transcript / chunking / render / io
          -> models
      -> provider leaves -> net
```

## Forbidden directions

```text
models -> app/cli/sources/transforms/render/io
render -> sources/transforms/net
sources -> render/transforms
pure transforms -> provider clients/network
cli -> provider libraries
```

## Authority split

- `docs/context.md`: why and rules.
- `docs/architecture.md`: durable boundaries.
- `docs/api.md`: CLI/config/artifacts.
- `docs/graph/*.md`: current module/API correspondence.
- `docs/report/*`: local/historical development notes; not commit authority.
