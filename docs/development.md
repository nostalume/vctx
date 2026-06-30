# Development

## Setup

```bash
uv sync
uv sync --extra full   # ASR + visual extras when needed
```

Use `uv run ...` for all project commands.

## Daily checks

Focused first:

```bash
uv run pytest tests/test_visual_cases.py tests/test_visual_acquisition_planning.py -q
uv run pytest tests/test_prepare_visual_ocr.py tests/test_prepare_visual_capture.py tests/test_prepare_visual_vlm.py -q
```

Full gate:

```bash
uv run python scripts/check_module_layout.py
uv run ruff check .
uv run ty check .
uv run pytest -q -W error::DeprecationWarning
```

Whitespace/path gate before commit:

```bash
git diff --check -- docs src tests scripts pyproject.toml uv.lock
```


## GitHub Actions CI

The repository CI is defined in:

```text
.github/workflows/ci.yml
```

It uses the official uv setup action instead of hand-installing uv:

```yaml
- uses: astral-sh/setup-uv@v7
  with:
    python-version: "3.14"
    enable-cache: true
```

CI runs on pushes to `main` and pull requests:

```bash
uv sync --locked --dev
uv run python scripts/check_module_layout.py
uv run ruff check .
uv run ty check .
uv run pytest -q -W error::DeprecationWarning
uv build
```

Keep this aligned with the local full gate. If local verification changes, update CI in the same slice.

## Publishing to PyPI

The publish workflow is defined in:

```text
.github/workflows/publish.yml
```

It runs on:

```text
published GitHub releases
version tags matching v*
```

The publish job repeats the quality gate, builds distributions, then publishes with uv:

```bash
uv build
uv publish --trusted-publishing always
```

### Recommended PyPI authentication: trusted publishing

This repo is configured for PyPI trusted publishing through GitHub OIDC. The workflow grants:

```yaml
permissions:
  contents: read
  id-token: write
```

No PyPI API token is needed when trusted publishing is configured on PyPI.

Configure it in PyPI:

1. Open the PyPI project page for `vctx`.
2. Go to project settings / publishing.
3. Add a trusted publisher for GitHub Actions.
4. Use this repository owner/name.
5. Use workflow filename:

```text
publish.yml
```

6. Use environment name:

```text
pypi
```

The GitHub workflow job also declares:

```yaml
environment: pypi
```

If the PyPI project does not exist yet, create it with the first release path PyPI currently supports for trusted publishers, or publish once with an API token and then add the trusted publisher.

### Alternative: PyPI API token secret

Only use this if trusted publishing is unavailable.

1. Create a PyPI API token, preferably scoped to the `vctx` project.
2. In GitHub, open repository settings.
3. Go to Secrets and variables -> Actions.
4. Add a repository secret such as:

```text
PYPI_API_TOKEN
```

5. Change the publish step to:

```yaml
- name: Publish distributions to PyPI
  env:
    UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
  run: uv publish
```

Do not commit PyPI tokens, `.pypirc`, or secret values.

## Git hygiene

Commit code/docs/tests only.

Do not commit local scratch:

```text
docs/report/
.tmp/
.env
.env.*
```

Before staging:

```bash
git status --short --untracked-files=all
```

Stage explicit paths. Avoid broad `git add -A` unless you immediately inspect staged files.

## Architecture workflow

When changing docs or module boundaries:

1. Derive the live module/API shape from `src/vctx`.
2. Keep docs terse:
   - `context.md`: intent and rules.
   - `architecture.md`: boundaries.
   - `api.md`: CLI/config/artifacts.
   - `graph/*.md`: concrete module/API graph.
3. Do not rewrite historical `docs/report/*` into authority.
4. Keep graph docs aligned with current code.

## Visual pipeline workflow

Current contract:

```text
transcript
  -> cases
  -> motives
  -> executable actions
  -> records
  -> scores
  -> manifest
```

Never reintroduce:

```text
VisualSourceSignals
plan_visual_acquisition
list[str] VLM outcome compatibility
```

Expected artifacts:

```text
visual_records.json   evidence only
visual_scores.json    satisfaction diagnostics only
visual/frames/*.png   frame artifacts
manifest.json         warnings/status
```

## Real side-effect integration test

This test uses the fixed TED URL and real side effects.

Requires:

```bash
OPENROUTER_API_KEY=...   # do not print it
ffmpeg on PATH
network access
```

Run:

```bash
VCTX_RUN_TED_VISUAL_INTEGRATION=1 \
uv run pytest tests/integration/test_fixed_ted_visual_side_effects.py::test_fixed_ted_visual_workflow_preserves_pipeline_invariants -q -s
```

Expected:

```text
1 passed
```

The test validates invariant correctness:

- manifest status is `ok`
- frame artifacts exist
- visual refs render in `context.md`
- frame Markdown renders in `readable.md`
- artifact paths are relative
- credentials/secrets are not leaked
- native OCR text, when present, is preserved

## Real output quality check

After the real integration, find the temp output:

```bash
python - <<'PY'
import tempfile
from pathlib import Path
root = Path(tempfile.gettempdir())
for p in sorted(root.rglob('fixed-ted-visual/manifest.json'), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
    print(p.parent)
PY
```

Inspect artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path
out = Path(r'PASTE_OUTPUT_DIR')
manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
records = json.loads((out/'visual_records.json').read_text(encoding='utf-8'))
scores = json.loads((out/'visual_scores.json').read_text(encoding='utf-8'))
context = (out/'context.md').read_text(encoding='utf-8')
readable = (out/'readable.md').read_text(encoding='utf-8')
frames = sorted((out/'visual'/'frames').glob('*.png'))
print('status', manifest['status'])
print('warnings', manifest['warnings'])
print('artifacts', [a['path'] for a in manifest['artifacts']])
print('record kinds', [r['kind'] for r in records['records']])
print('scores', [(s['operation'], s['reason'], s['status']) for s in scores['satisfaction']])
print('frames', [(p.name, p.stat().st_size) for p in frames])
print('context visual refs', context.count('<visual_ref'))
print('readable frames', readable.count('![Frame '))
PY
```

Quality checklist:

- `visual_records.json` has only `records`.
- `visual_scores.json` has satisfaction checks.
- all satisfaction checks are expected for the motive.
- missed satisfaction has a manifest warning.
- frame files are non-empty.
- descriptions match the frame pixels.
- no output contains secret literals.

For TED shoe tying, expected current shape:

```text
record kinds: description + capture
satisfaction: capture/describe action_demonstration satisfied
warnings: []
frames: 2
```

## Commit checklist

Before commit/push:

```bash
git diff --cached --name-status
git diff --cached | grep '^+' | grep -iE '(api_key|secret|password|token|passwd)\s*=\s*["'"''][^"'"'']{6,}["'"'']' || true
uv run python scripts/check_module_layout.py
uv run ruff check .
uv run ty check .
uv run pytest -q -W error::DeprecationWarning
```
