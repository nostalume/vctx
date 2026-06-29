# Knowledge Flow Slice Priority Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Move `vctx` from an auditable deterministic evidence/knowledge-flow pack toward a useful video-site knowledge-flow summarizer without adding recursive/bloated validation layers.

**Architecture:** Keep the core pipeline evidence-linked and ergonomic. LLM summaries and non-deterministic model calls are acceptable when they produce a useful surface for a human or downstream AI. The failure mode to avoid is recursive enrichment: adding validation/graph/chapter layers that are hard to consume, beyond `vctx`'s role, or introduced without asking who needs them and how they will be used. Avoid chapters for now. Avoid broad claim-validation/rich-graph features unless they have a clear consumer-facing ergonomic payoff and a small verification surface.

**Tech Stack:** Python, Pydantic models, Typer CLI, pytest, ruff, ty, existing `vctx` transcript/visual/knowledge-flow modules.

---

## Priority Order

### P0 — Plain-Prose Deterministic Knowledge-Flow Extraction

**Why first:** Current `knowledge_flow.json` only extracts explicit arrow chains (`A -> B -> C`). Most real videos explain workflows in prose. This slice gives the biggest usefulness jump with a small and verifiable user-facing contract. It should not block later LLM summary work; it gives that summary a better evidence spine.

**Scope:** Convert simple ordered prose into evidence-linked flow chains.

**In scope deterministic cues:**
- ordinal sequence: `first`, `then`, `next`, `finally`
- numbered steps: `step 1`, `step 2`, `1.`, `2.`
- input/output phrases: `input is`, `output is`, `produces`, `turns X into Y`
- pipeline/workflow phrases: `pipeline is A, B, C`, `workflow goes from A to B to C`

**Out of scope:**
- arbitrary semantic inference inside this slice
- adding graph taxonomies
- broad claim-validation subsystems

**Likely files:**
- Modify: `src/vctx/transforms/knowledge_flow.py`
- Test: `tests/test_knowledge_flow.py`
- Possibly docs: `docs/api.md`, `docs/workflow.md`

**TDD tasks:**
1. Add RED test: `First acquire the URL. Then extract subtitles. Finally summarize.` -> `acquire the URL -> extract subtitles -> summarize` with `seg_000001` evidence.
2. Add RED test: `Step 1: download media. Step 2: transcribe audio. Step 3: extract frames.` -> three-step chain.
3. Add RED test: `The input is video URL. The output is knowledge-flow pack.` -> `video URL -> knowledge-flow pack`.
4. Implement minimal deterministic extractors in `knowledge_flow.py`.
5. Verify focused tests and full gate.

**Acceptance:**
- Produces `knowledge_flow.json` from plain prose without visual evidence.
- Every node/edge cites the segment ID.
- This slice does not require LLM calls.
- No new graph taxonomy.

---

### P1 — Ergonomic LLM Summary, Grounded by Existing Flow/Evidence

**Why second:** The user-facing product is a summary/knowledge-flow explanation, not only JSON. LLM summary is allowed and useful if its role is ergonomic: turn the existing transcript chunks, kept visual records, and `KnowledgeFlow` into a compact human/AI-readable explanation. Do not turn this into recursive validation or rich graph construction.

**Scope:** Add an optional summary artifact/section generated from bounded evidence inputs. If no model is configured, keep deterministic template rendering as fallback.

**Likely files:**
- Modify: `src/vctx/render/knowledge_flow_md.py`
- Modify: `src/vctx/render/context_md.py`
- Modify: `src/vctx/render/readable_md.py`
- Test: `tests/test_render_knowledge_flow.py`

**TDD tasks:**
1. RED test: summary section answers “what is the workflow?” in 3-6 bullets/short paragraphs.
2. RED test: every summary bullet cites source evidence IDs or flow chain evidence.
3. RED test: missing model/config falls back to deterministic summary projection rather than failing the pack.
4. Implement a narrow summary boundary that consumes existing artifacts; do not add new graph layers.
5. Verify full gate.

**Acceptance:**
- Summary improves ergonomics for human/downstream AI reading.
- It does not introduce a validation subsystem.
- It does not invent new unsupported graph semantics.
- LLM output, if used, is bounded by existing evidence artifacts.

---

### P2 — Env-File Credential Presence for `model = "auto"`

**Why third:** Improves real usability without changing core semantics. Current adapter can read `runtime.env_files`, but auto route planning only sees shell env.

**Scope:** Let auto route discovery know that `OPENROUTER_API_KEY` exists in env files without exposing the value.

**Likely files:**
- Modify: `src/vctx/app/credentials.py` or add credential-presence helper
- Modify: `src/vctx/app/prepare.py`
- Test: likely `tests/test_prepare_visual_vlm.py` or new focused credential test

**TDD tasks:**
1. RED test: config with `runtime.env_files = [".env"]` and `OPENROUTER_API_KEY=...` lets `model="auto"` select cached free VLM.
2. Implement `credential_env_presence(...) -> Mapping[str, str]` or equivalent sentinel map.
3. Ensure manifest does not include secret values.
4. Verify full gate.

**Acceptance:**
- `.env`-only OpenRouter key works for route planning.
- Secret value never appears in manifest/artifacts.

---

### P3 — Optional Real-World Smoke Script, Not Default CI

**Why fourth:** Verifies actual video-site behavior without making CI flaky.

**Scope:** Add an opt-in script or documented command that exercises a real URL when env/config is available.

**Likely files:**
- Add: `scripts/smoke_video_knowledge_flow.py` or docs-only smoke section
- Docs: `docs/workflow.md`

**TDD stance:** This is smoke/manual, not default deterministic tests. Keep unit/integration tests mocked.

**Acceptance:**
- Command is opt-in.
- Skips clearly when no URL/network/key.
- Verifies files exist: `context.md`, `readable.md`, `knowledge_flow.json`, `manifest.json`.

---

### P4 — Optional LLM Essential-Case Extraction, Strict Fallback

**Why later:** Useful for better visual sampling, but introduces model variability. Only do after deterministic prose flow is useful.

**Scope:** `transcript -> EssentialVisualCase[]` through configured/free text model, with strict JSON validation and deterministic fallback.

**Guardrails:**
- LLM output is advisory, not authoritative.
- Invalid/missing output falls back to deterministic cases.
- Tests monkeypatch adapter; no live LLM in CI.

**Acceptance:**
- Typed cases only.
- Bad JSON fallback tested.
- No expansion into broad claim validation.

---

## Explicitly Deprioritized / Avoid for Now

### Chapters

**Decision:** Deprioritize. The current goal is knowledge-flow summarization, not chaptering. Chapters add another output family and model route without improving the deterministic evidence spine.

**Revisit only if:** user specifically needs navigation/timestamps as a product output.

### Rich Graph Semantics

**Decision:** Avoid for now.

Concerns:
- Node/edge taxonomies can become subjective.
- Verification can become recursive: feature adds semantics, then verifier needs semantics to validate semantics.
- Likely bloat relative to current deterministic flow needs.

**Allowed minimal invariant:** concept nodes + directed edges + evidence IDs only.

### Broad Claim Validation

**Decision:** Avoid for now.

Concerns:
- Natural-language claim validation is recursively hard.
- It encourages building a second reasoning system to validate the first.
- It may bloat artifacts and tests without deterministic behavior.

**Allowed minimal invariant:** every generated node/edge must cite source evidence, and dropped visual records cannot support nodes/edges. This is already deterministic and testable.

### LLM Summary

**Decision:** Allowed when it has an ergonomic contract.

Good use:
- compress transcript + kept visual records + knowledge flow into a readable explanation
- answer what a human/downstream AI actually needs next
- cite existing evidence IDs/chains

Bad use:
- generate a new recursive claim graph
- introduce broad validation layers before there is a consumer
- add artifacts whose only purpose is to justify other artifacts



## Definition of “Complete Stage”

The current completed stage is:

```text
deterministic auditable evidence/knowledge-flow pack generator
```

It is complete when:
- `prepare` writes transcript/chunk/context artifacts.
- visual workflow can produce scored records when enabled.
- `knowledge_flow.json` exists when transcript/visual evidence contains explicit or simple-prose flow.
- `context.md` and `readable.md` render compact flow chains with evidence IDs.
- full gate passes.

It is not yet:

```text
full semantic video summarizer
```

That stage should be approached through ergonomic user/AI-facing surfaces with bounded evidence inputs, not through broad validation/rich-graph layers added for their own sake.

---

## Next Immediate Slice to Execute

### Slice: Plain-Prose Deterministic Knowledge-Flow Extraction

**Commit message target:**

```bash
git commit -m "feat: extract prose knowledge flow"
```

**Verification command:**

```bash
git diff --check -- docs src tests && uv run ruff check . && uv run ty check . && uv run pytest -q -W error::DeprecationWarning
```

**Expected after completion:**
- Test count increases.
- `knowledge_flow.json` is useful for simple narrated workflows without arrows.
- This slice remains small: no chapters, no rich graph, no claim-validation expansion.
