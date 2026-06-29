# Transform graph

`src/vctx/transforms/` owns bounded source-preparation transforms.

Roles:

```text
route metadata/planning
pure product transforms
provider leaves
executor composition
```

## Module map

| Module | Role | Public surface |
| --- | --- | --- |
| `ai_routes.py` | serializable AI route metadata | `AiRoute`, `AiResult`, `resolve_openrouter_ai_route` |
| `model_resolution.py` | model refs + OpenRouter registry | `ModelRef`, `ResolvedModelRoute`, `parse_model_ref`, `resolve_model_ref`, `choose_openrouter_model` |
| `planning.py` | ASR route planning | `RoutePlan`, `SourceState`, `TransformEnvironment`, `plan_asr` |
| `asr.py` | ASR leaves | `run_asr`, `FasterWhisperAsrAdapter`, `OpenAiCompatibleAsrAdapter` |
| `text_ai.py` | typed text-model supplements | `OpenAiCompatibleTextAdapter` |
| `knowledge_flow.py` | flow extraction/merge | `extract_knowledge_flow`, `merge_knowledge_flow_supplement` |
| `visual_cases.py` | visual case extraction/merge | `deterministic_essential_cases`, `uncertain_visual_segments`, `merge_essential_case_supplement` |
| `visual_planning.py` | pure motive planner | `VisualAction`, `VisualAssessment`, `plan_visual_motives`, `visual_motives_from_cases` |
| `visual_routes.py` | executable visual action discovery | `discover_visual_actions`, `rapidocr_available` |
| `visual_frames.py` | ffmpeg frames | `extract_frames` |
| `visual_ocr.py` | OCR leaf | `RapidOcrAdapter` |
| `visual_vlm.py` | VLM leaf | `VlmOutcome`, `OpenAiCompatibleVisionAdapter` |
| `visual_execute.py` | visual executor | `run_visual_context` |
| `visual_evidence.py` | scoring/satisfaction | `score_visual_records`, `score_visual_record` |

## Dependency rules

Allowed:

```text
app -> transforms
transforms -> models/transcript/config
provider leaves -> net
```

Forbidden:

```text
pure transforms -> provider clients/network
transforms -> render/io final artifact writer/cli
models -> transforms
AiRoute -> HTTP/session runtime
```

## Visual pipeline

```text
Transcript
  -> deterministic_essential_cases(...)
  -> uncertain_visual_segments(...)
  -> optional text_ai.essential_case_supplement(...)
  -> merge_essential_case_supplement(...)
  -> visual_motives_from_cases(...)
  -> discover_visual_actions(...)
  -> plan_visual_motives(SourceAccess, motives, actions)
  -> run_visual_context(plan.assessment, media, out_dir)
  -> score_visual_records(records, transcript, motives)
```

## Visual data contracts

```text
SourceAccess              transcript/audio/video bits
EssentialVisualCase       transcript-anchored visual need
VisualOperationMotive     operation + reason + segment
VisualAction              executable action: sample/ocr/describe/capture
VisualAssessment          recipe + rationale
ValidVisualPlan           executable plan
SkippedVisualPlan         no-op with reason
VisualRecordSet           evidence records only
VisualScoreReport         satisfaction diagnostics only
VlmOutcome                per-frame VLM result or error
```

## Artifact mapping

```text
VisualRecordSet      -> visual_records.json
VisualScoreReport    -> visual_scores.json
capture artifact refs -> visual/frames/*.png
manifest step        -> transform.visual_satisfaction
```

## Current visual rules

- Deleted: `VisualSourceSignals`, `plan_visual_acquisition`.
- No score-driven visual fetch.
- No transitional `list[str]` VLM compatibility.
- `describe_many(...)` returns `list[VlmOutcome]`.
- Uncertain LLM input is bounded automatically:

```text
explicit max_segments wins
else min(40, max(8, round(total_segments * 0.2)))
```

- Non-English/mixed/unknown non-Latin scripts are selected automatically.

## Text supplement graph

```text
Transcript
  -> OpenAiCompatibleTextAdapter.knowledge_flow_supplement(...)
  -> merge_knowledge_flow_supplement(...)
```

```text
Transcript subset
  -> OpenAiCompatibleTextAdapter.essential_case_supplement(...)
  -> merge_essential_case_supplement(...)
```

Supplements must stay transcript-anchored. Unsupported segment ids are discarded.
