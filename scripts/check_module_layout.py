from __future__ import annotations

from pathlib import Path

DELETED_MODULES = [
    Path("src/vctx/io/json_dump.py"),
    Path("src/vctx/chunking/tokens.py"),
    Path("src/vctx/sources/base.py"),
    Path("src/vctx/render/context_md.py"),
    Path("src/vctx/render/readable_md.py"),
    Path("src/vctx/render/transcript_md.py"),
    Path("src/vctx/render/knowledge_flow_md.py"),
    Path("src/vctx/models/llm_products.py"),
    Path("src/vctx/models/common.py"),
    Path("src/vctx/models/transcript.py"),
    Path("src/vctx/models/chunks.py"),
    Path("src/vctx/transforms/openrouter_registry.py"),
    Path("src/vctx/transforms/ai.py"),
    Path("src/vctx/app/config.py"),
    Path("src/vctx/app/errors.py"),
    Path("src/vctx/transcript"),
    Path("src/vctx/subtitles"),
    Path("src/vctx/chunking"),
    Path("src/vctx/io"),
    Path("src/vctx/util"),
    Path("docs/workflow.md"),
    Path("docs/graph/chunking.md"),
    Path("docs/graph/cli.md"),
    Path("docs/graph/io.md"),
    Path("docs/graph/manifest.md"),
    Path("docs/graph/render.md"),
    Path("docs/graph/sources.md"),
    Path("docs/graph/subtitles.md"),
    Path("docs/graph/transcript.md"),
]

BANNED_IMPORTS = [
    "vctx.io.json_dump",
    "vctx.chunking.tokens",
    "vctx.sources.base",
    "vctx.render.context_md",
    "vctx.render.readable_md",
    "vctx.render.transcript_md",
    "vctx.render.knowledge_flow_md",
    "vctx.models.llm_products",
    "vctx.models.common",
    "vctx.models.transcript",
    "vctx.models.chunks",
    "vctx.transforms.openrouter_registry",
    "from vctx.transforms.ai import",
    "vctx.app.config",
    "vctx.app.errors",
    "vctx.transcript.clean_text",
    "vctx.transcript.normalize",
    "vctx.subtitles.parse",
    "vctx.subtitles.srt_parser",
    "vctx.subtitles.webvtt_parser",
    "vctx.chunking.chunker",
    "vctx.io.cache",
    "vctx.io.writer",
    "vctx.util.timefmt",
    "vctx.util.versions",
]

BANNED_SNIPPETS = [
    (Path("src/vctx/models/common.py"), "class TimeRange"),
    (Path("src/vctx/transforms/ai_routes.py"), "transcript_cleanup"),
    (Path("src/vctx/transforms/ai_routes.py"), "chapter_suggestion"),
    (Path("src/vctx/transforms/model_resolution.py"), "CLEANUP"),
    (Path("src/vctx/transforms/model_resolution.py"), "CHAPTERS"),
    (Path("src/vctx/transforms/planning.py"), "def plan_cleanup"),
    (Path("src/vctx/transforms/planning.py"), "def plan_chapters"),
    (Path("src/vctx/transforms/planning.py"), "def plan_visual_context"),
    (Path("src/vctx/transforms/planning.py"), "visual_need"),
    (Path("src/vctx/transforms/planning.py"), "installed_ocr"),
    (Path("src/vctx/transforms/planning.py"), "configured_vision"),
    (Path("src/vctx/transforms/planning.py"), "free_online_vision"),
    (Path("src/vctx/transforms/planning.py"), "configured_ocr"),
    (Path("src/vctx/transforms/planning.py"), "free_online_ocr"),
    (Path("src/vctx/config.py"), "cleanup: CapabilityPolicy"),
    (Path("src/vctx/config.py"), "chapters: CapabilityPolicy"),
    (Path("src/vctx/models/manifest.py"), '"cleanup"'),
    (Path("src/vctx/models/manifest.py"), '"chapters"'),
    (Path("src/vctx/transforms/visual_planning.py"), "class VisualOperation"),
    (Path("src/vctx/transforms/visual_planning.py"), "class AcquisitionAction"),
    (Path("src/vctx/transforms/visual_planning.py"), "provider_type:"),
    (Path("src/vctx/transforms/visual_planning.py"), "base_url:"),
    (Path("src/vctx/transforms/visual_planning.py"), "api_key_env:"),
    (Path("src/vctx/transforms/visual_planning.py"), "model:"),
    (Path("src/vctx/transforms/visual_planning.py"), "cost:"),
    (Path("src/vctx/transforms/visual_planning.py"), "baseline_visual_operations"),
    (Path("src/vctx/transforms/visual_planning.py"), "operations: list[VisualAction]"),
    (Path("src/vctx/transforms/visual_planning.py"), "def _operation"),
    (Path("src/vctx/transforms/visual_routes.py"), "discover_visual_operations"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _configured_describe_operation"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _resolved_describe_operation"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _ai_route_describe_operation"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _operation_provider_id"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _describe_action"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _describe_route"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _action_provider_id"),
    (Path("src/vctx/transforms/ai_routes.py"), "def ai_route_from_model_route"),
    (Path("src/vctx/transforms/planning.py"), "def _asr_ai_route"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _configured_ai_route"),
    (Path("src/vctx/transforms/visual_vlm.py"), "provider: ProviderConfig"),
    (Path("src/vctx/transforms/visual_vlm.py"), "env_files:"),
    (Path("src/vctx/transforms/visual_vlm.py"), "resolve_env_credential"),
    (Path("src/vctx/transforms/visual_execute.py"), "vision_providers"),
    (Path("src/vctx/config.py"), "ProviderGroup = Literal"),
    (Path("src/vctx/config.py"), "class ProviderConfig"),
    (Path("src/vctx/config.py"), "class ProviderRegistry"),
    (Path("src/vctx/config.py"), "asr: dict[str, ProviderConfig]"),
    (Path("src/vctx/config.py"), "ocr: dict[str, ProviderConfig]"),
    (Path("src/vctx/config.py"), "text: dict[str, ProviderConfig]"),
    (Path("src/vctx/transforms/visual_routes.py"), " import ProviderConfig"),
    (Path("src/vctx/transforms/visual_routes.py"), "dict[str, ProviderConfig]"),
    (Path("src/vctx/transforms/visual_routes.py"), "vision_providers"),
    (Path("src/vctx/transforms/visual_routes.py"), "def _configured_vision_provider"),
    (Path("src/vctx/transforms/visual_routes.py"), "_vision_provider_allowed"),
    (Path("src/vctx/transforms/visual_planning.py"), "def _ai_route_action_provider_id"),
    (Path("src/vctx/app/prepare.py"), "vision_providers="),
    (Path("src/vctx/app/prepare.py"), "load_openrouter_models("),
    (Path("src/vctx/app/prepare.py"), "resolve_model_ref("),
    (Path("src/vctx/app/prepare.py"), "def _visual_ai_routes"),
    (Path("src/vctx/app/prepare.py"), "def _knowledge_flow_ai_route"),
    (Path("src/vctx/app/prepare.py"), "def _essential_case_ai_route"),
    (Path("src/vctx/app/prepare.py"), "def _text_product_ai_route"),
    (Path("src/vctx/app/prepare.py"), "def _knowledge_flow_evidence"),
    (Path("src/vctx/app/prepare.py"), "def _essential_case_evidence"),
    (Path("src/vctx/app/prepare.py"), "def _text_product_evidence"),
    (Path("src/vctx/app/prepare.py"), "def _knowledge_flow_route_detail"),
    (Path("tests/test_config_and_transforms.py"), " import ProviderConfig"),
    (Path("tests/test_config_and_transforms.py"), " ProviderConfig("),
    (Path("tests/test_config_and_transforms.py"), "vision_providers="),
    (Path("tests/test_visual_model_ref_routes.py"), " import ProviderConfig"),
    (Path("tests/test_visual_model_ref_routes.py"), " ProviderConfig("),
    (Path("tests/test_visual_model_ref_routes.py"), "vision_providers="),
    (Path("src/vctx/config.py"), "class ConfigLayer"),
    (Path("src/vctx/config.py"), "class ConfigDiscovery"),
    (Path("src/vctx/config.py"), "def discover_config_layers"),
    (Path("src/vctx/config.py"), "def _read_config_layers"),
    (Path("src/vctx/config.py"), "def _merge_config"),
    (Path("src/vctx/config.py"), "user_config_path"),
    (Path("src/vctx/config.py"), "VCTX_CONFIG"),
]

ALLOWED_DIRECT_NETWORK_FILES = {
    Path("src/vctx/net.py"),
}

DIRECT_NETWORK_SNIPPETS = [
    "urllib.request",
    "urlopen",
]

LIMITED_SNIPPETS = [
    (Path("src/vctx/app/prepare.py"), "OpenAiCompatibleTextAdapter(", 1),
    (Path("src/vctx/app/prepare.py"), "env_with_credential_presence(", 1),
]


def main() -> int:
    errors: list[str] = []

    for module_path in DELETED_MODULES:
        if module_path.exists():
            errors.append(f"deleted module still exists: {module_path}")

    for source_path in [*Path("src/vctx").rglob("*.py"), *Path("tests").rglob("*.py")]:
        source = source_path.read_text(encoding="utf-8")
        for banned_import in BANNED_IMPORTS:
            if banned_import in source:
                errors.append(f"banned import {banned_import!r} in {source_path}")
        if source_path not in ALLOWED_DIRECT_NETWORK_FILES:
            for snippet in DIRECT_NETWORK_SNIPPETS:
                if snippet in source:
                    errors.append(f"direct network snippet {snippet!r} in {source_path}")

    for source_path, banned_snippet in BANNED_SNIPPETS:
        if source_path.exists() and banned_snippet in source_path.read_text(encoding="utf-8"):
            errors.append(f"banned snippet {banned_snippet!r} in {source_path}")

    for source_path, snippet, limit in LIMITED_SNIPPETS:
        if not source_path.exists():
            continue
        count = source_path.read_text(encoding="utf-8").count(snippet)
        if count > limit:
            errors.append(
                f"snippet {snippet!r} appears {count} times in {source_path}; limit {limit}"
            )

    if errors:
        for error in errors:
            print(error)
        return 1
    print("module layout checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
