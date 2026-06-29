from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, cast

from platformdirs import user_cache_path
from pydantic import BaseModel, Field

from vctx.render.bundle import DEFAULT_FORMATS, OutputFormat

ConfigScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
ConfigValue: TypeAlias = (  # noqa: UP040
    ConfigScalar | list["ConfigValue"] | dict[str, "ConfigValue"]
)
ConfigTable: TypeAlias = dict[str, ConfigValue]  # noqa: UP040

class CapabilityEnabled(StrEnum):
    AUTO = "auto"
    TRUE = "true"
    FALSE = "false"


class WorkflowProfile(StrEnum):
    DEFAULT = "default"
    TRANSCRIPT = "transcript"
    VISUAL = "visual"
    FULL = "full"
    METADATA = "metadata"


class MediaProfile(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


class NoSourceSession(BaseModel):
    kind: Literal["none"] = "none"


class BrowserSourceSession(BaseModel):
    kind: Literal["browser"] = "browser"
    browser: str


class CookieFileSourceSession(BaseModel):
    kind: Literal["cookies_file"] = "cookies_file"
    path: Path


SourceSession = Annotated[
    NoSourceSession | BrowserSourceSession | CookieFileSourceSession,
    Field(discriminator="kind"),
]


class DirectSourceNetwork(BaseModel):
    kind: Literal["direct"] = "direct"


class ProxySourceNetwork(BaseModel):
    kind: Literal["proxy"] = "proxy"
    url: str


SourceNetwork = Annotated[
    DirectSourceNetwork | ProxySourceNetwork,
    Field(discriminator="kind"),
]


class DefaultPlaylistSelection(BaseModel):
    kind: Literal["default"] = "default"


class PlaylistItemsSelection(BaseModel):
    kind: Literal["items"] = "items"
    spec: str


PlaylistSelection = Annotated[
    DefaultPlaylistSelection | PlaylistItemsSelection,
    Field(discriminator="kind"),
]


class YtDlpSourceOptions(BaseModel):
    session: SourceSession = Field(default_factory=NoSourceSession)
    network: SourceNetwork = Field(default_factory=DirectSourceNetwork)
    playlist: PlaylistSelection = Field(default_factory=DefaultPlaylistSelection)
    media_profile: MediaProfile = MediaProfile.BALANCED
    subtitle_languages: list[str] = Field(default_factory=list)


CapabilityRoute = Literal[
    "auto",
    "default",
    "local",
    "free-online",
    "configured-online",
    "disabled",
    "explicit",
]
AsrInstanceType = Literal["local-faster-whisper", "openai-compatible-audio"]
InstanceCachePolicy = Literal["persistent", "disabled"]


class PrepareRequest(BaseModel):
    input: str
    out_dir: Path
    overwrite: bool = False
    chunk_max_chars: int = 6000
    chunk_max_seconds: int | None = None
    cache_dir: Path | None = None
    keep_temp: bool = False
    formats: set[OutputFormat] = DEFAULT_FORMATS
    workflow: WorkflowProfile = WorkflowProfile.DEFAULT
    offline: bool = False
    config_path: Path | None = None
    subtitle_languages: list[str] = Field(default_factory=list)
    output_language: str = "native"


class RuntimeConfig(BaseModel):
    cache_dir: Path
    keep_temp: bool
    offline: bool
    workflow: WorkflowProfile
    env_files: list[Path] = Field(default_factory=list)


class SourceConfig(BaseModel):
    yt_dlp: YtDlpSourceOptions = Field(default_factory=YtDlpSourceOptions)


class CapabilityPolicy(BaseModel):
    enabled: CapabilityEnabled
    route: CapabilityRoute = "auto"
    instance: str | None = None
    allow_network: bool = True
    allow_upload: bool = True
    preferred_provider: str | None = None
    model: str | None = None


class TransformConfig(BaseModel):
    asr: CapabilityPolicy
    visual_context: CapabilityPolicy
    knowledge_flow: CapabilityPolicy


class OutputConfig(BaseModel):
    formats: set[OutputFormat]
    chunk_max_chars: int
    chunk_max_seconds: int | None
    language: str = "native"


class VisionInstanceConfig(BaseModel):
    type: str
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None


class AsrInstanceConfig(BaseModel):
    type: AsrInstanceType
    model: str | None = None
    model_policy: Literal["auto", "tiny", "base", "small", "medium", "large"] = "auto"
    cache: InstanceCachePolicy = "persistent"
    base_url: str | None = None
    api_key_env: str | None = None


class InstanceRegistry(BaseModel):
    asr: dict[str, AsrInstanceConfig] = Field(default_factory=dict)
    vision: dict[str, VisionInstanceConfig] = Field(default_factory=dict)


class ConfigPathContext(BaseModel):
    base_dir: Path | None = None

    def resolve_config_path(self, value: Path) -> Path:
        if value.is_absolute() or self.base_dir is None:
            return value
        return self.base_dir / value

    def resolve_config_paths(self, values: list[Path]) -> list[Path]:
        return [self.resolve_config_path(value) for value in values]


class ResolvedConfig(BaseModel):
    runtime: RuntimeConfig
    source: SourceConfig
    transforms: TransformConfig
    output: OutputConfig
    instances: InstanceRegistry = Field(default_factory=InstanceRegistry)


def _policy(
    enabled: CapabilityEnabled, *, offline: bool, allow_upload: bool = True
) -> CapabilityPolicy:
    network_allowed = not offline
    return CapabilityPolicy(
        enabled=enabled,
        route="auto" if enabled != CapabilityEnabled.FALSE else "disabled",
        allow_network=network_allowed,
        allow_upload=network_allowed and allow_upload,
    )


def _workflow_capabilities(
    workflow: WorkflowProfile,
) -> tuple[CapabilityEnabled, CapabilityEnabled, CapabilityEnabled]:
    if workflow == WorkflowProfile.METADATA:
        return (
            CapabilityEnabled.FALSE,
            CapabilityEnabled.FALSE,
            CapabilityEnabled.FALSE,
        )
    if workflow == WorkflowProfile.TRANSCRIPT:
        return (
            CapabilityEnabled.AUTO,
            CapabilityEnabled.FALSE,
            CapabilityEnabled.FALSE,
        )
    if workflow == WorkflowProfile.VISUAL:
        return (
            CapabilityEnabled.AUTO,
            CapabilityEnabled.TRUE,
            CapabilityEnabled.AUTO,
        )
    if workflow == WorkflowProfile.FULL:
        return (
            CapabilityEnabled.AUTO,
            CapabilityEnabled.TRUE,
            CapabilityEnabled.TRUE,
        )
    return (
        CapabilityEnabled.AUTO,
        CapabilityEnabled.AUTO,
        CapabilityEnabled.AUTO,
    )


def _read_config(path: Path | None) -> ConfigTable:
    if path is None:
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return cast(ConfigTable, data) if isinstance(data, dict) else {}


def _section(config: ConfigTable, name: str) -> ConfigTable:
    value = config.get(name, {})
    return cast(ConfigTable, value) if isinstance(value, dict) else {}


def _capability_config(config: ConfigTable, capability: str) -> ConfigTable:
    transforms = _section(config, "transforms")
    value = transforms.get(capability, {})
    return cast(ConfigTable, value) if isinstance(value, dict) else {}


def _config_value(section: ConfigTable, name: str, default: ConfigValue) -> ConfigValue:
    value = section.get(name, default)
    return default if value == "auto" else value


def _resolve_ytdlp_source_options(
    config: ConfigTable, paths: ConfigPathContext
) -> YtDlpSourceOptions:
    source = _section(config, "source")
    ytdlp = source.get("yt_dlp", {})
    values = ytdlp if isinstance(ytdlp, dict) else {}
    return YtDlpSourceOptions(
        session=_source_session(_config_value(values, "session", "none"), paths),
        network=_source_network(_config_value(values, "network", "direct")),
        playlist=_playlist_selection(_config_value(values, "playlist", "default")),
        media_profile=MediaProfile(_config_value(values, "media_profile", MediaProfile.BALANCED)),
        subtitle_languages=_string_list(_config_value(values, "subtitle_languages", [])),
    )


def _string_list(value: ConfigValue) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _string_value(value: ConfigValue, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _source_session(value: ConfigValue, paths: ConfigPathContext) -> SourceSession:
    if not isinstance(value, str) or value in {"", "none"}:
        return NoSourceSession()
    if value.startswith("browser:"):
        return BrowserSourceSession(browser=value.removeprefix("browser:"))
    if value.startswith("cookies-file:"):
        raw_path = Path(value.removeprefix("cookies-file:"))
        return CookieFileSourceSession(path=paths.resolve_config_path(raw_path))
    raise ValueError("source.yt_dlp.session must be none, browser:<name>, or cookies-file:<path>")


def _source_network(value: ConfigValue) -> SourceNetwork:
    if not isinstance(value, str) or value in {"", "direct"}:
        return DirectSourceNetwork()
    if value.startswith("proxy:"):
        return ProxySourceNetwork(url=value.removeprefix("proxy:"))
    raise ValueError("source.yt_dlp.network must be direct or proxy:<url>")


def _playlist_selection(value: ConfigValue) -> PlaylistSelection:
    if not isinstance(value, str) or value in {"", "default"}:
        return DefaultPlaylistSelection()
    if value.startswith("items:"):
        return PlaylistItemsSelection(spec=value.removeprefix("items:"))
    raise ValueError("source.yt_dlp.playlist must be default or items:<spec>")


def _resolve_instance_registry(config: ConfigTable, paths: ConfigPathContext) -> InstanceRegistry:
    instances = _section(config, "instances")
    raw_asr_table = instances.get("asr", {})
    raw_vision_table = instances.get("vision", {})
    asr_table = raw_asr_table if isinstance(raw_asr_table, dict) else {}
    vision_table = raw_vision_table if isinstance(raw_vision_table, dict) else {}
    return InstanceRegistry(
        asr={
            name: _resolve_asr_instance_paths(AsrInstanceConfig.model_validate(raw_instance), paths)
            for name, raw_instance in asr_table.items()
            if isinstance(raw_instance, dict)
        },
        vision={
            name: VisionInstanceConfig.model_validate(raw_instance)
            for name, raw_instance in vision_table.items()
            if isinstance(raw_instance, dict)
        },
    )

def _resolve_asr_instance_paths(
    instance: AsrInstanceConfig, paths: ConfigPathContext
) -> AsrInstanceConfig:
    if instance.model is None or not _looks_like_path_value(instance.model):
        return instance
    resolved_model = paths.resolve_config_path(Path(instance.model))
    return instance.model_copy(update={"model": str(resolved_model)})


def _looks_like_path_value(value: str) -> bool:
    path = Path(value)
    return (
        path.is_absolute()
        or value.startswith(".")
        or any(separator in value for separator in ("/", "\\"))
    )


def _default_cache_dir() -> Path:
    return user_cache_path("vctx", appauthor=False)


def _paths(values: ConfigValue) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, str):
        return [Path(values)]
    if isinstance(values, list):
        return [Path(value) for value in values if isinstance(value, str)]
    return []


def _optional_path(value: ConfigValue) -> Path | None:
    return Path(value) if isinstance(value, str) else None


def _int_value(value: ConfigValue, default: int) -> int:
    return int(value) if isinstance(value, int | float | str) else default


def _optional_int(value: ConfigValue) -> int | None:
    return int(value) if isinstance(value, int | float | str) else None


def _output_formats(value: ConfigValue) -> set[OutputFormat]:
    if not isinstance(value, list):
        return DEFAULT_FORMATS
    return {cast(OutputFormat, item) for item in value if isinstance(item, str)}


def _resolve_policy(
    capability: str,
    enabled: CapabilityEnabled,
    *,
    offline: bool,
    config: ConfigTable,
    allow_upload: bool = True,
) -> CapabilityPolicy:
    values = _capability_config(config, capability)
    policy = _policy(enabled, offline=offline, allow_upload=allow_upload)
    if values:
        policy = policy.model_copy(update=values)
    if offline:
        policy = policy.model_copy(update={"allow_network": False, "allow_upload": False})
    return policy


def resolve_config(request: PrepareRequest) -> ResolvedConfig:
    """Resolve user request/config omissions into concrete default/auto policy."""

    config = _read_config(request.config_path)
    path_context = ConfigPathContext(
        base_dir=request.config_path.parent if request.config_path is not None else None
    )
    runtime = _section(config, "runtime")
    output = _section(config, "output")

    configured_workflow = _config_value(runtime, "workflow", WorkflowProfile.DEFAULT)
    workflow = request.workflow
    if workflow == WorkflowProfile.DEFAULT and configured_workflow != WorkflowProfile.DEFAULT:
        workflow = WorkflowProfile(configured_workflow)

    configured_offline = bool(_config_value(runtime, "offline", False))
    offline = request.offline or configured_offline
    asr, visual_context, knowledge_flow = _workflow_capabilities(workflow)

    cache_dir = request.cache_dir
    if cache_dir is None:
        configured_cache = _config_value(runtime, "cache_dir", None)
        configured_cache_path = _optional_path(configured_cache)
        cache_dir = (
            path_context.resolve_config_path(configured_cache_path)
            if configured_cache_path is not None
            else _default_cache_dir()
        )

    formats = request.formats
    if formats == DEFAULT_FORMATS and "formats" in output:
        formats = _output_formats(output["formats"])
    language = request.output_language
    if language == "native":
        language = _string_value(_config_value(output, "language", "native"), "native")

    ytdlp_source = _resolve_ytdlp_source_options(config, path_context)
    if request.subtitle_languages:
        ytdlp_source = ytdlp_source.model_copy(
            update={"subtitle_languages": request.subtitle_languages}
        )

    return ResolvedConfig(
        runtime=RuntimeConfig(
            cache_dir=cache_dir,
            keep_temp=request.keep_temp or bool(_config_value(runtime, "keep_temp", False)),
            offline=offline,
            workflow=workflow,
            env_files=path_context.resolve_config_paths(
                _paths(_config_value(runtime, "env_files", []))
            ),
        ),
        source=SourceConfig(yt_dlp=ytdlp_source),
        transforms=TransformConfig(
            asr=_resolve_policy("asr", asr, offline=offline, config=config),
            visual_context=_resolve_policy(
                "visual_context", visual_context, offline=offline, config=config
            ),
            knowledge_flow=_resolve_policy(
                "knowledge_flow",
                knowledge_flow,
                offline=offline,
                config=config,
            ),
        ),
        output=OutputConfig(
            formats=formats,
            chunk_max_chars=_int_value(
                _config_value(output, "chunk_max_chars", request.chunk_max_chars),
                request.chunk_max_chars,
            ),
            chunk_max_seconds=_optional_int(
                _config_value(output, "chunk_max_seconds", request.chunk_max_seconds)
            ),
            language=language,
        ),
        instances=_resolve_instance_registry(config, path_context),
    )
