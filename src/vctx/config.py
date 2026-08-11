from __future__ import annotations

import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from platformdirs import user_cache_path
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    model_validator,
)

from vctx.ai import AiInstanceConfig
from vctx.errors import ConfigError
from vctx.render.bundle import DEFAULT_FORMATS, OutputFormat


class WorkflowProfile(StrEnum):
    DEFAULT = "default"
    TRANSCRIPT = "transcript"
    VISUAL = "visual"
    FULL = "full"
    METADATA = "metadata"


class MediaQuality(StrEnum):
    AUTO = "auto"
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


def _source_session(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value in {"", "none"}:
        return NoSourceSession()
    if value.startswith("browser:"):
        return BrowserSourceSession(browser=value.removeprefix("browser:"))
    if value.startswith("cookies-file:"):
        return CookieFileSourceSession(path=Path(value.removeprefix("cookies-file:")))
    raise ValueError("source.yt_dlp.session must be none, browser:<name>, or cookies-file:<path>")


SourceSession = Annotated[
    NoSourceSession | BrowserSourceSession | CookieFileSourceSession,
    Field(discriminator="kind"),
    BeforeValidator(_source_session),
]


class DirectSourceNetwork(BaseModel):
    kind: Literal["direct"] = "direct"


class ProxySourceNetwork(BaseModel):
    kind: Literal["proxy"] = "proxy"
    url: str


def _source_network(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value in {"", "direct"}:
        return DirectSourceNetwork()
    if value.startswith("proxy:"):
        return ProxySourceNetwork(url=value.removeprefix("proxy:"))
    raise ValueError("source.yt_dlp.network must be direct or proxy:<url>")


SourceNetwork = Annotated[
    DirectSourceNetwork | ProxySourceNetwork,
    Field(discriminator="kind"),
    BeforeValidator(_source_network),
]


class DefaultPlaylistSelection(BaseModel):
    kind: Literal["default"] = "default"


class PlaylistItemsSelection(BaseModel):
    kind: Literal["items"] = "items"
    spec: str


def _playlist_selection(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value in {"", "default"}:
        return DefaultPlaylistSelection()
    if value.startswith("items:"):
        return PlaylistItemsSelection(spec=value.removeprefix("items:"))
    raise ValueError("source.yt_dlp.playlist must be default or items:<spec>")


PlaylistSelection = Annotated[
    DefaultPlaylistSelection | PlaylistItemsSelection,
    Field(discriminator="kind"),
    BeforeValidator(_playlist_selection),
]


class YtDlpSourceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: SourceSession = Field(default_factory=NoSourceSession)
    network: SourceNetwork = Field(default_factory=DirectSourceNetwork)
    playlist: PlaylistSelection = Field(default_factory=DefaultPlaylistSelection)
    subtitle_languages: list[str] = Field(default_factory=list)


class RuntimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: WorkflowProfile | None = None
    keep_temp: bool = False
    offline: bool = False
    env_files: list[Path] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_old_cache(cls, value: object) -> object:
        if isinstance(value, dict) and "cache_dir" in value:
            raise ValueError("runtime.cache_dir was removed; use [cache].source_dir and model_dir")
        return value


class CacheInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: Path | None = None
    model_dir: Path | None = None


class SourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yt_dlp: YtDlpSourceOptions = Field(default_factory=YtDlpSourceOptions)
    media_quality: MediaQuality = MediaQuality.AUTO

    @model_validator(mode="before")
    @classmethod
    def _reject_old_media_profile(cls, value: object) -> object:
        if isinstance(value, Mapping):
            yt_dlp = cast(Mapping[str, object], value).get("yt_dlp")
            if isinstance(yt_dlp, Mapping) and "media_profile" in yt_dlp:
                raise ValueError("source.yt_dlp.media_profile: use source.media_quality")
        return value


class AutoUse(BaseModel):
    kind: Literal["auto"] = "auto"


class DisabledUse(BaseModel):
    kind: Literal["none"] = "none"


class InstanceUse(BaseModel):
    kind: Literal["instance"] = "instance"
    name: str


class ModelRefUse(BaseModel):
    kind: Literal["model_ref"] = "model_ref"
    ref: str


def _transform_use(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value == "auto":
        return AutoUse()
    if value == "none":
        return DisabledUse()
    if value.startswith("instance:"):
        return InstanceUse(name=value.removeprefix("instance:"))
    prefix, separator, _rest = value.partition(":")
    if separator and prefix in {"path", "local", "hf"}:
        return ModelRefUse(ref=value)
    raise ValueError(
        "transform use must be auto, none, instance:<name>, path:<local-path>, "
        "local:<path-or-id>, or hf:<repo-id>"
    )


TransformUse = Annotated[
    AutoUse | DisabledUse | InstanceUse | ModelRefUse,
    Field(discriminator="kind"),
    BeforeValidator(_transform_use),
]

AsrInstanceType = Literal["local-faster-whisper"]
InstanceCachePolicy = Literal["persistent", "disabled"]


class PrepareRequest(BaseModel):
    inputs: list[str] = Field(min_length=1)
    out_dir: Path
    overwrite: bool = False
    chunk_max_chars: int | None = None
    chunk_max_seconds: int | None = None
    cache_dir: Path | None = None
    keep_temp: bool | None = None
    formats: set[OutputFormat] | None = None
    workflow: WorkflowProfile | None = None
    asr_use: TransformUse | str | None = None
    ocr_use: TransformUse | str | None = None
    vision_use: TransformUse | str | None = None
    offline: bool | None = None
    config_path: Path | None = None
    subtitle_languages: list[str] = Field(default_factory=list)
    output_language: str | None = None
    retain_media: bool | None = None
    media_quality: MediaQuality | None = None


class RuntimeConfig(BaseModel):
    keep_temp: bool
    offline: bool
    workflow: WorkflowProfile
    env_files: list[Path] = Field(default_factory=list)


class CacheConfig(BaseModel):
    source_dir: Path
    model_dir: Path


class SourceConfig(BaseModel):
    yt_dlp: YtDlpSourceOptions = Field(default_factory=YtDlpSourceOptions)
    media_quality: MediaQuality


class CapabilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    use: TransformUse = Field(default_factory=AutoUse)

    @model_validator(mode="after")
    def _use_matches_enabled(self) -> CapabilityPolicy:
        if self.enabled and isinstance(self.use, DisabledUse):
            raise ValueError("enabled capability cannot use none")
        if not self.enabled:
            self.use = DisabledUse()
        return self

    def disabled(self) -> bool:
        return not self.enabled or isinstance(self.use, DisabledUse)

    def instance_name(self) -> str | None:
        return self.use.name if isinstance(self.use, InstanceUse) else None

    def model_ref(self) -> str | None:
        return self.use.ref if isinstance(self.use, ModelRefUse) else None

    def auto(self) -> bool:
        return isinstance(self.use, AutoUse)


class CapabilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool | None = None
    use: TransformUse = Field(default_factory=AutoUse)


def _capability_input(value: object) -> object:
    return {"use": value} if isinstance(value, str) else value


CapabilitySelection = Annotated[CapabilityInput, BeforeValidator(_capability_input)]


class TransformInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asr: CapabilityInput = Field(default_factory=CapabilityInput)


class TransformConfig(BaseModel):
    asr: CapabilityPolicy


class EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner: CapabilitySelection = Field(default_factory=CapabilityInput)
    vision: CapabilitySelection = Field(default_factory=CapabilityInput)
    ocr: CapabilitySelection = Field(default_factory=CapabilityInput)


class EvidenceConfig(BaseModel):
    planner: CapabilityPolicy
    vision: CapabilityPolicy
    ocr: CapabilityPolicy


class OutputConfig(BaseModel):
    formats: set[OutputFormat]
    chunk_max_chars: int
    chunk_max_seconds: int | None
    language: str = "native"
    retain_media: bool = True


class OutputInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formats: set[OutputFormat] | None = None
    chunk_max_chars: int | None = None
    chunk_max_seconds: int | None = None
    language: str | None = None
    retain_media: StrictBool | None = None


class AsrInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AsrInstanceType
    model: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute: str = "auto"
    cache: InstanceCachePolicy = "persistent"


class InstanceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asr: dict[str, AsrInstanceConfig] = Field(default_factory=dict)
    ai: dict[str, AiInstanceConfig] = Field(default_factory=dict)


class ConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeInput = Field(default_factory=RuntimeInput)
    cache: CacheInput = Field(default_factory=CacheInput)
    source: SourceInput = Field(default_factory=SourceInput)
    transforms: TransformInput = Field(default_factory=TransformInput)
    evidence: EvidenceInput = Field(default_factory=EvidenceInput)
    output: OutputInput = Field(default_factory=OutputInput)
    instances: InstanceRegistry = Field(default_factory=InstanceRegistry)


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
    cache: CacheConfig
    source: SourceConfig
    transforms: TransformConfig
    evidence: EvidenceConfig
    output: OutputConfig
    instances: InstanceRegistry = Field(default_factory=InstanceRegistry)


def _workflow_capabilities(
    workflow: WorkflowProfile,
) -> tuple[bool, bool, bool]:
    if workflow == WorkflowProfile.METADATA:
        return (False, False, False)
    if workflow == WorkflowProfile.TRANSCRIPT:
        return (True, False, False)
    if workflow == WorkflowProfile.VISUAL:
        return (True, True, True)
    if workflow == WorkflowProfile.FULL:
        return (True, True, True)
    return (True, False, False)


def load_config(path: Path | None) -> ConfigInput:
    if path is None:
        return ConfigInput()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return ConfigInput.model_validate(data)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc


def _resolve_ytdlp_source_paths(
    options: YtDlpSourceOptions, paths: ConfigPathContext
) -> YtDlpSourceOptions:
    if not isinstance(options.session, CookieFileSourceSession):
        return options
    return options.model_copy(
        update={
            "session": options.session.model_copy(
                update={"path": paths.resolve_config_path(options.session.path)}
            )
        }
    )


def _resolve_instance_registry(
    instances: InstanceRegistry, paths: ConfigPathContext
) -> InstanceRegistry:
    return InstanceRegistry(
        asr={
            name: _resolve_asr_instance_paths(instance, paths)
            for name, instance in instances.asr.items()
        },
        ai=instances.ai,
    )


def _resolve_asr_instance_paths(
    instance: AsrInstanceConfig, paths: ConfigPathContext
) -> AsrInstanceConfig:
    if instance.model is None or not instance.model.startswith("path:"):
        return instance
    resolved_model = paths.resolve_config_path(Path(instance.model.removeprefix("path:")))
    return instance.model_copy(update={"model": str(resolved_model)})


def _validate_instance_refs(config: ConfigInput) -> None:
    if (
        isinstance(config.transforms.asr.use, InstanceUse)
        and config.transforms.asr.use.name not in config.instances.asr
    ):
        raise ValueError(
            f"transforms.asr.use references unknown ASR instance: {config.transforms.asr.use.name}"
        )
    if (
        isinstance(config.evidence.vision.use, InstanceUse)
        and config.evidence.vision.use.name not in config.instances.ai
    ):
        raise ValueError(
            f"evidence.vision references unknown AI instance: {config.evidence.vision.use.name}"
        )
    if (
        isinstance(config.evidence.planner.use, InstanceUse)
        and config.evidence.planner.use.name not in config.instances.ai
    ):
        raise ValueError(
            f"evidence.planner references unknown AI instance: {config.evidence.planner.use.name}"
        )


def _validate_instance_compatibility(evidence: EvidenceConfig, instances: InstanceRegistry) -> None:
    for policy in (evidence.vision, evidence.planner):
        if isinstance(policy.use, InstanceUse) and policy.enabled:
            instances.ai[policy.use.name].admit(policy.use.name)


def default_cache_dir() -> Path:
    return user_cache_path("vctx", appauthor=False)


def _coalesce[T](*values: T | None, default: T) -> T:
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_policy(
    raw: CapabilityInput,
    enabled: bool,
) -> CapabilityPolicy:
    if raw.enabled is not None:
        enabled = raw.enabled
    elif "use" in raw.model_fields_set and not isinstance(raw.use, AutoUse):
        enabled = True
    return CapabilityPolicy(enabled=enabled, use=raw.use)


def _request_policy(raw: CapabilityInput, use: TransformUse | str | None) -> CapabilityInput:
    if use is None:
        return raw
    if isinstance(use, str):
        use = TypeAdapter(TransformUse).validate_python(use)
    return raw.model_copy(
        update={
            "enabled": not isinstance(use, DisabledUse),
            "use": use,
        }
    )


def resolve_config(
    request: PrepareRequest,
    config: ConfigInput,
    *,
    default_cache_root: Path,
) -> ResolvedConfig:
    """Resolve user request/config omissions into concrete default/auto policy."""

    try:
        _validate_instance_refs(config)
        return _resolve_config(request, config, default_cache_root=default_cache_root)
    except ValueError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc


def _resolve_config(
    request: PrepareRequest,
    config: ConfigInput,
    *,
    default_cache_root: Path,
) -> ResolvedConfig:
    path_context = ConfigPathContext(
        base_dir=request.config_path.parent if request.config_path is not None else None
    )

    workflow = _coalesce(
        request.workflow,
        config.runtime.workflow,
        default=WorkflowProfile.DEFAULT,
    )
    offline = _coalesce(request.offline, config.runtime.offline, default=False)
    keep_temp = _coalesce(request.keep_temp, config.runtime.keep_temp, default=False)
    asr, visual_enabled, planner_enabled = _workflow_capabilities(workflow)

    cache_root = request.cache_dir or default_cache_root
    source_dir = (
        cache_root / "source"
        if request.cache_dir is not None or config.cache.source_dir is None
        else path_context.resolve_config_path(config.cache.source_dir)
    )
    model_dir = (
        cache_root / "models"
        if request.cache_dir is not None or config.cache.model_dir is None
        else path_context.resolve_config_path(config.cache.model_dir)
    )

    formats = _coalesce(request.formats, config.output.formats, default=DEFAULT_FORMATS)
    language = _coalesce(
        request.output_language,
        config.output.language,
        default="native",
    )

    ytdlp_source = _resolve_ytdlp_source_paths(config.source.yt_dlp, path_context)
    if request.subtitle_languages:
        ytdlp_source = ytdlp_source.model_copy(
            update={"subtitle_languages": request.subtitle_languages}
        )

    transforms = TransformConfig(
        asr=_resolve_policy(_request_policy(config.transforms.asr, request.asr_use), asr),
    )
    evidence = EvidenceConfig(
        ocr=_resolve_policy(
            _request_policy(config.evidence.ocr, request.ocr_use),
            visual_enabled,
        ),
        vision=_resolve_policy(
            _request_policy(config.evidence.vision, request.vision_use),
            visual_enabled,
        ),
        planner=_resolve_policy(config.evidence.planner, planner_enabled),
    )
    instances = _resolve_instance_registry(config.instances, path_context)
    _validate_instance_compatibility(evidence, instances)

    return ResolvedConfig(
        runtime=RuntimeConfig(
            keep_temp=keep_temp,
            offline=offline,
            workflow=workflow,
            env_files=path_context.resolve_config_paths(config.runtime.env_files),
        ),
        cache=CacheConfig(source_dir=source_dir, model_dir=model_dir),
        source=SourceConfig(
            yt_dlp=ytdlp_source,
            media_quality=request.media_quality or config.source.media_quality,
        ),
        transforms=transforms,
        evidence=evidence,
        output=OutputConfig(
            formats=formats,
            chunk_max_chars=_coalesce(
                request.chunk_max_chars,
                config.output.chunk_max_chars,
                default=6000,
            ),
            chunk_max_seconds=_coalesce(
                request.chunk_max_seconds,
                config.output.chunk_max_seconds,
                default=None,
            ),
            language=language,
            retain_media=_coalesce(
                request.retain_media,
                config.output.retain_media,
                default=True,
            ),
        ),
        instances=instances,
    )


def load_resolved_config(request: PrepareRequest) -> ResolvedConfig:
    return resolve_config(
        request,
        load_config(request.config_path),
        default_cache_root=default_cache_dir(),
    )
