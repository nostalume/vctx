from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Annotated, Literal, cast

from platformdirs import user_cache_path, user_config_path
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
from vctx.options import MediaQuality, PrepareTarget

type Projection = Literal["context", "read"]


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

    offline: bool = False
    env_files: list[Path] = Field(default_factory=list)


class CacheInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: Path | None = None
    model_dir: Path | None = None


class SourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yt_dlp: YtDlpSourceOptions = Field(default_factory=YtDlpSourceOptions)
    media_quality: MediaQuality = MediaQuality.AUTO


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
    projections: set[Projection] | None = None
    target: PrepareTarget = PrepareTarget.TRANSCRIPT
    asr_use: TransformUse | str | None = None
    ocr_use: TransformUse | str | None = None
    vision_use: TransformUse | str | None = None
    offline: bool | None = None
    config_path: Path | None = None
    subtitle_languages: list[str] = Field(default_factory=list)
    retain_media: bool | None = None
    media_quality: MediaQuality | None = None
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def valid_interval(self) -> PrepareRequest:
        if self.end_seconds is not None and self.start_seconds is None:
            self.start_seconds = 0
        if self.end_seconds is not None and self.end_seconds <= (self.start_seconds or 0):
            raise ValueError("--end must be greater than --start")
        return self


class RuntimeConfig(BaseModel):
    offline: bool
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


class EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner: CapabilitySelection = Field(default_factory=CapabilityInput)
    vision: CapabilitySelection = Field(default_factory=CapabilityInput)
    ocr: CapabilitySelection = Field(default_factory=CapabilityInput)


class EvidenceConfig(BaseModel):
    planner: CapabilityPolicy
    vision: CapabilityPolicy
    ocr: CapabilityPolicy


class SummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use: TransformUse = Field(default_factory=AutoUse)
    language: str = "native"


class SummaryConfig(BaseModel):
    policy: CapabilityPolicy
    language: str


class OutputConfig(BaseModel):
    projections: set[Projection]
    chunk_max_chars: int
    chunk_max_seconds: int | None
    retain_media: bool = True


class OutputInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projections: set[Projection] | None = None
    chunk_max_chars: int | None = None
    chunk_max_seconds: int | None = None
    retain_media: StrictBool | None = None


class AsrInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AsrInstanceType
    model: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute: str = "auto"
    cpu_threads: Literal["auto"] | Annotated[int, Field(ge=1, le=256)] = "auto"
    batch_size: Literal["auto"] | Annotated[int, Field(ge=1, le=256)] = "auto"
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
    summary: SummaryInput = Field(default_factory=SummaryInput)
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


class ConfigFile(BaseModel):
    origin: Literal["explicit", "workspace", "environment", "global", "none"] = "none"
    path: Path | None = None


class ResolvedConfig(BaseModel):
    target: PrepareTarget
    runtime: RuntimeConfig
    cache: CacheConfig
    source: SourceConfig
    asr: CapabilityPolicy
    evidence: EvidenceConfig
    summary: SummaryConfig
    output: OutputConfig
    instances: InstanceRegistry = Field(default_factory=InstanceRegistry)
    config_file: ConfigFile = Field(default_factory=ConfigFile)


def _target_capabilities(target: PrepareTarget) -> tuple[bool, bool]:
    evidence = target in {PrepareTarget.EVIDENCE, PrepareTarget.SUMMARY}
    return evidence, target == PrepareTarget.SUMMARY


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
    if isinstance(config.summary.use, InstanceUse):
        name = config.summary.use.name
        if name not in config.instances.ai:
            raise ValueError(f"summary.use references unknown AI instance: {name}")


def _validate_instance_compatibility(
    evidence: EvidenceConfig, summary: SummaryConfig, instances: InstanceRegistry
) -> None:
    for policy in (evidence.vision, evidence.planner, summary.policy):
        if isinstance(policy.use, InstanceUse) and policy.enabled:
            instances.ai[policy.use.name].admit(policy.use.name)


def default_cache_dir() -> Path:
    return user_cache_path("vctx", appauthor=False)


def select_config_file(explicit: Path | None) -> ConfigFile:
    if explicit is not None:
        return ConfigFile(origin="explicit", path=_absolute(explicit))
    candidate = Path.cwd() / "vctx.toml"
    if candidate.is_file():
        return ConfigFile(origin="workspace", path=candidate)
    value = os.environ.get("VCTX_CONFIG")
    if value:
        return ConfigFile(origin="environment", path=_absolute(Path(value)))
    global_path = user_config_path("vctx", appauthor=False) / "config.toml"
    if global_path.is_file():
        return ConfigFile(origin="global", path=global_path)
    return ConfigFile()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _coalesce[T](*values: T | None, default: T) -> T:
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_policy(
    raw: CapabilityInput,
    enabled: bool,
) -> CapabilityPolicy:
    if not enabled:
        return CapabilityPolicy(enabled=False)
    if raw.enabled is not None:
        enabled = raw.enabled
    elif "use" in raw.model_fields_set and not isinstance(raw.use, AutoUse):
        enabled = not isinstance(raw.use, DisabledUse)
    return CapabilityPolicy(enabled=enabled, use=raw.use)


def _resolve_use(use: TransformUse, enabled: bool) -> CapabilityPolicy:
    return CapabilityPolicy(
        enabled=enabled and not isinstance(use, DisabledUse),
        use=use,
    )


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

    target = request.target
    offline = _coalesce(request.offline, config.runtime.offline, default=False)
    evidence_enabled, summary_enabled = _target_capabilities(target)

    cache_root = (
        _absolute(request.cache_dir) if request.cache_dir is not None else default_cache_root
    )
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

    projections = _coalesce(
        request.projections,
        config.output.projections,
        default=cast(set[Projection], {"context", "read"}),
    )

    ytdlp_source = _resolve_ytdlp_source_paths(config.source.yt_dlp, path_context)
    if request.subtitle_languages:
        ytdlp_source = ytdlp_source.model_copy(
            update={"subtitle_languages": request.subtitle_languages}
        )

    asr = _resolve_policy(_request_policy(config.transforms.asr, request.asr_use), True)
    evidence = EvidenceConfig(
        ocr=_resolve_policy(
            _request_policy(config.evidence.ocr, request.ocr_use),
            evidence_enabled,
        ),
        vision=_resolve_policy(
            _request_policy(config.evidence.vision, request.vision_use),
            evidence_enabled,
        ),
        planner=_resolve_policy(config.evidence.planner, evidence_enabled),
    )
    summary = SummaryConfig(
        policy=_resolve_use(config.summary.use, summary_enabled),
        language=config.summary.language,
    )
    instances = _resolve_instance_registry(config.instances, path_context)
    _validate_instance_compatibility(evidence, summary, instances)

    return ResolvedConfig(
        target=target,
        runtime=RuntimeConfig(
            offline=offline,
            env_files=path_context.resolve_config_paths(config.runtime.env_files),
        ),
        cache=CacheConfig(source_dir=source_dir, model_dir=model_dir),
        source=SourceConfig(
            yt_dlp=ytdlp_source,
            media_quality=request.media_quality or config.source.media_quality,
        ),
        asr=asr,
        evidence=evidence,
        summary=summary,
        output=OutputConfig(
            projections=projections,
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
            retain_media=_coalesce(
                request.retain_media,
                config.output.retain_media,
                default=True,
            ),
        ),
        instances=instances,
    )


def load_resolved_config(request: PrepareRequest) -> ResolvedConfig:
    selected = select_config_file(request.config_path)
    selected_request = request.model_copy(update={"config_path": selected.path})
    resolved = resolve_config(
        selected_request,
        load_config(selected.path),
        default_cache_root=default_cache_dir(),
    )
    return resolved.model_copy(update={"config_file": selected})
