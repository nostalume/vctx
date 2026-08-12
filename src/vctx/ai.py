from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, model_validator

from vctx.artifact.manifest import ManifestEffect
from vctx.net import NetError, NetPurpose, NetRequest, NetResponse, NetRuntime, RetryPolicy

type AiTask = Literal["evidence_plan", "summary", "vision_description"]
type AiFormat = Literal["schema", "json", "prompt"]
type AiFormatPolicy = Literal["auto", "schema", "json", "prompt"]
type AiRequestPolicy = Literal["generic", "openrouter-free-zdr"]
type AiSelectedRoute = Literal["local", "free-online", "configured-online"]
type AiFailureCode = Literal[
    "unavailable",
    "connection",
    "timeout",
    "http",
    "context_length",
    "invalid_response",
    "invalid_output",
]
type AiOutcome[T] = AiSuccess[T] | AiFailure


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CredentialRef(RootModel[str]):
    @model_validator(mode="after")
    def valid_locator(self) -> CredentialRef:
        _credential_parts(self.root)
        return self

    @property
    def kind(self) -> Literal["env", "keyring"]:
        kind, _name = _credential_parts(self.root)
        return kind

    @property
    def name(self) -> str:
        _kind, name = _credential_parts(self.root)
        return name

    def __str__(self) -> str:
        return self.root


@dataclass(frozen=True)
class Credential:
    value: str = field(repr=False)


class AiInstance(ClosedModel):
    name: str
    provider_id: str
    base_url: str
    model: str
    credential: CredentialRef | None = None
    request_policy: AiRequestPolicy = "generic"
    format: AiFormatPolicy = "auto"
    timeout_s: int = Field(default=120, ge=1, le=900)
    insecure: bool = False
    privacy: Literal["standard", "zdr"] = "standard"
    cost: Literal["free", "unknown"] = "unknown"
    require_parameters: bool = False

    @model_validator(mode="after")
    def portable_endpoint(self) -> AiInstance:
        parsed = urlsplit(self.base_url)
        loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("AI base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment or not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError("AI base_url must identify one /v1 root without query or fragment")
        if parsed.scheme == "http" and not loopback and not self.insecure:
            raise ValueError("remote cleartext AI requires insecure = true")
        if self.credential is None and not loopback:
            raise ValueError("remote AI requires a credential reference")
        self.base_url = self.base_url.rstrip("/")
        return self

    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"


class AiInstanceConfig(ClosedModel):
    base_url: str
    model: str
    credential: CredentialRef | None = None
    format: AiFormatPolicy = "auto"
    timeout_s: int = Field(default=120, ge=1, le=900)
    insecure: bool = False

    def admit(self, name: str) -> AiInstance:
        return AiInstance(
            name=name,
            provider_id=name,
            base_url=self.base_url,
            model=self.model,
            credential=self.credential,
            format=self.format,
            timeout_s=self.timeout_s,
            insecure=self.insecure,
        )


class AiRoute(ClosedModel):
    task: AiTask
    selected: AiSelectedRoute
    instance: AiInstance
    reason: str
    warnings: list[str] = Field(default_factory=list)

    @property
    def provider_id(self) -> str:
        return self.instance.provider_id

    @property
    def model(self) -> str:
        return self.instance.model

    @property
    def base_url(self) -> str:
        return self.instance.endpoint()

    @property
    def credential(self) -> str | None:
        return str(self.instance.credential) if self.instance.credential is not None else None

    def detail(self) -> str:
        return f"{self.provider_id}: {self.model}"

    def effect(self, operation: AiTask) -> ManifestEffect:
        return ManifestEffect(
            operation=operation,
            status="selected",
            route=self.selected,
            provider=self.provider_id,
            model=self.model,
            uploaded=self.selected != "local",
            cost_may_apply=self.instance.cost != "free",
            diagnostic="; ".join([self.reason, *self.warnings])[:500],
        )


@dataclass(frozen=True)
class AiBinding:
    route: AiRoute
    credential: Credential | None


class AiTextPart(ClosedModel):
    type: Literal["text"] = "text"
    text: str


class AiImageUrl(ClosedModel):
    url: str


class AiImagePart(ClosedModel):
    type: Literal["image_url"] = "image_url"
    image_url: AiImageUrl


class AiMessage(ClosedModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[AiTextPart | AiImagePart]


class AiReceipt(ClosedModel):
    task: AiTask
    request_id: str
    provider: str
    configured_model: str
    actual_model: str | None = None
    format: AiFormat
    attempts: int = Field(ge=1)
    latency_ms: int = Field(ge=0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reported_cost: float | None = None
    privacy: Literal["standard", "zdr"]


class AiSuccess[T](ClosedModel):
    kind: Literal["ok"] = "ok"
    value: T
    receipt: AiReceipt


class AiFailure(ClosedModel):
    kind: Literal["failed"] = "failed"
    failure: AiFailureCode
    reason: str
    receipt: AiReceipt


@dataclass
class AiRuntimePool:
    formats: dict[str, AiFormat] = field(default_factory=dict)
    limits: dict[str, threading.BoundedSemaphore] = field(default_factory=dict)

    def limit(self, instance: AiInstance) -> threading.BoundedSemaphore:
        key = instance.model_dump_json()
        limit = self.limits.get(key)
        if limit is None:
            limit = threading.BoundedSemaphore(2)
            self.limits[key] = limit
        return limit


class AiClient:
    def __init__(
        self,
        binding: AiBinding,
        *,
        net: NetRuntime,
        runtimes: AiRuntimePool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.instance = binding.route.instance
        self.credential = binding.credential.value if binding.credential is not None else None
        self.net = net
        self.runtimes = runtimes or AiRuntimePool()
        self.clock = clock

    def complete[T: BaseModel](
        self, *, task: AiTask, request_id: str, messages: list[AiMessage], result: type[T]
    ) -> AiOutcome[T]:
        started, attempts = self.clock(), 0
        formats = self._formats()
        with self.runtimes.limit(self.instance):
            for format in formats:
                response, used = self._request(task, request_id, messages, result, format)
                attempts += used
                if isinstance(response, Exception):
                    cause = response.cause if isinstance(response, NetError) else response
                    return self._failure(
                        task,
                        request_id,
                        format,
                        attempts,
                        started,
                        "timeout" if isinstance(cause, TimeoutError) else "connection",
                        type(cause).__name__,
                    )
                if _unsupported_format(response, format) and self.instance.format == "auto":
                    continue
                if not 200 <= response.status_code < 300:
                    failure = "context_length" if _context_length_rejected(response) else "http"
                    return self._failure(
                        task,
                        request_id,
                        format,
                        attempts,
                        started,
                        failure,
                        f"AI endpoint returned HTTP {response.status_code}",
                    )
                admitted = _admit_chat_response(response)
                if isinstance(admitted, str):
                    return self._failure(
                        task,
                        request_id,
                        format,
                        attempts,
                        started,
                        "invalid_response",
                        admitted,
                    )
                try:
                    value = result.model_validate_json(admitted.content)
                except ValidationError:
                    return self._failure(
                        task,
                        request_id,
                        format,
                        attempts,
                        started,
                        "invalid_output",
                        "AI output failed local schema validation",
                        chat=admitted,
                    )
                self.runtimes.formats[self.instance.name] = format
                return AiSuccess(
                    value=value,
                    receipt=self._receipt(task, request_id, format, attempts, started, admitted),
                )
        return self._failure(
            task,
            request_id,
            formats[-1],
            attempts,
            started,
            "unavailable",
            "AI endpoint supports none of the configured structured-output formats",
        )

    def _formats(self) -> list[AiFormat]:
        if self.instance.format != "auto":
            return [self.instance.format]
        cached = self.runtimes.formats.get(self.instance.name)
        return [cached] if cached else ["schema", "json", "prompt"]

    def _request[T: BaseModel](
        self,
        task: AiTask,
        request_id: str,
        messages: list[AiMessage],
        result: type[T],
        format: AiFormat,
    ) -> tuple[NetResponse | Exception, int]:
        try:
            response = self.net.request(
                self._net_request(task, request_id, messages, result, format)
            )
        except NetError as exc:
            return exc, exc.attempts
        return response, response.attempts

    def _net_request[T: BaseModel](
        self,
        task: AiTask,
        request_id: str,
        messages: list[AiMessage],
        result: type[T],
        format: AiFormat,
    ) -> NetRequest:
        headers = {"Content-Type": "application/json", "Idempotency-Key": request_id}
        if self.credential:
            headers["Authorization"] = f"Bearer {self.credential}"
        body = _request_body(self.instance, messages, result, format)
        return NetRequest(
            method="POST",
            url=self.instance.endpoint(),
            headers=headers,
            body=json.dumps(body, separators=(",", ":")).encode(),
            timeout_s=180 if task == "vision_description" else self.instance.timeout_s,
            connect_timeout_s=10,
            purpose=_net_purpose(task),
            provider_id=self.instance.provider_id,
            retry=RetryPolicy(
                max_attempts=2,
                statuses=(429, 500, 502, 503, 504),
                retry_connect=True,
            ),
        )

    def _receipt(
        self,
        task: AiTask,
        request_id: str,
        format: AiFormat,
        attempts: int,
        started: float,
        chat: _Chat | None = None,
    ) -> AiReceipt:
        usage = chat.usage if chat else None
        return AiReceipt(
            task=task,
            request_id=request_id,
            provider=self.instance.provider_id,
            configured_model=self.instance.model,
            actual_model=chat.model if chat else None,
            format=format,
            attempts=attempts,
            latency_ms=max(0, round((self.clock() - started) * 1000)),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            reported_cost=usage.cost if usage else None,
            privacy=self.instance.privacy,
        )

    def _failure(
        self,
        task: AiTask,
        request_id: str,
        format: AiFormat,
        attempts: int,
        started: float,
        failure: AiFailureCode,
        reason: str,
        *,
        chat: _Chat | None = None,
    ) -> AiFailure:
        return AiFailure(
            failure=failure,
            reason=reason,
            receipt=self._receipt(task, request_id, format, max(1, attempts), started, chat),
        )


class _Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message


class _Chat(BaseModel):
    model: str | None = None
    choices: list[_Choice]
    usage: _Usage | None = None

    @property
    def content(self) -> str:
        return self.choices[0].message.content


class _ErrorDetail(BaseModel):
    message: str = ""


class _Error(BaseModel):
    error: _ErrorDetail


def _request_body[T: BaseModel](
    instance: AiInstance, messages: list[AiMessage], result: type[T], format: AiFormat
) -> dict[str, object]:
    schema = _portable_schema(result.model_json_schema())
    admitted_messages = [message.model_dump() for message in messages]
    body: dict[str, object] = {"model": instance.model, "messages": admitted_messages}
    if format == "schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": result.__name__, "strict": True, "schema": schema},
        }
    elif format == "json":
        body["response_format"] = {"type": "json_object"}
    else:
        admitted_messages.insert(
            0, {"role": "system", "content": f"Return only JSON matching this schema: {schema}"}
        )
    if instance.request_policy == "openrouter-free-zdr":
        body["provider"] = {
            "zdr": True,
            "require_parameters": True,
        }
    return body


def _portable_schema(value: object) -> object:
    if isinstance(value, list):
        return [_portable_schema(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _portable_schema(item)
            for key, item in value.items()
            if key not in {"title", "default"}
        }
    return value


def _admit_chat_response(response: NetResponse) -> _Chat | str:
    try:
        chat = _Chat.model_validate_json(response.body)
    except ValidationError:
        return "AI endpoint returned an invalid response envelope"
    if not chat.choices or not chat.content.strip():
        return "AI endpoint returned no message content"
    return chat


def _unsupported_format(response: NetResponse, format: AiFormat) -> bool:
    if response.status_code not in {400, 404, 422}:
        return False
    try:
        message = _Error.model_validate_json(response.body).error.message.casefold()
    except ValidationError:
        return False
    names = {"schema": ("json_schema", "response_format"), "json": ("json_object",), "prompt": ()}[
        format
    ]
    return (
        bool(names)
        and any(name in message for name in names)
        and any(
            marker in message for marker in ("unsupported", "unknown", "unrecognized", "invalid")
        )
    )


def _context_length_rejected(response: NetResponse) -> bool:
    if response.status_code not in {400, 413, 422}:
        return False
    try:
        message = _Error.model_validate_json(response.body).error.message.casefold()
    except ValidationError:
        return False
    return any(
        marker in message
        for marker in ("context length", "context_length", "too many tokens", "maximum context")
    )


def read_credential(
    reference: CredentialRef | None,
    *,
    env_files: list[Path] | None = None,
    environ: Mapping[str, str] | None = None,
    keyring: Keyring | None = None,
) -> Credential | None:
    if reference is None:
        return None
    if reference.kind == "env":
        env = environ if environ is not None else os.environ
        found = env.get(reference.name) or _dotenv_credential(reference.name, env_files or [])
        if not found:
            raise ValueError(f"missing environment credential {reference.name}")
        return Credential(found)
    if reference.kind == "keyring":
        if keyring is None or keyring.priority <= 0:
            raise ValueError("keyring credential requires an admitted secure backend")
        found = keyring.get_password("vctx", reference.name)
        if not found:
            raise ValueError(f"missing stored credential {reference.name}")
        return Credential(found)
    raise AssertionError("unreachable credential kind")


def select_ai_route(
    *,
    task: AiTask,
    instance_name: str | None,
    auto: bool,
    instances: Mapping[str, AiInstanceConfig],
    offline: bool,
    auto_credential: CredentialRef | None = None,
) -> AiRoute | None:
    if offline:
        return None
    if instance_name is not None:
        config = instances.get(instance_name)
        if config is None:
            return None
        instance = config.admit(instance_name)
        host = urlsplit(instance.base_url).hostname
        return AiRoute(
            task=task,
            selected=(
                "local" if host in {"localhost", "127.0.0.1", "::1"} else "configured-online"
            ),
            instance=instance,
            reason=f"selected configured AI instance {instance_name}",
        )
    if not auto:
        return None
    if auto_credential is None:
        return None
    instance = AiInstance(
        name="openrouter-free",
        provider_id="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        credential=auto_credential,
        request_policy="openrouter-free-zdr",
        format="schema",
        privacy="zdr",
        cost="free",
        require_parameters=True,
    )
    return AiRoute(
        task=task,
        selected="free-online",
        instance=instance,
        reason="authenticated OpenRouter free/ZDR route",
    )


def admit_ai_binding(
    *,
    task: AiTask,
    instance_name: str | None,
    auto: bool,
    instances: Mapping[str, AiInstanceConfig],
    offline: bool,
    env_files: list[Path] | None = None,
    keyring: Keyring | None = None,
) -> AiBinding | None:
    if offline:
        return None
    credential = None
    auto_ref = None
    if auto:
        for candidate in (
            CredentialRef("env:OPENROUTER_API_KEY"),
            CredentialRef("keyring:openrouter"),
        ):
            try:
                credential = read_credential(candidate, env_files=env_files, keyring=keyring)
            except ValueError:
                continue
            auto_ref = candidate
            break
    route = select_ai_route(
        task=task,
        instance_name=instance_name,
        auto=auto,
        instances=instances,
        offline=False,
        auto_credential=auto_ref,
    )
    if route is None:
        return None
    if route.instance.credential is not None and credential is None:
        try:
            credential = read_credential(
                route.instance.credential,
                env_files=env_files,
                keyring=keyring,
            )
        except ValueError:
            return None
    return AiBinding(route, credential)


class Keyring(Protocol):
    priority: float

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_KEYRING_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _credential_parts(reference: str) -> tuple[Literal["env", "keyring"], str]:
    kind, separator, name = reference.partition(":")
    valid = separator and (
        kind == "env"
        and _ENV_NAME.fullmatch(name)
        or kind == "keyring"
        and _KEYRING_NAME.fullmatch(name)
    )
    if not valid:
        raise ValueError("credential must be env:NAME or keyring:NAME")
    return ("env" if kind == "env" else "keyring"), name


def _dotenv_credential(name: str, paths: list[Path]) -> str | None:
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = raw.strip().partition("=")
            if separator and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]
                return value or None
    return None


def _net_purpose(task: AiTask) -> NetPurpose:
    return task
