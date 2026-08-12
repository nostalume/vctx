from __future__ import annotations


class VctxError(Exception):
    exit_code = 1


class ConfigError(VctxError):
    exit_code = 2


class UnsupportedSourceError(VctxError):
    exit_code = 3


class SourceConflictError(VctxError):
    exit_code = 3

    def __init__(self, source_id: str, key: str) -> None:
        super().__init__("conflicting revisions were admitted for one source identity")
        self.source_id = source_id
        self.key = key


class NoTranscriptError(VctxError):
    exit_code = 4


class InvalidTranscriptError(VctxError):
    exit_code = 4


class OutputExistsError(VctxError):
    exit_code = 5


class CacheError(VctxError):
    exit_code = 5


class OfflineSourceError(VctxError):
    exit_code = 6


class ProviderError(VctxError):
    exit_code = 7


class OperationCancelledError(VctxError):
    exit_code = 130


class EmptyChunksError(VctxError):
    exit_code = 1
