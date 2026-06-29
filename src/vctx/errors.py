from __future__ import annotations


class VctxError(Exception):
    exit_code = 1


class UnsupportedSourceError(VctxError):
    exit_code = 3


class NoTranscriptError(VctxError):
    exit_code = 4


class InvalidTranscriptError(VctxError):
    exit_code = 4


class OutputExistsError(VctxError):
    exit_code = 5


class EmptyChunksError(VctxError):
    exit_code = 1
