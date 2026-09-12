from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BufferedReader
from threading import Thread
from typing import BinaryIO, Protocol

_CAPTURE_BYTES = 1024 * 1024


class _Capture(Protocol):
    def seek(self, offset: int, /) -> int: ...
    def read(self, n: int = -1, /) -> bytes: ...


@dataclass(frozen=True)
class SupervisedResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def run_supervised(
    command: Sequence[str],
    payload: bytes,
    *,
    timeout_s: float,
    environment: dict[str, str] | None = None,
    stderr_sink: Callable[[bytes], None] | None = None,
) -> SupervisedResult:
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write(payload)
        stdin.seek(0)
        process = subprocess.Popen(
            command,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            ),
            env=environment,
        )
        assert process.stdout is not None and process.stderr is not None
        readers = (
            Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
            Thread(
                target=_drain,
                args=(process.stderr, stderr),
                kwargs={"sink": stderr_sink},
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()
        expired = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            expired = True
            _terminate_tree(process)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join(timeout=2)
        return SupervisedResult(
            returncode=124 if expired else process.returncode,
            stdout=_read_capture(stdout),
            stderr=_read_capture(stderr),
        )


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def _read_capture(stream: _Capture) -> bytes:
    stream.seek(0)
    return stream.read(_CAPTURE_BYTES)


def _drain(
    source: BufferedReader,
    capture: BinaryIO,
    *,
    sink: Callable[[bytes], None] | None = None,
) -> None:
    remaining = _CAPTURE_BYTES
    while block := source.read1(64 * 1024):
        if remaining:
            admitted = block[:remaining]
            capture.write(admitted)
            remaining -= len(admitted)
        if sink is not None:
            try:
                sink(block)
            except OSError:
                sink = None
