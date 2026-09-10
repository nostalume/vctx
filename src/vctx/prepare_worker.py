from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vctx.app.pack import prepare_context_pack
from vctx.app.progress import configure_logging
from vctx.config import PrepareRequest
from vctx.errors import VctxError
from vctx.supervise import SupervisedResult, run_supervised

_REQUEST_BYTES = 1024 * 1024


class WorkerRequest(BaseModel):
    request: PrepareRequest
    verbose: bool = False
    debug: bool = False
    log_file: Path | None = None
    profile_json: Path | None = None


def supervise_prepare(
    request: PrepareRequest,
    *,
    timeout_s: int,
    verbose: bool,
    debug: bool,
    log_file: Path | None,
    profile_json: Path | None,
    stderr_sink: Callable[[bytes], None] | None = None,
) -> SupervisedResult:
    payload = (
        WorkerRequest(
            request=request,
            verbose=verbose,
            debug=debug,
            log_file=log_file,
            profile_json=profile_json,
        )
        .model_dump_json()
        .encode()
    )
    return run_supervised(
        [sys.executable, "-m", "vctx.prepare_worker"],
        payload,
        timeout_s=timeout_s,
        stderr_sink=stderr_sink,
    )


def main() -> int:
    body = sys.stdin.buffer.read(_REQUEST_BYTES + 1)
    if len(body) > _REQUEST_BYTES:
        print("error: supervised request exceeds byte limit", file=sys.stderr)
        return 2
    try:
        command = WorkerRequest.model_validate_json(body)
    except ValidationError:
        print("error: invalid supervised prepare request", file=sys.stderr)
        return 2
    configure_logging(
        verbose=command.verbose,
        debug=command.debug,
        log_file=command.log_file,
        profile_json=command.profile_json,
    )
    try:
        result = prepare_context_pack(command.request)
    except VctxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    print(result.render_cli(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
