from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "vctx"
TESTS = ROOT / "tests"


def python_lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


source_files = {path: python_lines(path) for path in SOURCE.rglob("*.py")}
test_files = {path: python_lines(path) for path in TESTS.rglob("*.py")}
source_lines = sum(source_files.values())
test_lines = sum(test_files.values())
limit = source_lines // 2
app_max = max(lines for path, lines in source_files.items() if path.parent == SOURCE / "app")
print(
    f"nonblank Python LOC: source={source_lines}/8000, tests={test_lines}/3500, "
    f"ratio={test_lines}/{source_lines} (limit {limit}), "
    f"files={max(source_files.values())}/600, app={app_max}/400, "
    f"test-file={max(test_files.values())}/300"
)
failures = [
    message
    for failed, message in (
        (source_lines > 8000, "source Python LOC exceeds 8000"),
        (test_lines > 3500, "test Python LOC exceeds 3500"),
        (test_lines > limit, "test Python LOC exceeds half of source Python LOC"),
        (max(source_files.values()) > 600, "a source module exceeds 600 LOC"),
        (app_max > 400, "an app composer exceeds 400 LOC"),
        (max(test_files.values()) > 300, "a test module exceeds 300 LOC"),
    )
    if failed
]
if failures:
    raise SystemExit("; ".join(failures))
