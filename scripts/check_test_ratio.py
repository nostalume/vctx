from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def python_lines(directory: str) -> int:
    return sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in (ROOT / directory).rglob("*.py")
    )


source_lines = python_lines("src")
test_lines = python_lines("tests")
limit = source_lines // 2
print(f"test/source LOC: {test_lines}/{source_lines} (limit {limit})")
if test_lines > limit:
    raise SystemExit("test Python LOC exceeds half of source Python LOC")
