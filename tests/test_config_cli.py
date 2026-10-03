"""The root program selects a JSON config by file name or by a stdin number."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
CONFIG_DIR = ROOT / "configs"


def _run(args, stdin_text=None):
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        input=stdin_text,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )


def _field(stdout: str, name: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(f"{name}:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"{name} was not in stdout:\n{stdout}")


def test_file_name_uses_that_configs_warmup():
    for filename in ("warmup_4.json", "warmup_8.json", "tiny.json"):
        document = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        result = _run([filename])
        assert result.returncode == 0, result.stderr
        assert _field(result.stdout, "config") == filename
        assert int(_field(result.stdout, "pool_length")) == document["warmup"]
        indices = [int(part) for part in _field(result.stdout, "far_point_indices").split(",") if part]
        assert indices
        assert all(0 <= index < document["warmup"] for index in indices)
        assert len(indices) <= document["warmup"]
        assert _field(result.stdout, "stream_query_count").isdigit()


def test_stdin_number_selects_the_listed_config():
    listed = _run([], stdin_text="1\n")
    assert listed.returncode == 0, listed.stderr
    menu = {}
    for line in listed.stdout.splitlines():
        if ". " in line and line.split(".", 1)[0].isdigit():
            number, filename = line.split(". ", 1)
            menu[int(number)] = filename.strip()
    assert 1 in menu
    chosen = menu[1]
    assert _field(listed.stdout, "config") == chosen
    document = json.loads((CONFIG_DIR / chosen).read_text(encoding="utf-8"))
    assert int(_field(listed.stdout, "pool_length")) == document["warmup"]

    second = _run([], stdin_text="2\n")
    assert second.returncode == 0, second.stderr
    assert _field(second.stdout, "config") == menu[2]
    assert menu[1] != menu[2]


def test_unknown_name_and_bad_number_do_not_run_a_config():
    unknown = _run(["does_not_exist.json"])
    assert unknown.returncode != 0
    assert "pool_length:" not in unknown.stdout
    assert "config:" not in unknown.stdout
    assert "Unknown config" in unknown.stderr

    disguised = _run(["missing/tiny.json"])
    assert disguised.returncode != 0
    assert "pool_length:" not in disguised.stdout
    assert "config:" not in disguised.stdout

    out_of_range = _run([], stdin_text="99\n")
    assert out_of_range.returncode != 0
    assert "pool_length:" not in out_of_range.stdout
    assert "config:" not in out_of_range.stdout
    assert "out of range" in out_of_range.stderr

    not_a_number = _run([], stdin_text="nope\n")
    assert not_a_number.returncode != 0
    assert "pool_length:" not in not_a_number.stdout
    assert "config:" not in not_a_number.stdout
