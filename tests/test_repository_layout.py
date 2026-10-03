"""Keep published entrypoints discoverable and free from stale local references."""

from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = sorted((ROOT / "scripts").rglob("*.sh"))


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("document", ["README.md", "docs/history/README.md"])
def test_navigation_links_exist(document):
    source = ROOT / document
    links = re.findall(r"\]\(([^)]+)\)", source.read_text())
    for target in links:
        if target.startswith(("https://", "http://", "#")):
            continue
        path = target.split("#", 1)[0]
        assert (source.parent / path).is_file(), f"{document}: {target}"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_literal_repository_entrypoints_exist(script):
    # Ignore absolute paths inside external repositories and variable-built paths.
    pattern = r"(?<![\w/])(?:scripts/[\w./-]+\.(?:py|sh)|configs/[\w./-]+\.yaml)"
    references = re.findall(pattern, script.read_text())
    for target in references:
        assert (ROOT / target).is_file(), f"{script.relative_to(ROOT)}: {target}"
