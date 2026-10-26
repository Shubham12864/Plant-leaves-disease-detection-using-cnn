"""Shared fixtures. Tests run against the repository, not a copy of it."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def repo() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def manifests(repo: Path) -> Path:
    return repo / "data" / "manifests"
