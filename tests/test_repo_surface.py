from importlib import import_module
import tomllib
from pathlib import Path

import pytest


def test_web_entrypoints_removed():
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "app.py").exists()
    assert not (repo_root / "static" / "index.html").exists()


def test_claude_agent_sdk_is_the_packaging_surface():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert "claude-agent-sdk>=0.0.20" in dependencies
    assert not any("deepagents" in dependency for dependency in dependencies)
    assert not any("langgraph" in dependency for dependency in dependencies)


def test_pyproject_mentions_claude_agent_sdk_and_not_deepagents():
    pyproject = Path("pyproject.toml").read_text()

    assert "claude-agent-sdk" in pyproject
    assert "deepagents" not in pyproject
    assert "langgraph" not in pyproject


def test_agent_package_is_importable():
    import_module("arxiv_rag.agent")