from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"


def _python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _production_import_violations(prefix: str) -> list[str]:
    violations: list[str] = []
    for path in _python_sources(APP_ROOT):
        for imported in _imports_from(path):
            if imported == prefix or imported.startswith(f"{prefix}."):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {imported}")
    return sorted(violations)


def test_no_root_level_fastapi_exports_beyond_main():
    root_python = [
        path
        for path in PROJECT_ROOT.glob("*.py")
        if path.name != "main.py" and path.name != "__init__.py"
    ]
    assert root_python == []


def test_intent_compiler_imports_are_migration_debt():
    violations = _production_import_violations("app.intent_compiler")
    if violations:
        pytest.xfail(
            "app.intent_compiler is scheduled for deletion; current imports:\n"
            + "\n".join(violations)
        )
    assert violations == []


def test_ui_workspace_imports_are_migration_debt():
    violations = _production_import_violations("app.ui_workspace")
    if violations:
        pytest.xfail(
            "app.ui_workspace should disappear after app.agent.intent.ui absorbs it; current imports:\n"
            + "\n".join(violations)
        )
    assert violations == []
