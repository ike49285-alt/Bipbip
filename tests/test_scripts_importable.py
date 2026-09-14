"""Every script must at least be able to START.

`scripts/swing_compare.py` imported `bipbip` with no sys.path setup and so
raised ModuleNotFoundError on every invocation - `python3 scripts/x.py` puts
scripts/ on sys.path, never the repo root. It also took
`cross_sectional_run.py` down with it, which imports it as a sibling.

Nothing caught that, because no test runs the scripts and the failure is at
import time rather than in any function. These are static checks: cheap, and
they fail for exactly the reason the scripts did.
"""
import ast
import pathlib

import pytest

SCRIPTS = sorted(pathlib.Path(__file__).resolve().parents[1].glob("scripts/*.py"))
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ids(paths):
    return [p.name for p in paths]


@pytest.mark.parametrize("path", SCRIPTS, ids=_ids(SCRIPTS))
def test_every_script_parses(path):
    """A syntax error in a research driver is invisible until someone runs it."""
    ast.parse(path.read_text(), filename=str(path))


def _imports_bipbip(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "bipbip" for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "bipbip":
                return True
    return False


def _inserts_repo_root(tree):
    """Look for sys.path.insert(..., <something>.parents[1] ...)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "insert"):
            continue
        if "parents" in ast.unparse(node):
            return True
    return False


@pytest.mark.parametrize("path", SCRIPTS, ids=_ids(SCRIPTS))
def test_a_script_importing_bipbip_puts_the_repo_root_on_the_path(path):
    """`parents[1]`, not `parent`. The latter reaches sibling scripts and not
    the package, which is the shape of the bug in cross_sectional_run.py: it
    inserted scripts/ only, so its sibling import worked and `bipbip` did not.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    if not _imports_bipbip(tree):
        pytest.skip("does not import bipbip")
    assert _inserts_repo_root(tree), (
        f"{path.name} imports bipbip but never puts the repo root on sys.path; "
        "it cannot run as `python3 scripts/...`")


@pytest.mark.parametrize("path", SCRIPTS, ids=_ids(SCRIPTS))
def test_a_script_importing_a_sibling_also_puts_scripts_on_the_path(path):
    """The mirror of the above: importing a sibling module by bare name only
    works if scripts/ itself is on the path."""
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    names = {p.stem for p in SCRIPTS} - {path.stem, "__init__"}
    siblings = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } & names
    if not siblings:
        pytest.skip("imports no sibling script")
    assert "sys.path.insert" in src, (
        f"{path.name} imports sibling {sorted(siblings)} without putting "
        "scripts/ on sys.path")
