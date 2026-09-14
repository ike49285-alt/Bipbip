"""A script must not be able to destroy a committed result by being stopped.

`rotation_null.py` and `panel_null100.py` write the CSVs that hold this repo's
headline nulls. Both opened the real path in "w", which truncates it the moment
the script starts - so an interrupted run leaves a partial file where the
evidence used to be, before a single row has been computed.

That is not hypothetical. A 90-second cap during an audit sweep cut
data/rotation_null.csv from 203 rows to 40, and only git noticed.

These are static checks against the pattern, because the real behaviour takes
minutes to reproduce.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: Scripts that write a committed result file rather than a scratch one.
RESULT_WRITERS = ["scripts/rotation_null.py", "scripts/panel_null100.py"]


@pytest.mark.parametrize("rel", RESULT_WRITERS)
def test_the_destination_is_a_partial_file_not_the_committed_one(rel):
    src = (ROOT / rel).read_text()
    assert ".partial" in src, (
        f"{rel} writes its result path directly; an interrupted run truncates "
        "the committed file")
    assert "os.replace" in src, (
        f"{rel} never renames the temporary file into place")


@pytest.mark.parametrize("rel", RESULT_WRITERS)
def test_the_rename_happens_after_the_file_is_closed(rel):
    """Renaming inside the `with` block would publish a file still being
    written, which is the same failure with extra steps."""
    tree = ast.parse((ROOT / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            body = ast.unparse(node)
            assert "os.replace" not in body, (
                f"{rel} renames while the file is still open")


@pytest.mark.parametrize("rel", RESULT_WRITERS)
def test_the_committed_result_it_writes_is_actually_tracked(rel):
    """If the default output were untracked, none of this would matter - and
    the test would be pinning nothing. Both defaults are committed files."""
    import subprocess
    tree = ast.parse((ROOT / rel).read_text())
    default = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "")
                == "add_argument"
                and node.args and getattr(node.args[0], "value", "") == "--out"):
            for kw in node.keywords:
                if kw.arg == "default":
                    default = kw.value.value
    assert default, f"{rel} has no --out default to check"
    tracked = subprocess.run(["git", "ls-files", default], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    assert tracked == default, f"{default} is not tracked; this guard is moot"


def test_partial_files_are_ignored_so_a_killed_run_leaves_no_mess():
    assert "*.partial" in (ROOT / ".gitignore").read_text()
