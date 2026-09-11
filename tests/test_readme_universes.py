"""The universe sizes the README quotes, pinned against the code that defines them.

This file exists because a README number rotted silently. The survivorship
paragraph said "the archive holds 309 symbols" and "273 of those symbols
cannot support the test" - the first had drifted to 550 as collectors ran, and
the second was largecap250's size rather than a subset of the archive at all.

The split is deliberate. Universe membership is CODE, so it is pinned here and
cannot drift without a failing test. The archive's symbol count is DATA that
grows on every collector run, so pinning it would make the suite fail for
succeeding; the README states it with a date instead.
"""
from bipbip.data.universe import UNIVERSES, get_universe, survivorship_warning


def test_the_clean_universe_is_the_size_the_readme_claims():
    assert len(get_universe("etf_wide")) == 34


def test_largecap250_is_the_size_the_readme_claims():
    assert len(get_universe("largecap250")) == 273


def test_etf_wide_is_the_only_universe_the_readme_calls_clean():
    """The README's argument rests on exactly one universe being usable for
    selection. If another became `mild`, the paragraph would need rewriting
    rather than the number bumping."""
    mild = {n for n in UNIVERSES if "mild" in survivorship_warning(n)}
    assert mild == {"etf_core", "etf_wide", "etf_all"}
    # ...and every stock list carries the loud warning, which is the
    # load-bearing half. Note the CODE says "SURVIVORSHIP BIAS"; "SEVERE" is
    # the prose shorthand used in README and CLAUDE.md, and matching on it here
    # would pass for the wrong reason or fail for none.
    for name in ("megacap", "everything", "largecap250", "liquid", "full"):
        assert "SURVIVORSHIP BIAS" in survivorship_warning(name), name


def test_no_universe_is_silently_unflagged():
    """A universe with no warning at all would read as safe."""
    for name in UNIVERSES:
        w = survivorship_warning(name)
        assert w and ("mild" in w or "SURVIVORSHIP BIAS" in w), (name, w)


def test_every_universe_names_itself_in_its_own_warning():
    """A warning quoting the wrong universe is worse than none, because it
    reads as a specific check that was actually performed on something else."""
    for name in UNIVERSES:
        assert f"({name})" in survivorship_warning(name), name
