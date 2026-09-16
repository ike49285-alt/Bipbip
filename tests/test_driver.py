from pathlib import Path

import pytest

from bot.driver import INBOUND, OUTBOUND, LocalDriver, seed_demo


@pytest.fixture
def driver(tmp_path):
    with LocalDriver(tmp_path / "t.db") as d:
        yield d


def test_post_returns_id_and_lands_on_timeline(driver):
    pid = driver.post(Path("content/img/a.jpg"), "radiator again")
    assert pid
    tl = driver.timeline()
    assert len(tl) == 1
    assert tl[0].caption == "radiator again"
    assert tl[0].image == "content/img/a.jpg"


def test_post_ids_are_unique(driver):
    ids = {driver.post(Path("a.jpg"), f"n{i}") for i in range(20)}
    assert len(ids) == 20


def test_recent_posts_is_newest_first_and_respects_limit(driver):
    for i in range(5):
        driver.post(Path(f"{i}.jpg"), f"caption {i}")
    recent = driver.recent_posts(limit=3)
    assert [p.caption for p in recent] == ["caption 4", "caption 3", "caption 2"]


def test_fetch_dms_returns_only_inbound(driver):
    driver.receive_dm("u/1", "hello")
    driver.send_dm("u/1", "hi back")
    inbound = driver.fetch_dms()
    assert [m.body for m in inbound] == ["hello"]
    assert all(m.inbound for m in inbound)


def test_fetch_dms_since_cursor_advances(driver):
    first = driver.receive_dm("u/1", "one")
    assert [m.body for m in driver.fetch_dms(since=first.id)] == []
    driver.receive_dm("u/1", "two")
    assert [m.body for m in driver.fetch_dms(since=first.id)] == ["two"]


def test_thread_is_chronological_and_mixed_direction(driver):
    driver.receive_dm("u/1", "are you real?")
    driver.send_dm("u/1", "no, ai persona")
    driver.receive_dm("u/1", "fair enough")
    convo = [(m.direction, m.body) for m in driver.thread("u/1")]
    assert convo == [
        (INBOUND, "are you real?"),
        (OUTBOUND, "no, ai persona"),
        (INBOUND, "fair enough"),
    ]


def test_threads_are_isolated(driver):
    driver.receive_dm("u/1", "from one")
    driver.receive_dm("u/2", "from two")
    assert [m.body for m in driver.thread("u/1")] == ["from one"]
    assert [m.body for m in driver.thread("u/2")] == ["from two"]
    assert {t["thread_id"] for t in driver.threads()} == {"u/1", "u/2"}


def test_thread_limit_keeps_the_most_recent(driver):
    for i in range(10):
        driver.receive_dm("u/1", f"m{i}")
    tail = driver.thread("u/1", limit=3)
    assert [m.body for m in tail] == ["m7", "m8", "m9"]


def test_state_survives_reopening(tmp_path):
    db = tmp_path / "persist.db"
    with LocalDriver(db) as d:
        d.post(Path("a.jpg"), "before")
        d.receive_dm("u/1", "also before")
    with LocalDriver(db) as d:
        assert [p.caption for p in d.timeline()] == ["before"]
        assert [m.body for m in d.thread("u/1")] == ["also before"]


def test_creates_parent_directory(tmp_path):
    with LocalDriver(tmp_path / "nested" / "deep" / "x.db") as d:
        assert d.db_path.exists()


def test_seed_demo_populates_timeline(driver):
    seed_demo(driver, ["one", "two"])
    assert {p.caption for p in driver.timeline()} == {"one", "two"}


def test_local_driver_satisfies_the_protocol(driver):
    from bot.driver import SocialDriver

    assert isinstance(driver, SocialDriver)
