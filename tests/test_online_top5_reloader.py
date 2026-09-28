import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src/gd_lab/rl"))
from online_top5 import Top5CriteriaReloader  # noqa: E402


def test_reloader_cache_invalid_edit_and_reload(tmp_path: Path):
    path = tmp_path / "online_top5.json"
    path.write_text('{"max_base_contact_rate": 0.05}')
    reloader = Top5CriteriaReloader(path)

    criteria, message = reloader.refresh()
    assert criteria["max_base_contact_rate"] == 0.05
    assert criteria["top5_min_spacing"] == 100
    assert message.startswith("Reloaded")
    assert reloader.refresh()[1] is None

    path.write_text("{")
    criteria, message = reloader.refresh()
    assert criteria["max_base_contact_rate"] == 0.05
    assert "retaining last valid values" in message

    path.write_text('{"max_base_contact_rate": 0.04, "top5_min_spacing": 50}')
    criteria, message = reloader.refresh()
    assert criteria["max_base_contact_rate"] == 0.04
