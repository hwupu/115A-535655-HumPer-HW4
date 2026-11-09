import numpy as np

from humper import store


def test_round_trip_preserves_values_and_types(tmp_path):
    grid = np.arange(49, dtype=np.float32).reshape(7, 7)
    store.save_record("a", {"grid": grid}, tmp_path)
    loaded = store.load_record("a", tmp_path)["grid"]
    assert loaded.dtype == np.float32
    assert (loaded == grid).all()


def test_save_leaves_no_temporary_file(tmp_path):
    store.save_record("a", {"grid": np.zeros(2)}, tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["a.npz"]


def test_has_record_needs_every_key(tmp_path):
    store.save_record("a", {"x": np.zeros(1), "y": np.zeros(1)}, tmp_path)
    assert store.has_record("a", {"x", "y"}, tmp_path)
    assert not store.has_record("a", {"x", "z"}, tmp_path)
    assert not store.has_record("missing", {"x"}, tmp_path)
