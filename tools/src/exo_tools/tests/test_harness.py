from exo_tools.harness import Comm, placement_filter


def test_placement_filter_supports_sglang() -> None:
    assert Comm.SGLANG.value == "Sglang"
    assert placement_filter("Sglang", "sglang")
    assert placement_filter("Sglang", "both")
    assert not placement_filter("MlxRing", "sglang")
