import torch

from plant_disease.checkpoint_io import load_checkpoint


def test_round_trips_a_plain_state_dict(tmp_path):
    path = tmp_path / "plain.pt"
    state = {"classes": ["a", "b"], "model_state_dict": {"w": torch.arange(4.0)}}
    torch.save(state, path)
    loaded = load_checkpoint(path)
    assert loaded["classes"] == ["a", "b"]
    assert torch.equal(loaded["model_state_dict"]["w"], torch.arange(4.0))


class _Custom:
    def __init__(self):
        self.value = 7


def test_falls_back_when_safe_load_refuses(tmp_path, capsys):
    path = tmp_path / "custom.pt"
    torch.save({"obj": _Custom()}, path)
    loaded = load_checkpoint(path)
    assert loaded["obj"].value == 7
    assert "compatibility mode" in capsys.readouterr().out
