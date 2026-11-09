import torch

from humper import get_device


def test_explicit_device_wins():
    assert get_device("cpu") == torch.device("cpu")


def test_env_override(monkeypatch):
    monkeypatch.setenv("HUMPER_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")


def test_default_device_runs_a_tensor(monkeypatch):
    monkeypatch.delenv("HUMPER_DEVICE", raising=False)
    device = get_device()
    assert torch.ones(2, device=device).sum().item() == 2
