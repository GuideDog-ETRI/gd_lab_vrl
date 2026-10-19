import pytest,torch
from gd_lab.gast_nav.map_backend import use_triton

def test_backend_selection_fails_closed(monkeypatch):
    class Device:
        is_cuda=True
    monkeypatch.setenv('GAST_NAV_MAP_BACKEND','torch');assert not use_triton(Device())
    monkeypatch.setenv('GAST_NAV_MAP_BACKEND','triton');assert use_triton(Device())
    assert not use_triton(torch.zeros(1))
    monkeypatch.setenv('GAST_NAV_MAP_BACKEND','typo')
    with pytest.raises(ValueError):use_triton(Device())
