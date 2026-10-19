"""Explicit diagnostic backend selection; CPU always uses the reference."""
import os

def use_triton(tensor):
    backend=os.environ.get('GAST_NAV_MAP_BACKEND','triton')
    if backend not in ('triton','torch'):raise ValueError('unknown GAST_NAV_MAP_BACKEND: '+backend)
    return tensor.is_cuda and backend=='triton'
