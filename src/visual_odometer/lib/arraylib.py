
import numpy as np
from scipy.sparse.linalg import svds as svds_cpu

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cupyx_linalg
    _cupy_available = True
except ImportError:
    cp = None
    cupyx_linalg = None
    _cupy_available = False

# Estado global atual
_current_xp = np
_current_svds = svds_cpu
_current_use_gpu = False

# Funções para retornar o módulo correto com base na configuração
def get_array_module(use_gpu: bool = False):
    return cp if use_gpu and _cupy_available else np

def get_svds(use_gpu: bool = False):
    if use_gpu and _cupy_available:
        return cupyx_linalg.svds
    else:
        return svds_cpu

# Define o backend a ser usado globalmente
def set_backend(use_gpu: bool):
    global _current_xp, _current_svds, _current_use_gpu

    if use_gpu is True and _cupy_available is False:
        print("[arraylib] ⚠️ Atenção: GPU foi solicitada, mas CuPy não está disponível. Revertendo para CPU.")

    _current_xp = get_array_module(use_gpu)
    _current_svds = get_svds(use_gpu)
    _current_use_gpu = use_gpu

# Acesso global ao backend atual
def xp_backend():
    return _current_xp

def svds_backend():
    return _current_svds

def is_gpu():
    return _current_use_gpu

# Converte valores de volta para float automaticamente
def as_float(value):
    return float(value.get()) if is_gpu() and hasattr(value, "get") else float(value)
