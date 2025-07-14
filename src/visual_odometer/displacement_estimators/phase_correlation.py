import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

def phase_correlation_method(fft_beg, fft_end, use_gpu=False):
    xp = cp if use_gpu and 'cp' in globals() else np

    # Cross-power spectrum
    R = fft_end * xp.conj(fft_beg)
    R /= xp.maximum(xp.abs(R), 1e-10)  # evitar divisão por zero

    # Correlation (IFFT)
    corr = xp.fft.ifft2(R)
    corr = xp.fft.fftshift(corr)
    corr_abs = xp.abs(corr)

    # Pico (sem subpixel)
    max_idx = xp.unravel_index(xp.argmax(corr_abs), corr.shape)

    # Centro
    mid_y, mid_x = corr.shape[0] // 2, corr.shape[1] // 2

    # Deslocamento inteiro
    dx = max_idx[1] - mid_x
    dy = max_idx[0] - mid_y

    return float(dx), float(dy)

