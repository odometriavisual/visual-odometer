import numpy as np
try:
    import cupy as cp
except:
    pass


def phase_correlation_method(fft_beg, fft_end, use_gpu=False):
    xp = cp if use_gpu and cp else np

    # Cross-power spectrum
    R = fft_end * xp.conj(fft_beg)
    R /= xp.abs(R) + 1e-8  # evitar divisão por zero
    # Correlation (FFT inversa)
    corr = xp.fft.ifft2(R)
    corr = xp.fft.fftshift(corr)  # centraliza o pico
    corr_abs = xp.abs(corr)

    # Localiza o pico
    max_idx = xp.unravel_index(xp.argmax(corr_abs), corr.shape)
    mid_y, mid_x = corr.shape[0] // 2, corr.shape[1] // 2

    # Calcula o deslocamento relativo ao centro
    dy = max_idx[0] - mid_y
    dx = max_idx[1] - mid_x

    return float(dx), float(dy)
