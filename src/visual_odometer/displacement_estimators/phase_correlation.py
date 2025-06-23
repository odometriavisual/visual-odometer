import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

def subpixel_peak_position(corr_abs, peak, xp):
    y, x = peak

    def parabolic(f, x):
        """Retorna o deslocamento subpixel do vértice da parábola definida por três pontos."""
        if x <= 0 or x >= f.shape[0] - 1:
            return 0.0  # borda, não é possível interpolar
        denom = (f[x-1] - 2*f[x] + f[x+1])
        if denom == 0:
            return 0.0
        return 0.5 * (f[x-1] - f[x+1]) / denom

    dx = dy = 0.0
    if 1 <= x < corr_abs.shape[1] - 1:
        dx = parabolic(corr_abs[y, :], x)
    if 1 <= y < corr_abs.shape[0] - 1:
        dy = parabolic(corr_abs[:, x], y)

    return x + dx, y + dy

def phase_correlation_method(fft_beg, fft_end, use_gpu=False):
    xp = cp if use_gpu and cp else np

    # Cross-power spectrum
    R = fft_end * xp.conj(fft_beg)
    R /= xp.maximum(xp.abs(R), 1e-10)  # evitar divisão por zero

    # Correlation (IFFT)
    corr = xp.fft.ifft2(R)
    corr = xp.fft.fftshift(corr)
    corr_abs = xp.abs(corr)

    # Pico
    max_idx = xp.unravel_index(xp.argmax(corr_abs), corr.shape)
    sub_x, sub_y = subpixel_peak_position(corr_abs, max_idx, xp)

    # Centro
    mid_y, mid_x = corr.shape[0] // 2, corr.shape[1] // 2

    # Deslocamento
    dx = sub_x - mid_x
    dy = sub_y - mid_y

    return float(dx), float(dy)
