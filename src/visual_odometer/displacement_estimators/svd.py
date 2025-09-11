import numpy as np
from numpy import ndarray

from scipy.signal import convolve
from scipy.sparse.linalg import svds as svds_cpu

def normalize_product(F, G):
    Q = F * np.conj(G)
    Q /= np.abs(Q)
    return Q

def phase_fringe_filter(cross_power_spectrum: ndarray, window_size: tuple = (5, 5), threshold: float = 0.03) -> ndarray:
    # Aplica o filtro de média para reduzir o ruído
    filtered_spectrum = convolve(cross_power_spectrum, np.ones(window_size) / np.prod(window_size), mode='same')  # Alterado de 'constant' para 'same'

    # Calcula a diferença entre o espectro original e o filtrado
    diff_spectrum = cross_power_spectrum - filtered_spectrum

    # Aplica o limiar para identificar as regiões de pico
    peak_mask = np.abs(diff_spectrum) > threshold

    # Atenua as regiões de pico no espectro original
    phase_filtered_spectrum = cross_power_spectrum.copy()
    phase_filtered_spectrum[peak_mask] *= 0.5  # Reduz a amplitude nas regiões de pico

    return phase_filtered_spectrum


def linear_regression(x: ndarray, y: ndarray) -> (float, float):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c


def phase_unwrapping(phase_vec: ndarray, factor: float = 0.7) -> ndarray:
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
            phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)


def svd_estimate_shift(phase_vec: ndarray, N: int, phase_windowing=None) -> float:
    phase_unwrapped = np.unwrap(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2

    if phase_windowing == "central":
        x = r[M - 50:M + 50]
        y = phase_unwrapped[M - 50:M + 50]
    elif phase_windowing == "initial":
        x = r[M - 80:M - 10]
        y = phase_unwrapped[M - 80:M - 10]
    else:
        x = r
        y = phase_unwrapped

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    mu = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    delta = mu * N / (2 * np.pi)
    return float(delta)

def svd_method(fft_beg, fft_end, M: int, N: int, phase_windowing=None, finge_filter=True) -> (float, float):

    Q = normalize_product(fft_beg, fft_end)

    if finge_filter is True:
        Q = phase_fringe_filter(Q)

    qu, s, qv = svds_cpu(Q, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltax = svd_estimate_shift(ang_qv, M, phase_windowing)
    deltay = svd_estimate_shift(ang_qu, N, phase_windowing)

    return deltax, deltay


