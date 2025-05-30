import numpy as np
from numpy import ndarray

from scipy.signal import convolve
from scipy.sparse.linalg import svds as svds_cpu
from cupyx.scipy.sparse.linalg import svds as svds_gpu

try:
    import cupy as cp
except:
    pass

def normalize_product(F, G, use_gpu=False):
    if use_gpu:
        Q = F * cp.conj(G)
        Q /= cp.abs(Q)
        return Q
    else:
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


def svd_estimate_shift(phase_vec: ndarray, N: int, phase_windowing=None, use_gpu=False) -> float:
    if use_gpu:
        xp = cp
    else:
        xp = np

    phase_unwrapped = xp.unwrap(phase_vec)
    r = xp.arange(0, phase_unwrapped.size)
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

    x_mean = xp.mean(x)
    y_mean = xp.mean(y)
    mu = xp.sum((x - x_mean) * (y - y_mean)) / xp.sum((x - x_mean) ** 2)
    delta = mu * N / (2 * xp.pi)

    return float(delta.get()) if use_gpu else float(delta)

def randomized_svd_gpu(A, k=1, n_iter=20):
    m, n = A.shape
    G = cp.random.randn(n, k)
    Y = A @ G
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    Q, _ = cp.linalg.qr(Y)
    B = Q.T @ A
    U_hat, S, Vt = cp.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    return U[:, :k], S[:k], Vt[:k, :]


def svd_method(fft_beg, fft_end, M: int, N: int, phase_windowing=None, finge_filter=True,
               use_gpu=False) -> (float, float):

    Q = normalize_product(fft_beg, fft_end, use_gpu=use_gpu)

    #if finge_filter is True:
        #Q = phase_fringe_filter(Q)
    if use_gpu:
        qu, s, qv = randomized_svd_gpu(Q, k=1)
        ang_qu = cp.angle(qu[:, 0])
        ang_qv = cp.angle(qv[0, :])
    else:
        qu, s, qv = svds_cpu(Q, k=1)
        ang_qu = np.angle(qu[:, 0])
        ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:


    deltay = svd_estimate_shift(ang_qu, M, phase_windowing, use_gpu)
    deltax = svd_estimate_shift(ang_qv, N, phase_windowing, use_gpu)

    return deltax, deltay
