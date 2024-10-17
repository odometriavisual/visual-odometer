import numpy as np
from virtualencoder.visualodometry.dsp_utils import crosspower_spectrum, normalize_product, phase_fringe_filter
from scipy.sparse.linalg import svds
from virtualencoder.visualodometry.image_utils import filter_array_by_maxmin

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def linear_regression(x, y):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c


def phase_unwrapping(phase_vec, factor=0.7):
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
                phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)


def svd_estimate_shift(phase_vec, N, phase_windowing=None):
    # Phase unwrapping:
    phase_unwrapped = phase_unwrapping(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2
    if phase_windowing == None or phase_windowing == False:
        x = r
        y = phase_unwrapped
    elif phase_windowing == "central":
        x = r[M - 50:M + 50]
        y = phase_unwrapped[M - 50:M + 50]
    elif phase_windowing == "initial":
        x = r[M - 80:M - 10]
        y = phase_unwrapped[M - 80:M - 10]
    mu, c = linear_regression(x, y)
    delta = mu * N / (2 * np.pi)
    return delta


def svd_method(img_beg, img_end, frequency_window="Stone_et_al_2001", use_gpu = False):
    M, N = img_beg.shape
    q, Q = crosspower_spectrum(img_end, img_beg, frequency_window)
    if use_gpu:
        if GPU_AVAILABLE:
            # Usar SVD de Cupy
            qu, s, qv = cp.linalg.svd(Q, full_matrices=False)
            # Obter o ângulo dos vetores U e V
            ang_qu = cp.angle(qu[:, 0])
            ang_qv = cp.angle(qv[0, :])
        else:
            raise Exception("Erro, cupy não está instalado, coloque use_gpu como False ou instale o cupy usando pip install cupy-cuda11x")
    else:
        # Usar SVD de CPU (SciPy)
        qu, s, qv = svds(Q, k=1)
        ang_qu = np.angle(qu[:, 0])
        ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M)
    deltax = svd_estimate_shift(ang_qv, N)
    return deltax, deltay

def optimized_svd_method(processed_img_beg, processed_img_end, M, N, phase_windowing=None, finge_filter = True):
    Q = normalize_product(processed_img_end, processed_img_beg)
    if finge_filter is True:
        Q = phase_fringe_filter(Q)

    qu, s, qv = svds(Q, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    # if filter_values: # desativado, resolução inicial do bug nomeado de f153
    #     ang_qu = filter_array_by_maxmin(ang_qu)
    #     ang_qv = filter_array_by_maxmin(ang_qv)

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M, phase_windowing)
    deltax = svd_estimate_shift(ang_qv, N, phase_windowing)
    return deltax, deltay