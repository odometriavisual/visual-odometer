import numpy as np
from numpy.typing import NDArray

from scipy.sparse.linalg import svds as svds_cpu
from ..phase_unwrap import phase_unwrap
from ..dsp import normalized_cps


def linear_regression(x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[float, float]:
    """
    Linear regression assuming y(x) = mu*x + c.

    Parameters
    ----------
    x : NDArray[np.float32]
        1-D Array representing horizontal coordinates.
    y : NDArray[np.float32]
        1-D Array representing vertical coordinates.

    Returns
    -------
    tuple[float, float]
        Angular (mu) and linear (c) coefficients of the slope.
    """
    R = np.ones((x.size, 2))
    R[:, 0] = x
    x_sol = np.linalg.lstsq(R, y)
    mu, c = x_sol[0]
    return mu, c


def svd_estimate_shift(phase_vec: NDArray[np.float32], N: int, phase_windowing: str = "") -> float:
    """
    Estimate from a 1-D phase vector the space displacement.

    Parameters
    ----------
    phase_vec : NDArray[np.float32]
        A 1-D Array representing unwrapped phase vector.
    N : int
        Size of the original image size (horizontal or vertical) which the displacement is beeing estimated on.
    phase_windowing : str, optional
        Type of windowing applied to the phase vector to extract, by default ""

    Returns
    -------
    float
        Horizontal or vertical displacement proportional to the phase slope, assuming linear phase.
    """
    r = np.arange(0, phase_vec.size)
    M = r.size // 2

    if phase_windowing == "central":
        x = r[M - 50:M + 50]
        y = phase_vec[M - 50:M + 50]
    elif phase_windowing == "initial":
        x = r[M - 80:M - 10]
        y = phase_vec[M - 80:M - 10]
    else:
        x = r
        y = phase_vec

    mu, _ = linear_regression(x, y)
    return mu * N / (2 * np.pi)


def svd_method(fft_beg, fft_end, M: int, N: int, phase_windowing: str = "", unwrap_method: str = 'itoh1982') -> tuple[float, float]:
    """
    Estimate displacement between two spatialy shifted images, i.e.:
    
    I_end[y, x] = I_beg[y - dy, x - dx]
    
    where fft_beg = FFT(I_beg) and fft_end = FFT(I_end), by using subspace identification extension to the phase correlation method [1]_.

    Parameters
    ----------
    fft_beg : _type_
         A 2-D array represeting the spectrum of I_beg
    fft_end : _type_
         A 2-D array represeting the spectrum of I_end
    M : int
        Number of rows of the original image.
    N : int
        Number of columns of the original image.
    phase_windowing : str, optional
        Type of window to be applied on the cross-power spectrum phase, by default ""
    unwrap_method : str, optional
        Phase unwrapping method, by default "itoh1982"

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical (x and y) displacement values, assuming I[y, x].
        
    References
    ----------
    .. [1] Hoge, W. S. (2003). A subspace identification extension to the phase correlation method [MRI application]. IEEE transactions on medical imaging, 22(2), 277-280.
    """
    Q = normalized_cps(fft_beg, fft_end)

    qu, s, qv = svds_cpu(Q, k=1)
    ang_qu = phase_unwrap(np.angle(qu[:, 0]), unwrap_method)
    ang_qv = phase_unwrap(np.angle(qu[0, :]), unwrap_method)

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltax = svd_estimate_shift(ang_qv, M, phase_windowing)
    deltay = svd_estimate_shift(ang_qu, N, phase_windowing)

    return deltax, deltay
