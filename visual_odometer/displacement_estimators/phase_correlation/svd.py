import numpy as np
from numpy.typing import NDArray

from scipy.sparse.linalg import svds as svds_cpu
from visual_odometer.dsp import normalized_cps

from .common import phase_unwrap


def linear_regression(x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[float, float]:
    r"""
    Linear regression assuming

    .. math::
        y(x) = \mu x + c

    Parameters
    ----------
    x : NDArray[np.float32]
        1-D Array representing horizontal coordinates.
    y : NDArray[np.float32]
        1-D Array representing vertical coordinates.

    Returns
    -------
    tuple[float, float]
        Angular (:math:`\mu`) and linear (:math:`c`) coefficients of the slope.
    """
    R = np.ones((x.size, 2))
    R[:, 0] = x
    x_sol = np.linalg.lstsq(R, y, rcond=None)
    mu, c = x_sol[0]
    residuals = np.sum(x_sol[1])

    return mu, c, residuals


def svd_estimate_shift(phase_vec: NDArray[np.float32], N: int, phase_windowing = None) -> float:
    """
    Estimate from a 1-D phase vector the space displacement.

    Parameters
    ----------
    phase_vec : NDArray[np.float32]
        A 1-D Array representing unwrapped phase vector.
    N : int
        Size of the original image size (horizontal or vertical) which the displacement is being estimated on.
    phase_windowing : {"central", "initial", None}, optional
        Type of windowing applied to the phase vector to extract, by default None

    Returns
    -------
    float
        Horizontal or vertical displacement proportional to the phase slope, assuming linear phase.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented windowing methods.

    """
    r = np.arange(0, phase_vec.size)
    M = r.size // 2

    match phase_windowing:
        case "central":
            x = r[M - 50:M + 50]
            y = phase_vec[M - 50:M + 50]
        case "initial":
            x = r[M - 80:M - 10]
            y = phase_vec[M - 80:M - 10]
        case None:
            x = r
            y = phase_vec
        case _:
            raise ValueError(f"Invalid windowing method: {phase_windowing}")


    mu, c, residuals = linear_regression(x, y)
    return mu * N / (2 * np.pi), residuals


def svd_method(fft_beg, fft_end, M: int, N: int, phase_windowing = None, unwrap_method = 'itoh1982') -> tuple[float, float]:
    r"""
    Estimate vertical and horizontal displacement vector :math:`[\Delta y, \Delta x]^T` between two spatially shifted spectra:

    .. math::

        I_{end}[y, x] = I_{beg}[y - \Delta y, x - \Delta x]

    where `fft_beg` is :math:`\mathcal{F}\{ I_{beg}[y, x]\} (u, v)` and `fft_end` is :math:`\mathcal{F}\{ I_{end}[y, x]\} (u, v)`.

    The algorithm behind the estimation it is subspace identification extension to the phase correlation method :cite:`hoge_subspace_2003`.

    Parameters
    ----------
    fft_beg : NDArray[np.complex64]
        A 2-D array representing the spectrum of :math:`I_{beg}[y, x]`
    fft_end : NDArray[np.complex64]
        A 2-D array representing the spectrum of :math:`I_{end}[y,x]`
    M : int
        Number of rows of the original image.
    N : int
        Number of columns of the original image.
    phase_windowing : {"central", "initial", None}, optional
        Type of windowing applied to the phase vector to extract, by default None
    unwrap_method : {"itoh1982", "numpy"}, optional
        Phase unwrapping method, by default "itoh1982".

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical displacement values :math:`[\Delta y, \Delta x]^T`, assuming :math:`I[y, x]`.
        
    References
    ----------
    :cite:`hoge_subspace_2003` Hoge, W. S. (2003). A subspace identification extension to the phase correlation method [MRI application]. IEEE transactions on medical imaging, 22(2), 277-280. :doi:`10.1109/TMI.2002.808359`
    """
    Q = normalized_cps(fft_beg, fft_end)

    qu, s, qv = svds_cpu(Q, k=1)
    ang_qu = phase_unwrap(np.angle(qu[:, 0]), unwrap_method)
    ang_qv = phase_unwrap(np.angle(qv[0, :]), unwrap_method)

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltax, residualsx = svd_estimate_shift(ang_qv, M, phase_windowing)
    deltay, residualsy = svd_estimate_shift(ang_qu, N, phase_windowing)

    quality = np.max(np.abs([residualsx, residualsy]))

    return deltax, deltay, quality
