import numpy as np

from numpy.typing import NDArray
from scipy.sparse.linalg import svds as svds_cpu
from ..phase_unwrap import phase_unwrap
from ..dsp import normalized_cps
from .svd import svd_estimate_shift


def apply_projection_phase_filter(CPS: NDArray[np.complex64], R: int, dx_min: float, dx_max: float, dy_min: float,
                                  dy_max: float, keep_dim: bool = False) -> NDArray[np.complex64]:
    r"""
    Apply time-domain filtering on the cross-power spectrum between the spectra of :math:`I_{beg}[y, x]` and
    :math:`I_{beg}[y, x]` where :math:`I_{end}[y,x]=I_{beg}[y - \Delta y, x - \Delta x]` :cite:`keller_projection-based_2007`.

    Parameters
    ----------
    CPS : NDArray[np.complex64]
        A 2-D Array with complex-valued entries representing the result of cross-power spectrum operation between
        :math:`I_{beg}[y, x]` and :math:`I_{end}[y, x]`. The `CPS` zero-frequency is at the image center (it was fftshifted).
    R : int
        Safe margin between maximum detectable displacement dx and dy.
    dx_min : float
        Minimum detectable displacement along horizontal axis.
    dx_max : float
        Maximum detectable displacement along horizontal axis.
    dy_min : float
        Minimum detectable displacement along vertical axis.
    dy_max : float
        Maximum detectable displacement along vertical axis.
    keep_dim : bool, optional
        Whether to keep the dimensions of the original image or discard filtered values during low-pass filtering, by default False

    Returns
    -------
    NDArray[np.complex64]
        Filtered `CPS`.
        
    References
    ----------
     :cite:`keller_projection-based_2007` Keller, Y., & Averbuch, A. (2007). A projection-based extension to phase correlation image alignment. Signal processing, 87(1), 124-133. :doi:`10.1016/j.sigpro.2006.04.013`
    """

    Cn_t = np.fft.ifft2(CPS)
    M, N = Cn_t.shape

    delta_x = (dx_max - dx_min) + R
    delta_y = (dy_max - dy_min) + R

    # Create an indexing safe lower and upper bound for the low-pass filter:
    lb_y, ub_y = np.clip(M // 2 - delta_y // 2, a_min=0, a_max=M), np.clip(M // 2 + delta_y // 2, a_min=0, a_max=M)
    lb_x, ub_x = np.clip(N // 2 - delta_x // 2, a_min=0, a_max=N), np.clip(N // 2 + delta_x // 2, a_min=0, a_max=N)

    if keep_dim:
        mask = np.zeros_like(Cn_t, dtype=int)
        mask[lb_y: ub_y, lb_x: ub_x] = 1
        return mask * Cn_t
    else:
        return np.fft.fftshift(np.fft.fftshift(Cn_t)[lb_y: ub_y, lb_x: ub_x])  # reduce the dimension


def proj_svd_method(fft_beg: NDArray[np.complex64], fft_end: NDArray[np.complex64], M: int, N: int, R: int = 6, dx_min: int=0, dx_max: int=64, dy_min: int=0, dy_max: int=48,
                    phase_windowing  = None, unwrap_method  = "itoh1982") -> tuple[float, float]:
    r"""
    Estimate vertical and horizontal displacement vector :math:`[\Delta y, \Delta x]^T` between two spatially shifted spectra:

    .. math::

        I_{end}[y, x] = I_{beg}[y - \Delta y, x - \Delta x]

    where `fft_beg` is :math:`\mathcal{F}\{ I_{beg}[y, x]\} (u, v)` and `fft_end` is :math:`\mathcal{F}\{ I_{end}[y, x]\} (u, v)`.

    The algorithm behind the estimation it is the projection-based phase correlation :cite:`keller_projection-based_2007`.

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
    R : int, optional
        Margin between maximum defined displacement value (dx_max or dy_max) and maximum capable displacement value, by default 6
    dx_min : int, optional
        Minimum expected horizontal displacement (x-axis), by default 0
    dx_max : int, optional
        Maximum expected horizontal displacement (x-axis), by default 64
    dy_min : int, optional
        Minimum expected vertical displacement (y-axis), by default 0
    dy_max : int, optional
        Maximum expected vertical displacement (y-axis), by default 48
    phase_windowing : {}, optional
        Type of window to be applied on the cross-power spectrum phase, by default None
    unwrap_method : {"itoh1982", "numpy"}, optional
        Phase unwrapping method, by default "itoh1982"

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical displacement values :math:`[\Delta y, \Delta x]^T`, assuming :math:`I[y, x]`.
        
    References
    ----------
    :cite:`keller_projection-based_2007` Keller, Y., & Averbuch, A. (2007). A projection-based extension to phase correlation image alignment. Signal processing, 87(1), 124-133.

    """
    
    Q = normalized_cps(fft_beg, fft_end)

    Cn_t_proj = apply_projection_phase_filter(Q, R, dx_min, dx_max, dy_min, dy_max)

    Q_filtered = np.fft.fft2(Cn_t_proj)

    qu, s, qv = svds_cpu(Q_filtered, k=1)
    ang_qu = phase_unwrap(np.angle(qu[:, 0]), unwrap_method)
    ang_qv = phase_unwrap(np.angle(qv[0, :]), unwrap_method)

    M_ = M * Q_filtered.shape[0] / Q.shape[0]
    N_ = N * Q_filtered.shape[1] / Q.shape[1]

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltax = svd_estimate_shift(ang_qv, int(M_), phase_windowing)
    deltay = svd_estimate_shift(ang_qu, int(N_), phase_windowing)

    return deltax, deltay
