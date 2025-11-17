import numpy as np
from numpy.typing import NDArray
from ..dsp import compute_dominant_singular_vectors


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
    return mu, c


def svd_estimate_shift(phase_vec: NDArray[np.float32], N: int,
                       phase_windowing=None) -> float:
    """
    Estimate from a 1-D phase vector the space displacement.
    (Sua implementação original)
    """
    from numpy.typing import NDArray

    def linear_regression(x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[float, float]:
        R = np.ones((x.size, 2))
        R[:, 0] = x
        x_sol = np.linalg.lstsq(R, y, rcond=None)
        mu, c = x_sol[0]
        return mu, c

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

    mu, _ = linear_regression(x, y)
    return mu * N / (2 * np.pi)


def svd_method(fft_beg: NDArray[np.complex64],
               fft_end: NDArray[np.complex64],
               M: int, N: int,
               phase_windowing=None,
               unwrap_method='itoh1982',
               svd_method = "svds",
               **svd_kwargs) -> tuple[float, float]:
    """
    Estimate displacement using SVD-based phase correlation.

    Parameters
    ----------
    ...
    svd_method : {"svds", "power_iteration"}
        Method to compute singular vectors
    **svd_kwargs : dict
        Additional arguments for compute_dominant_singular_vectors
        (e.g., max_iter, tol for power_iteration)
    """
    from ..dsp import normalized_cps
    from ..phase_unwrap import phase_unwrap

    Q = normalized_cps(fft_beg, fft_end)

    qu, s, qv = compute_dominant_singular_vectors(Q, method=svd_method, **svd_kwargs)

    ang_qu = phase_unwrap(np.angle(qu[:, 0]), unwrap_method)
    ang_qv = phase_unwrap(np.angle(qv[0, :]), unwrap_method)

    deltax = svd_estimate_shift(ang_qv, M, phase_windowing)
    deltay = svd_estimate_shift(ang_qu, N, phase_windowing)

    return deltax, deltay
