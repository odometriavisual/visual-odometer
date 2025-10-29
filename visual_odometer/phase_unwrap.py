import numpy as np
from numpy.typing import NDArray


def phase_unwrap(phase_wrapped: NDArray[np.float32], method = "itoh1982") -> NDArray[np.float32]:
    r"""
    Interface for applying different types of phase unwrapping algorithms.

    Parameters
    ----------
    phase_wrapped : NDArray[np.float32]
         A 1-D Array representing the wrapped phase values that is limited to the :math:`]-\pi, +\pi]` interval.
    method : {"itoh1982", "numpy"}, optional
        Phase unwrapping method, by default "itoh1982".

    Returns
    -------dsadas
    NDArray[np.float32]
        A 1-D Array representing unwrapped phase values that could range from :math::math:`]-\infty, +\infty]`.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented phase unwrapping methods. 
    """

    match method:
        case "itoh1982":
            return itoh1982_method(phase_wrapped)
        case "numpy":
            return np.unwrap(phase_wrapped)
        case _:
            raise ValueError(f"Phase unwrap method {method} not valid.")


def itoh1982_method(phase_vec: NDArray[np.float32], factor: float = 0.7) -> NDArray[np.float32]:
    r"""
    Phase unwrapping method based on :cite:`itoh_analysis_1982`.

    Parameters
    ----------
    phase_vec : NDArray[np.float32]
         A 1-D Array representing the wrapped phase values that is limited to the :math:`]-\pi, +\pi]` interval.
    factor : float, optional
        A constant that defines how close the first-order difference between two consecutive phase samples must be to :math:`2\pi` to be considered a wrapping event, by default 0.7

    Returns
    -------
    NDArray[np.float32]
        A 1-D Array representing unwrapped phase values that could range from :math::math:`]-\infty, +\infty]`.
        
    References
    ----------
    :cite:`itoh_analysis_1982` Itoh, K. (1982). Analysis of the phase unwrapping algorithm. Applied optics, 21(14), 2470-2470. :doi:`10.1364/AO.21.002470`
    """
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
            phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)
