import numpy as np
from numpy.typing import NDArray


def phase_unwrap(phase_wrapped: NDArray[np.float32], method: str = "itoh1982") -> NDArray[np.float32]:
    """
    Interface for applying different types of phase unwrapping algorithms.

    Parameters
    ----------
    phase_wrapped : NDArray[np.float32]
         A 1-D Array representing the wrapped phase values ranging from -2pi to 2pi.
    method : str, optional
        Phase unwrapping method, by default "itoh1982"

    Returns
    -------
    NDArray[np.float32]
        A 1-D Array representing unwrapped phase values ranging from -infinity to infinity.

    Raises
    ------
    NotImplementedError
        If the ``method`` is not among the implemented phase unwrapping methods. 
    """

    match method:
        case "itoh1982":
            return itoh1982_method(phase_wrapped)
        case "numpy":
            return np.unwrap(phase_wrapped)
        case _:
            raise NotImplementedError(f"Phase unwrap method {method} not valid.")


def itoh1982_method(phase_vec: NDArray[np.float32], factor: float = 0.7) -> NDArray[np.float32]:
    """
    Phase unwrapping method based on [1]_.

    Parameters
    ----------
    phase_vec : NDArray[np.float32]
        A 1-D Array representing the wrapped phase values ranging from -2pi to 2pi.
    factor : float, optional
        How close the 1st order difference between two consecutive phase samples must be to 2pi to be considered a wrapping event, by default 0.7

    Returns
    -------
    NDArray[np.float32]
        A 1-D Array representing unwrapped phase values ranging from -infinity to infinity.
        
    References
    ----------
    .. [1] Itoh, K. (1982). Analysis of the phase unwrapping algorithm. Applied optics, 21(14), 2470-2470.
    """
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
            phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)
