"""
The `objective` module implements objective functions to be used in optimization problems.
"""

from typing import Literal, Optional, Tuple
import jax.numpy as jnp
import jax.scipy as jsp


def compute_xcorr2d(signal0: jnp.ndarray, signal1: jnp.ndarray, shift: Tuple[Optional[int], Optional[int]] = (None, None)):
    """Computes cross-correlation between 2D arrays.

    Args:
        signal0 (jnp.ndarray): 2D array.
        signal1 (jnp.ndarray): 2D array
        shift (Tuple[Optional[int], Optional[int]], optional): Specifies the shift of interest along the axis (0 means no shift). Defaults to (None, None).

    Raises:
        ValueError:

    Returns:
        jnp.ndarray: Either the cross-correlation as a 2D array (if no shift is given), a 1D array (if shift is given along one axis), or a 0D array.
    """

    xcorr2d = jsp.signal.correlate2d(
        signal0, signal1) / jsp.signal.correlate2d(signal0, signal0).max()

    if shift == (None, None):
        return xcorr2d
    elif shift[1] is None and shift[0] is not None:
        # Slice along 1 axis
        return xcorr2d[signal1.shape[0] - 1 + shift[0], :]
    elif shift[0] is None and shift[1] is not None:
        # Slice along 0 axis
        return xcorr2d[:, signal1.shape[1] - 1 + shift[1]]
    elif shift[0] is not None and shift[1] is not None:
        return xcorr2d[signal1.shape[0] - 1 + shift[0], signal1.shape[1] - 1 + shift[1]]
    else:
        raise ValueError


def compute_xcorr(signal0: jnp.ndarray, signal1: jnp.ndarray, shift: Optional[int] = None):
    """Computes cross-correlation between 1D arrays.

    Args:
        signal0 (jnp.ndarray): 1D array
        signal1 (jnp.ndarray): 1D array
        shift (Optional[int], optional): Specifies the shift of interest (0 means no shift). Defaults to None.

    Returns:
        jnp.ndarray: Either the cross-correlation as a 1D array (if no shift is given), or a 0D array.
    """

    xcorr = jsp.signal.correlate(
        signal0, signal1) / jsp.signal.correlate(signal0, signal0).max()

    return xcorr if shift is None else xcorr[signal1.shape[0] - 1 + shift]


def compute_max_xcorr2d_at_shift(signal0: jnp.ndarray, signal1: jnp.ndarray, shift: int, shift_axis: Literal[0, 1] = 0):
    """Computes maximum 2d cross-correlation between two 2D arrays with a fixed shift along the specified axis.

    Args:
        signal0 (jnp.ndarray): 2D array.
        signal1 (jnp.ndarray): 2D array.

    Returns:
        Tuple[float, int]: tuple of max and delay of cross-correlation (delay>0 means that signal1 is delayed with respect to signal0).
    """

    xcorr2d_slice = compute_xcorr2d(signal0, signal1, shift=(
        shift, None) if shift_axis == 0 else (None, shift))
    max_xcorr, max_index = xcorr2d_slice.max(), xcorr2d_slice.argmax()

    return max_xcorr, -(max_index + 1 - signal1.shape[1 if shift_axis == 0 else 0])


def compute_space_time_xcorr(space_time0: jnp.ndarray, space_time1: jnp.ndarray):
    """Computes space-time cross-correlation between two space-time 2D arrays.

    Args:
        space_time0 (jnp.ndarray): 2D array interpreted as a space-time array (space: axis 0, time: axis 1).
        space_time1 (jnp.ndarray): 2D array of the same kind as space_time0.

    Returns:
        Tuple[float, int]: tuple of max cross-correlation and corresponding time delay.
    """

    return compute_max_xcorr2d_at_shift(space_time0, space_time1, shift=0, shift_axis=0)
