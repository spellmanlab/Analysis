"""Utility function for peri-event time histograms."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Sequence, Tuple, Union


def plotHist(
    series: Sequence[float],
    events: Iterable[int],
    window: Union[int, Tuple[int, int]],
    *,
    ax: plt.Axes | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot peri-event histogram with mean and SEM.

    Parameters
    ----------
    series:
        One-dimensional array-like time series of numerical values.
    events:
        Indices into ``series`` representing event times.
    window:
        If an integer ``w`` is provided, include ``w`` points before and after
        each event.  Alternatively, provide a ``(pre, post)`` tuple specifying
        different extents before and after the event index.
    ax:
        Optional matplotlib Axes on which to draw.  If ``None`` (default), a
        new figure and axes are created.

    Returns
    -------
    time_points, mean_trace, sem_trace
        Arrays representing the relative time points, the mean across events,
        and the standard error of the mean respectively.

    Notes
    -----
    Events that do not have enough data on either side to fill the requested
    window are skipped.
    """
    series = np.asarray(series)
    events = np.asarray(list(events), dtype=int)

    if isinstance(window, int):
        pre, post = window, window
    else:
        pre, post = window

    # Build matrix of segments around each event
    segments = []
    for idx in events:
        start = idx - pre
        end = idx + post + 1
        if start < 0 or end > len(series):
            continue
        segments.append(series[start:end])

    if not segments:
        raise ValueError("No events with sufficient window in series.")

    stack = np.vstack(segments)
    mean_trace = stack.mean(axis=0)
    sem_trace = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    time_points = np.arange(-pre, post + 1)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(time_points, mean_trace, color="C0")
    ax.fill_between(
        time_points,
        mean_trace - sem_trace,
        mean_trace + sem_trace,
        color="C0",
        alpha=0.3,
    )
    ax.set_xlabel("Time (points)")
    ax.set_ylabel("Mean value")
    ax.set_title("Peri-event Time Histogram")

    return time_points, mean_trace, sem_trace
