import numpy as np
import math
import numba


@numba.njit(cache=True, nogil=True)
def make_reverse_lookup(arr):
    """ """
    return {arr[i]: i for i in range(len(arr))}


@numba.njit(cache=True, nogil=True)
def _histogram2d(y, x, weight, bins_y, bins_x):
    """Histogram 2D"""
    n = len(x)

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    hist = np.zeros((ny, nx), dtype=numba.float64)

    for i in range(n):
        if y[i] < bins_y[0]:
            continue
        if y[i] > bins_y[-1]:
            continue
        if x[i] < bins_x[0]:
            continue
        if x[i] > bins_x[-1]:
            continue
        ind_y = int((y[i] - bins_y[0]) / step_y)
        ind_x = int((x[i] - bins_x[0]) / step_x)
        hist[ind_y, ind_x] += weight[i]

    return hist


@numba.njit(parallel=True, cache=True, nogil=True)
def _histogram2d_parallel(y, x, weight, bins_y, bins_x, n_threads):
    """Histogram 2D — parallel version using per-thread local arrays."""
    n = len(x)
    if (n < 10000) or (n_threads == 1):
        return _histogram2d(y, x, weight, bins_y, bins_x)

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    local_hists = np.zeros((n_threads, ny, nx), dtype=numba.float64)

    chunk = (n + n_threads - 1) // n_threads

    for t in numba.prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            if y[i] < bins_y[0] or y[i] > bins_y[-1]:
                continue
            if x[i] < bins_x[0] or x[i] > bins_x[-1]:
                continue
            ind_y = int((y[i] - bins_y[0]) / step_y)
            ind_x = int((x[i] - bins_x[0]) / step_x)
            local_hists[t, ind_y, ind_x] += weight[i]

    hist = np.zeros((ny, nx), dtype=numba.float64)
    for t in range(n_threads):
        hist += local_hists[t]
    return hist


def histogram2d(y, x, weight, bins_y, bins_x, n_threads=0):
    # resolve thread count outside numba: get_num_threads() in jitted code prevents caching
    if n_threads <= 0:
        n_threads = numba.get_num_threads()
    return _histogram2d_parallel(y, x, weight, bins_y, bins_x, n_threads)


@numba.njit(cache=True, nogil=True)
def _histogram1d(x, weight, bins):
    """Histogram 1D"""
    n = len(x)

    nx = len(bins) - 1

    step_x = bins[1] - bins[0]

    hist = np.zeros(nx, dtype=numba.float64)

    for i in range(n):
        if x[i] < bins[0]:
            continue
        if x[i] > bins[-1]:
            continue
        ind_x = int((x[i] - bins[0]) / step_x)
        hist[ind_x] += weight[i]

    return hist


@numba.njit(parallel=True, cache=True, nogil=True)
def _histogram1d_parallel(x, weight, bins, n_threads):
    """Histogram 1D — parallel version using per-thread local arrays."""
    n = len(x)
    if n < 10000:
        return _histogram1d(x, weight, bins)

    nx = len(bins) - 1
    step_x = bins[1] - bins[0]

    local_hists = np.zeros((n_threads, nx), dtype=numba.float64)

    chunk = (n + n_threads - 1) // n_threads

    for t in numba.prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            if x[i] < bins[0] or x[i] > bins[-1]:
                continue
            ind_x = int((x[i] - bins[0]) / step_x)
            local_hists[t, ind_x] += weight[i]

    hist = np.zeros(nx, dtype=numba.float64)
    for t in range(n_threads):
        hist += local_hists[t]
    return hist


def histogram1d(x, weight, bins, n_threads=0):
    # resolve thread count outside numba: get_num_threads() in jitted code prevents caching
    if n_threads <= 0:
        n_threads = numba.get_num_threads()
    return _histogram1d_parallel(x, weight, bins, n_threads)


@numba.njit(cache=True, nogil=True)
def _histogram2d_accumulate(y, x, weight, bins_y, bins_x, hist):
    """Histogram 2D accumulate into a 2D array — serial."""
    n = len(x)

    if n == 0:
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        ind_y = math.floor((y[i] - bins_y[0]) / step_y)
        if ind_y < 0 or ind_y >= ny:
            continue
        ind_x = math.floor((x[i] - bins_x[0]) / step_x)
        if ind_x < 0 or ind_x >= nx:
            continue
        hist[ind_y, ind_x] += weight[i]


@numba.njit(cache=True, nogil=True)
def _histogram2d_accumulate_mask(y, x, weight, bins_y, bins_x, hist, mask_dict):
    """Histogram 2D accumulate into a 1D array via mask_dict — serial."""
    n = len(x)

    if n == 0:
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        ind_y = math.floor((y[i] - bins_y[0]) / step_y)
        if ind_y < 0 or ind_y >= ny:
            continue
        ind_x = math.floor((x[i] - bins_x[0]) / step_x)
        if ind_x < 0 or ind_x >= nx:
            continue
        j = mask_dict[ind_y, ind_x]
        if j > -1:
            hist[j] += weight[i]


@numba.njit(parallel=True, cache=True, nogil=True)
def _histogram2d_accumulate_parallel(y, x, weight, bins_y, bins_x, hist, n_threads):
    """Histogram 2D accumulate into a 2D array — parallel."""
    n = len(x)

    if n == 0:
        return
    if n < 10000:
        _histogram2d_accumulate(y, x, weight, bins_y, bins_x, hist)
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    local_hists = np.zeros((n_threads, hist.shape[0], hist.shape[1]), dtype=numba.float64)

    chunk = (n + n_threads - 1) // n_threads

    for t in numba.prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            ind_y = math.floor((y[i] - bins_y[0]) / step_y)
            if ind_y < 0 or ind_y >= ny:
                continue
            ind_x = math.floor((x[i] - bins_x[0]) / step_x)
            if ind_x < 0 or ind_x >= nx:
                continue
            local_hists[t, ind_y, ind_x] += weight[i]

    for t in range(n_threads):
        hist += local_hists[t]


def histogram2d_accumulate(y, x, weight, bins_y, bins_x, hist, n_threads=0):
    # resolve thread count outside numba: get_num_threads() in jitted code prevents caching
    if n_threads <= 0:
        n_threads = numba.get_num_threads()
    return _histogram2d_accumulate_parallel(y, x, weight, bins_y, bins_x, hist, n_threads)


@numba.njit(parallel=True, cache=True, nogil=True)
def _histogram2d_accumulate_mask_parallel(y, x, weight, bins_y, bins_x, hist, mask_dict, n_threads):
    """Histogram 2D accumulate into a 1D array via mask_dict — parallel."""
    n = len(x)

    if n == 0:
        return
    if n < 10000:
        _histogram2d_accumulate_mask(y, x, weight, bins_y, bins_x, hist, mask_dict)
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    local_hists = np.zeros((n_threads, hist.shape[0]), dtype=numba.float64)

    chunk = (n + n_threads - 1) // n_threads

    for t in numba.prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            ind_y = math.floor((y[i] - bins_y[0]) / step_y)
            if ind_y < 0 or ind_y >= ny:
                continue
            ind_x = math.floor((x[i] - bins_x[0]) / step_x)
            if ind_x < 0 or ind_x >= nx:
                continue
            j = mask_dict[ind_y, ind_x]
            if j == -1:
                continue
            local_hists[t, j] += weight[i]

    for t in range(n_threads):
        hist += local_hists[t]


def histogram2d_accumulate_mask(y, x, weight, bins_y, bins_x, hist, mask_dict, n_threads=0):
    # resolve thread count outside numba: get_num_threads() in jitted code prevents caching
    if n_threads <= 0:
        n_threads = numba.get_num_threads()
    return _histogram2d_accumulate_mask_parallel(y, x, weight, bins_y, bins_x, hist, mask_dict, n_threads)
