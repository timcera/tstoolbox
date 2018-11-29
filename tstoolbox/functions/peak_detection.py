#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from builtins import range
from builtins import str
from builtins import zip

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import numpy as np

import pandas as pd

from past.utils import old_div

from .. import tsutils

warnings.filterwarnings('ignore')


def _boolrelextrema(data, comparator,
                    axis=0, order=1, mode='clip'):
    """Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    comparator(data[n],data[n+1:n+order+1]) = True.

    Parameters
    ----------
    data: ndarray
    comparator: function
        function to use to compare two data points.
        Should take 2 numbers as arguments
    axis: int, optional
        axis over which to select from `data`
    order: int, optional
        How many points on each side to require
        a `comparator`(n,n+x) = True.
    mode: string, optional
        How the edges of the vector are treated.
        'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See numpy.take

    Returns
    -------
    extrema: ndarray
        Indices of the extrema, as boolean array
        of same shape as data. True for an extrema,
        False else.

    See also
    --------
    argrelmax,argrelmin

    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0).tolist()
    [False, False, True, False, False]

    """
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis)
    for shift in range(1, order + 1):
        plus = np.take(data, locs + shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        minus = np.take(data, locs - shift, axis=axis, mode=mode)
        results &= comparator(main, minus)
        if ~results.any():
            return results
    return results


def _argrel(data, axis=0, window=1):
    """Private function to find relative min and max of data."""
    tmpmin = _argrelmin(data, axis=axis, order=window)
    tmpmax = _argrelmax(data, axis=axis, order=window)
    return (list(zip(tmpmax[0],
                     data[tmpmax[0]])),
            list(zip(tmpmin[0],
                     data[tmpmin[0]])))


def _argrelmin(data, axis=0, order=1, mode='clip'):
    """Calculate the relative minima of `data`.

    See also
    --------
    argrelextrema,argrelmax

    """
    return _argrelextrema(data, np.less, axis, order, mode)


def _argrelmax(data, axis=0, order=1, mode='clip'):
    """Calculate the relative maxima of `data`.

    See also
    --------
    argrelextrema,argrelmin

    """
    return _argrelextrema(data, np.greater, axis, order, mode)


def _argrelextrema(data, comparator,
                   axis=0, order=1, mode='clip'):
    """Calculate the relative extrema of `data`.

    Returns
    -------
    extrema: ndarray
        Indices of the extrema, as an array
        of integers (same format as argmin, argmax

    See also
    --------
    argrelmin, argrelmax

    """
    results = _boolrelextrema(data, comparator,
                              axis, order, mode)
    if ~results.any():
        return (np.array([]),) * 2
    else:
        return np.where(results)


def _datacheck_peakdetect(x_axis, y_axis):
    """Check x and y axis, creating an x data_set if necessary."""
    if x_axis is None:
        x_axis = list(range(len(y_axis)))

    if len(y_axis) != len(x_axis):
        raise ValueError("""
*
*   The length of y values must equal the length of x values.  Instead the
*   length of y values is {0} and the length of x values is {1}.
*
""".format(len(y_axis), len(x_axis)))

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def _peakdetect_parabola_fitter(raw_peaks, x_axis, y_axis, points):
    """Perform parabola fitting for the _peakdetect_parabola function.

    keyword arguments:
    raw_peaks -- A list of either the maximum or the minimum peaks, as given
        by the _peakdetect_zero_crossing function, with index used as x-axis
    x_axis -- A numpy list of all the x values
    y_axis -- A numpy list of all the y values
    points -- How many points around the peak should be used during curve
        fitting, must be odd.

    return -- A list giving all the peaks and the fitted waveform, format:
        [[x, y, [fitted_x, fitted_y]]]

    """
    from scipy.optimize import curve_fit

    def func(x, k, tau, m):
        return k * ((x - tau) ** 2) + m

    fitted_peaks = []
    for peak in raw_peaks:
        index = peak[0]
        x_data = x_axis[index - points // 2: index + points // 2 + 1]
        y_data = y_axis[index - points // 2: index + points // 2 + 1]
        # get a first approximation of tau (peak position in time)
        tau = x_axis[index]
        # get a first approximation of peak amplitude
        m = peak[1]

        # build list of approximations
        # k = -m as first approximation?
        p0 = (-m, tau, m)
        popt, _ = curve_fit(func, x_data, y_data, p0)
        # retrieve tau and m i.e x and y value of peak
        x, y = popt[1:3]

        # create a high resolution data set for the fitted waveform
        x2 = np.linspace(x_data[0], x_data[-1], points * 10)
        y2 = func(x2, *popt)

        fitted_peaks.append([x, y, [x2, y2]])

    return fitted_peaks


def _peakdetect(y_axis, x_axis=None, window=24, delta=0):
    """Private peak detection algorithm.

    Converted from/based on a MATLAB script
    at:http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    window -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)

    """
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)

    # perform some checks
    if window < 1:
        raise ValueError('window must be "1" or above in value')
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError('delta must be a positive number')

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'window' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-window],
                                       y_axis[:-window])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        # ####look for max####
        if y < mx-delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+window].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+window >= length:
                    # end is within window no more peaks can be found
                    break
                continue

        # ####look for min####
        if y > mn+delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+window].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+window >= length:
                    # end is within window no more peaks can be found
                    break

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def _peakdetect_fft(y_axis, x_axis, pad_len=5):
    """Private function to calculate FFT peak detection.

    Performs a FFT calculation on the data and zero-pads the results to
    increase the time domain resolution after performing the inverse fft and
    send the data to the '_peakdetect' function for peak
    detection.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as the index 50.234 or similar.

    Will find at least 1 less peak then the '_peakdetect_zero_crossing'
    function, but should result in a more precise value of the peak as
    resolution has been increased. Some peaks are lost in an attempt to
    minimize spectral leakage by calculating the fft between two zero
    crossings for n amount of signal periods.

    The biggest time eater in this function is the ifft and thereafter it's
    the '_peakdetect' function which takes only half the time of the ifft.
    Speed improvement could include to check if 2**n points could be used for
    fft and ifft or change the '_peakdetect' to the
    '_peakdetect_zero_crossing',
    which is maybe 10 times faster than 'peakdetct'. The pro of '_peakdetect'
    is that it results in one less lost peak. It should also be noted that the
    time used by the ifft function can change greatly depending on the input.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    pad_len -- (optional) By how many times the time resolution should be
        increased by, e.g. 1 doubles the resolution. The amount is rounded up
        to the nearest 2 ** n amount (default: 5)

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)

    """
    from scipy import fft, ifft
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    zero_indices = zero_crossings(y_axis, window=11)
    # select a n amount of periods
    last_indice = - 1 - (1 - len(zero_indices) & 1)
    # Calculate the fft between the first and last zero crossing
    # this method could be ignored if the beginning and the end of the signal
    # are discarded as any errors induced from not using whole periods
    # should mainly manifest in the beginning and the end of the signal, but
    # not in the rest of the signal
    if len(zero_indices) < 2:
        fft_data = fft(y_axis)
    else:
        fft_data = fft(y_axis[zero_indices[0]:zero_indices[last_indice]])

    def padd(x, c):
        return x[:len(x) // 2] + [0] * c + x[len(x) // 2:]

    def n(x):
        return (old_div(np.log(x), np.log(2))).astype('i') + 1

    # pads to 2**n amount of samples
    fft_padded = padd(list(fft_data), 2 **
                      n(len(fft_data) * pad_len) - len(fft_data))

    # There is amplitude decrease directly proportional to the sample increase
    sf = old_div(len(fft_padded), float(len(fft_data)))
    # There might be a leakage giving the result an imaginary component
    # Return only the real component
    y_axis_ifft = ifft(fft_padded).real * sf  # (pad_len + 1)
    x_axis_ifft = np.linspace(
        x_axis[zero_indices[0]], x_axis[zero_indices[last_indice]],
        len(y_axis_ifft))
    # get the peaks to the interpolated waveform
    max_peaks, min_peaks = _peakdetect(y_axis_ifft, x_axis_ifft, 500,
                                       delta=abs(np.diff(y_axis).max() * 2))
    # max_peaks, min_peaks = _peakdetect_zero_crossing(y_axis_ifft,
    # x_axis_ifft)

    return [max_peaks, min_peaks]


def _peakdetect_parabola(y_axis, x_axis, points=9):
    """Private function for parabola peak detection.

    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by fitting the model function: y = k (x - tau) ** 2 + m
    to the peaks. The amount of points used in the fitting is set by the
    points argument.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as index 50.234 or similar.

    will find the same amount of peaks as the '_peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    points -- (optional) How many points around the peak should be used during
        curve fitting, must be odd (default: 9)

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a list
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)

    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    # points += 1 - int(points) & 1 slower when int conversion needed

    # get raw peaks
    max_raw, min_raw = _peakdetect_zero_crossing(y_axis)

    # define output variable
    max_peaks = []
    min_peaks = []

    max_ = _peakdetect_parabola_fitter(max_raw, x_axis, y_axis, points)
    min_ = _peakdetect_parabola_fitter(min_raw, x_axis, y_axis, points)

    max_peaks = [[x[0], x[1]] for x in max_]
    # max_fitted = map(lambda x: x[-1], max_)
    min_peaks = [[x[0], x[1]] for x in min_]
    # min_fitted = map(lambda x: x[-1], min_)

    return [max_peaks, min_peaks]


def _peakdetect_sine(y_axis, x_axis, points=9, lock_frequency=False):
    """Private function for sine wave based peak detection.

    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by fitting the model function:
    y = A * sin(2 * pi * f * x - tau) to the peaks. The amount of points used
    in the fitting is set by the points argument.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as index 50.234 or similar.

    will find the same amount of peaks as the '_peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.

    The function might have some problems if the sine wave has a
    non-negligible total angle i.e. a k*x component, as this messes with the
    internal offset calculation of the peaks, might be fixed by fitting a
    k * x + m function to the peaks for offset calculation.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    points -- (optional) How many points around the peak should be used during
        curve fitting, must be odd (default: 9)
    lock_frequency -- (optional) Specifies if the frequency argument of the
        model function should be locked to the value calculated from the raw
        peaks or if optimization process may tinker with it. (default: False)

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)

    """
    from scipy.optimize import curve_fit
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    # points += 1 - int(points) & 1 slower when int conversion needed

    # get raw peaks
    max_raw, min_raw = _peakdetect_zero_crossing(y_axis)

    # define output variable
    max_peaks = []
    min_peaks = []

    # get global offset
    offset = np.mean([np.mean(max_raw, 0)[1], np.mean(min_raw, 0)[1]])
    # fitting a k * x + m function to the peaks might be better
    # offset_func = lambda x, k, m: k * x + m

    # calculate an approximate frequency of the signal
    hz = []
    for raw in [max_raw, min_raw]:
        if len(raw) > 1:
            peak_pos = [x_axis[index] for index in zip(*raw)[0]]
            hz.append(np.mean(np.diff(peak_pos)))
    hz = old_div(1, np.mean(hz))

    # model function
    # if cosine is used then tau could equal the x position of the peak
    # if sine were to be used then tau would be the first zero crossing
    if lock_frequency:
        # This is strange - only difference in the two "func"
        # definitions is the "hz" argument

        def func(x, a, tau):
            return a * np.sin(2 * np.pi * hz * (x - tau) + old_div(np.pi, 2))
    else:

        def func(x, a, hz, tau):
            return a * np.sin(2 * np.pi * hz * (x - tau) + old_div(np.pi, 2))
    # func = lambda x, a, hz, tau: a * np.cos(2 * np.pi * hz * (x - tau))

    # get peaks
    fitted_peaks = []
    for raw_peaks in [max_raw, min_raw]:
        peak_data = []
        for peak in raw_peaks:
            index = peak[0]
            x_data = x_axis[index - points // 2: index + points // 2 + 1]
            y_data = y_axis[index - points // 2: index + points // 2 + 1]
            # get a first approximation of tau (peak position in time)
            tau = x_axis[index]
            # get a first approximation of peak amplitude
            a = peak[1]

            # build list of approximations
            if lock_frequency:
                p0 = (a, tau)
            else:
                p0 = (a, hz, tau)

            # subtract offset from waveshape
            y_data -= offset
            popt, _ = curve_fit(func, x_data, y_data, p0)
            # retrieve tau and a i.e x and y value of peak
            x = popt[-1]
            y = popt[0]

            # create a high resolution data set for the fitted waveform
            x2 = np.linspace(x_data[0], x_data[-1], points * 10)
            y2 = func(x2, *popt)

            # add the offset to the results
            y += offset
            y2 += offset
            y_data += offset

            peak_data.append([x, y, [x2, y2]])

        fitted_peaks.append(peak_data)

    # structure date for output
    max_peaks = [[xval[0], xval[1]] for xval in fitted_peaks[0]]
    # max_fitted = map(lambda x: x[-1], fitted_peaks[0])
    min_peaks = [[xval[0], xval[1]] for xval in fitted_peaks[1]]
    # min_fitted = map(lambda x: x[-1], fitted_peaks[1])

    return [max_peaks, min_peaks]


def _peakdetect_sine_locked(y_axis, x_axis, points=9):
    """Private function for peak detection with locked sine wave.

    Convenience function for calling the '_peakdetect_sine' function with
    the lock_frequency argument as True.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    points -- (optional) How many points around the peak should be used during
        curve fitting, must be odd (default: 9)

    return -- see '_peakdetect_sine'

    """
    return _peakdetect_sine(y_axis, x_axis, points, True)


def _peakdetect_zero_crossing(y_axis, x_axis=None, window=5):
    """Private function for zero crossing peak detection.

    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by dividing the signal into bins and retrieving the
    maximum and minimum value of each the even and odd bins respectively.
    Division into bins is performed by smoothing the curve and finding the
    zero crossings.

    Suitable for repeatable signals, where some noise is tolerated. Executes
    faster than '_peakdetect', although this function will break if the offset
    of the signal is too large. It should also be noted that the first and
    last peak will probably not be found, as this function only can find peaks
    between the first and last zero crossing.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)

    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)

    zero_indices = zero_crossings(y_axis, window=window)
    zero_indices = np.concatenate(([0], zero_indices, [len(y_axis) - 1]))
    zero_indices = np.unique(zero_indices)

    period_lengths = np.diff(zero_indices)

    bins_y = [y_axis[index:index + diff] for index, diff in
              zip(zero_indices, period_lengths)]
    bins_x = [x_axis[index:index + diff] for index, diff in
              zip(zero_indices, period_lengths)]

    even_bins_y = bins_y[::2]
    odd_bins_y = bins_y[1::2]
    even_bins_x = bins_x[::2]
    odd_bins_x = bins_x[1::2]
    hi_peaks_x = []
    lo_peaks_x = []

    # check if even bin contains maxima
    if abs(even_bins_y[0].max()) > abs(even_bins_y[0].min()):
        hi_peaks = [bin.max() for bin in even_bins_y]
        lo_peaks = [bin.min() for bin in odd_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y == peak)[0][0]])
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y == peak)[0][0]])
    else:
        hi_peaks = [bin.max() for bin in odd_bins_y]
        lo_peaks = [bin.min() for bin in even_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y == peak)[0][0]])
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y == peak)[0][0]])

    # peaks or valley cannot be at 0
    max_peaks = [[x, y] for x, y in zip(hi_peaks_x, hi_peaks) if x != 0]
    min_peaks = [[x, y] for x, y in zip(lo_peaks_x, lo_peaks) if x != 0]

    return [max_peaks, min_peaks]


def _smooth(x, window_len=11, window='hanning'):
    """Smooth the data using a window of the requested size.

    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end parts of the output signal.

    Parameters
    ----------
    x:
        The input signal.
    window_len:
        The dimension of the smoothing window; should be an odd
        integer.
    window:
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman'.  The 'flat' window will produce a moving average smoothing.

    Returns
    -------
    out:
        The smoothed signal.

    Examples
    --------
        >>> t = np.linspace(-2, 2, 20)
        >>> x = np.sin(t)+np.random.randn(len(t))*0.1
        >>> y = _smooth(x)

    See Also
    --------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter

    """
    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(old_div(w, w.sum()), s, mode='valid')
    return y


def zero_crossings(y_axis, window=11):
    """Zero crossing peak detect.

    Algorithm to find zero crossings. Smooths the curve and finds the
    zero-crossings by looking for a sign change.

    keyword arguments:
    y_axis -- A list containing the signal over which to find zero-crossings
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)

    return -- the index for each zero-crossing

    """
    # smooth the curve
    length = len(y_axis)
    x_axis = np.asarray(list(range(length)), int)

    ymean = y_axis.mean()
    y_axis = y_axis - ymean

    # discard tail of smoothed signal
    y_axis = _smooth(y_axis, window)[:length]
    zero_crossings = np.where(np.diff(np.sign(y_axis)))[0]
    indices = [x_axis[index] for index in zero_crossings]

    # check if zero-crossings are valid
    # diff = np.diff(indices)
#    if diff.std() / diff.mean() > 0.2:
#        print diff.std() / diff.mean()
#        print np.diff(indices)
#        raise(ValueError,
#            "False zero-crossings found, indicates problem {0} or {1}".format(
#            "with smoothing window", "problem with offset"))
    # check if any zero crossings were found
    if len(zero_crossings) < 1:
        raise ValueError

    try:
        indices.remove(0)
    except ValueError:
        pass

    return indices
    # used this to test the fft function's sensitivity to spectral leakage
    # return indices + np.asarray(30 * np.random.randn(len(indices)), int)

# Frequency calculation#############################
#    diff = np.diff(indices)
#    time_p_period = diff.mean()
#
#    if diff.std() / time_p_period > 0.1:
#        raise ValueError,
#            "smoothing window too small, false zero-crossing found"
#
#    #return frequency
#    return 1.0 / time_p_period
##############################################################################


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def peak_detection(input_ts='-',
                   columns=None,
                   start_date=None,
                   end_date=None,
                   dropna='no',
                   skiprows=None,
                   index_type='datetime',
                   names=None,
                   clean=False,
                   method='rel',
                   extrema='peak',
                   window=24,
                   pad_len=5,
                   points=9,
                   lock_frequency=False,
                   float_format='%g',
                   round_index=None,
                   source_units=None,
                   target_units=None,
                   print_input=''):
    r"""Peak and valley detection.

    Parameters
    ----------
    extrema : str
        [optional, default is 'peak']

        'peak', 'valley', or 'both' to determine what should be
        returned.
    method : str
        [optional, default is 'rel']
        'rel', 'minmax', 'zero_crossing', 'parabola', 'sine' methods are
        available.  The different algorithms have different strengths
        and weaknesses.
    window : int
        [optional, default is 24]

        There will not usually be multiple peaks within the window
        number of values.  The different methods use this variable in
        different ways.  For 'rel' the window keyword specifies how many
        points on each side to require a comparator(n,n+x) = True.  For
        'minmax' the window keyword is the distance to look ahead from
        a peak candidate to determine if it is the actual peak.

        '(sample / period) / f'

        where f might be a good choice between 1.25 and 4.

        For 'zero_crossing' the window keyword is the dimension of the
        smoothing window and should be an odd integer.
    pad_len : int
        [optional, default is 5]

        Used with FFT to pad edges of time-series.
    points : int
        [optional, default is 9]

        For 'parabola' and 'sine' methods. How many points around the
        peak should be used during curve fitting, must be odd.  The
    lock_frequency
        [optional, default is False]

        For 'sine' method only.  Specifies if the frequency argument of
        the model function should be locked to the value calculated from
        the raw peaks or if optimization process may tinker with it.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {float_format}
    {round_index}
    {source_units}
    {target_units}
    {print_input}

    """
    # Couldn't get fft method working correctly.  Left pieces in
    # in case want to figure it out in the future.

    assert extrema in ['peak', 'valley', 'both'], """
*
*   The `extrema` argument must be one of 'peak',
*   'valley', or 'both'.  You supplied {0}.
*
""".format(extrema)

    assert method in ['rel',
                      'minmax',
                      'zero_crossing',
                      'parabola',
                      'sine'], """
*
*   The `method` argument must be one of 'rel', 'minmax',
*   'zero_crossing', 'parabola', or 'sine'.  You supplied {0}.
*
""".format(method)

    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna,
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    window = int(window)
    kwds = {}
    if method == 'rel':
        func = _argrel
        window = window / 2
        if window == 0:
            window = 1
        kwds['window'] = int(window)
    elif method == 'minmax':
        func = _peakdetect
        window = int(window / 2)
        if window == 0:
            window = 1
        kwds['window'] = int(window)
    elif method == 'zero_crossing':
        func = _peakdetect_zero_crossing
        if not window % 2:
            window = window + 1
        kwds['window'] = int(window)
    elif method == 'parabola':
        func = _peakdetect_parabola
        kwds['points'] = int(points)
    elif method == 'sine':
        func = _peakdetect_sine
        kwds['points'] = int(points)
        kwds['lock_frequency'] = lock_frequency
    elif method == 'fft':  # currently would never be used.
        func = _peakdetect_fft
        kwds['pad_len'] = int(pad_len)

    if extrema == 'peak':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_peak', copy=True)
    if extrema == 'valley':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_valley', copy=True)
    if extrema == 'both':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_peak', copy=True)
        tmptsd = tmptsd.join(
            tsd.rename(columns=lambda x: str(x) + '_valley', copy=True),
            how='outer')

    for cols in tmptsd.columns:
        if method in ['fft', 'parabola', 'sine']:
            maxpeak, minpeak = func(
                tmptsd[cols].values, list(range(len(tmptsd[cols]))), **kwds)
        else:
            maxpeak, minpeak = func(tmptsd[cols].values, **kwds)
        if cols[-5:] == '_peak':
            datavals = maxpeak
        if cols[-7:] == '_valley':
            datavals = minpeak
        maxx, _ = list(zip(*datavals))
        hold = tmptsd[cols][pd.np.array(maxx).astype('i')]
        tmptsd[cols][:] = pd.np.nan
        tmptsd[cols][pd.np.array(maxx).astype('i')] = hold

    tmptsd.index.name = 'Datetime'
    tsd.index.name = 'Datetime'
    return tsutils.print_input(print_input, tsd, tmptsd, None,
                               float_format=float_format)
