#!/usr/bin/env python

from __future__ import print_function

"""
The filter.py module contains a group of functions to filter
time-series values.
"""

#===imports======================
import numpy as np

#===globals======================
modname = "filter"

#---other---
__all__ = [
    'fft_lowpass',
    ]


########
# Exception classes


class MisMatchedKernel(Exception):
    '''
    Error class for the wrong length kernel.
    '''
    def __init__(self, rk, pw):
        self.rk = rk
        self.pw = pw

    def __str__(self):
        return """
Length of kernel must be %i.
Instead have %i""" % (self.rk, self.pw)


class BadKernelValues(Exception):
    '''
    Error class for the negative pad width.
    '''
    def __init__(self):
        pass

    def __str__(self):
        return "\n\nShould only have positive values."


########


########
# Private utility functions.


def _transform(vector, ramp_start_freq, ramp_end_freq):
    """

    Parameters
    ----------
    vector : array_like, evenly spaced samples in time
    ramp_start_freq : frequency below which the freq domain is 0
    ramp_end_freq : frequency above which the freq domain is unchanged

    Returns
    -------
    vector of filtered values

    See Also
    --------

    Examples
    --------

    """
    import numpy.fft as F
    result = F.rfft(vector)

    freq = F.fftfreq(len(vector))[:len(vector)/2]
    factor = np.ones_like(result)
    factor[freq > 1.0/ramp_start_freq] = 0.0

    sl = np.logical_and(1.0/ramp_start_freq < freq, freq < 1.0/ramp_end_freq)

    a = factor[sl]
    # Create float array of required length and reverse
    a = np.arange(len(a) + 2).astype(float)[::-1]

    # Ramp from 1 to 0 exclusive
    a = (a/a[0])[1:-1]

    # Insert ramp into factor
    factor[sl] = a

    result = result * factor

    relevation = F.irfft(result)
    return relevation


########
# Public functions


def fft_lowpass(vector, ramp_start_freq, ramp_end_freq):
    """

    Parameters
    ----------
    vector : array_like, evenly spaced samples in time
    ramp_start_freq : frequency below which the freq domain is 0
    ramp_end_freq : frequency above which the freq domain is unchanged

    Returns
    -------
    vector of filtered values

    See Also
    --------

    Examples
    --------

    """
    return _transform(vector, ramp_start_freq, ramp_end_freq)


def fft_highpass(vector, ramp_start_freq, ramp_end_freq):
    """

    Parameters
    ----------
    vector : array_like, evenly spaced samples in time
    ramp_start_freq : frequency below which the freq domain is 0
    ramp_end_freq : frequency above which the freq domain is unchanged

    Returns
    -------
    vector of filtered values

    See Also
    --------

    Examples
    --------

    """
    return _transform(vector, ramp_end_freq, ramp_start_freq)


########


if __name__ == '__main__':
    ''' This section is just used for testing.  Really you should only import
        this module.
    '''
    arr = np.arange(100)
    print(arr)
    print(np.median(arr, (3, )))
    print(np.constant(arr, (-25, 20), (10, 20)))
    arr = np.arange(30)
    arr = np.reshape(arr, (6, 5))
    print(np.mean(arr, pad_width=((2, 3), (3, 2), (4, 5)), stat_len=(3, )))
