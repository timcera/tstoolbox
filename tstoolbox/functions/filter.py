#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils

warnings.filterwarnings('ignore')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def filter(filter_type,
           input_ts='-',
           columns=None,
           start_date=None,
           end_date=None,
           dropna='no',
           print_input=False,
           cutoff_period=None,
           window_len=5,
           float_format='%g',
           round_index=None):
    """Apply different filters to the time-series.

    Parameters
    ----------
    filter_type : str
        'flat', 'hanning', 'hamming', 'bartlett', 'blackman',
        'fft_highpass' and 'fft_lowpass' for Fast Fourier Transform
        filter in the frequency domain.
    window_len : int
        [optional, default is 5]

        For the windowed types, 'flat', 'hanning', 'hamming',
        'bartlett', and 'blackman' specifies the length of the window.
    cutoff_period
        [optional, default is None]

        For 'fft_highpass' and 'fft_lowpass'.  Must be supplied if using
        'fft_highpass' or 'fft_lowpass'.  The period in input time units that
        will form the cutoff between low frequencies (longer periods) and high
        frequencies (shorter periods).  Filter will be smoothed by `window_len`
        running average.
    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {float_format}
    {dropna}
    {round_index}
    {print_input}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    assert len(tsd.values) > window_len, """
*
*   Input vector (length={0}) needs to be bigger than window size ({1}).
*
""".format(len(tsd.values), window_len)

    assert filter_type in ['flat',
                           'hanning',
                           'hamming',
                           'bartlett',
                           'blackman',
                           'fft_highpass',
                           'fft_lowpass'], """
*
*   Filter type {0} not implemented.
*
""".format(filter_type)

    from tstoolbox import filters

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    for col in tsd.columns:
        # fft_lowpass, fft_highpass
        if filter_type == 'fft_lowpass':
            tsd[col].values[:] = filters._transform(tsd[col].values,
                                                    cutoff_period,
                                                    window_len,
                                                    lopass=True)
        elif filter_type == 'fft_highpass':
            tsd[col].values[:] = filters._transform(tsd[col].values,
                                                    cutoff_period,
                                                    window_len)
        elif filter_type in ['flat',
                             'hanning',
                             'hamming',
                             'bartlett',
                             'blackman']:
            if window_len < 3:
                continue
            s = pd.np.pad(tsd[col].values, window_len // 2, 'reflect')

            if filter_type == 'flat':  # moving average
                w = pd.np.ones(window_len, 'd')
            else:
                w = eval('pd.np.' + filter_type + '(window_len)')
            tsd[col].values[:] = pd.np.convolve(w / w.sum(), s, mode='valid')

    return tsutils.print_input(print_input, otsd, tsd, '_filter',
                               float_format=float_format)
