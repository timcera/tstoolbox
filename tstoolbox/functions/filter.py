#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
import numpy as np
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd
from scipy import signal

from .. import tsutils

warnings.filterwarnings("ignore")


class MisMatchedKernel(Exception):
    """Error class for the wrong length kernel."""

    def __init__(self, rk, pw):
        """Initialize with lengths of kernel and requested length."""
        self.rk = rk
        self.pw = pw

    def __str__(self):
        """Return detailed error message."""
        return """
*
*   Length of kernel must be {0}.
*   Instead have {1}
*
""".format(
            self.rk, self.pw
        )


class BadKernelValues(Exception):
    """Error class for the negative pad width."""

    def __str__(self):
        """Return detailed error message."""
        return """
*
*   Should only have positive values.
*
"""


def _transform(vector, filter_pass, lowpass_cutoff, highpass_cutoff, window_len):
    """Private function used by FFT filtering.

    Parameters
    ----------
    vector: array_like, evenly spaced samples in time

    Returns
    -------
    vector of filtered values

    """
    import numpy.fft as F

    result = F.rfft(vector, len(vector))

    freq = F.fftfreq(len(vector))[: len(vector) // 2 + 1]
    factor = np.ones_like(freq)

    if filter_pass in ["lowpass", "bandpass"]:
        factor[freq > 1.0 / float(lowpass_cutoff)] = 0.0
    if filter_pass in ["highpass", "bandpass"]:
        factor[freq < 1.0 / float(highpass_cutoff)] = 0.0
    if filter_pass == "bandstop":
        factor[
            freq < 1.0 / float(lowpass_cutoff) and freq > 1.0 / float(highpass_cutoff)
        ] = 0.0

    factor = np.pad(factor, window_len + 1, mode="edge",)

    factor = np.convolve(factor, [1.0 / window_len] * window_len, mode=1)
    factor = factor[window_len + 1 : -(window_len + 1)]

    result = result * factor

    rvector = F.irfft(result, len(vector))

    return np.atleast_1d(rvector)


@mando.command("filter", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def filter_cli(
    filter_type,
    filter_pass,
    butterworth_order=1,
    reverse_second_stage=True,
    lowpass_cutoff=None,
    highpass_cutoff=None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    print_input=False,
    cutoff_period=None,
    window_len=3,
    float_format="g",
    source_units=None,
    target_units=None,
    round_index=None,
    tablefmt="csv",
):
    """Apply different filters to the time-series.

    Parameters
    ----------
    filter_type: str
        One of 'fft', or 'butterworth' filter types.
        
        The 'fft' and 'butterworth' types are defined by cutoff frequencies.
    filter_pass: str
        One of 'lowpass', 'highpass', 'bandpass', or 'bandstop'.  Indicates which what frequencies to 
        block.
    cutoff_period
        [optional, default is None]

        DEPRECATED: Use `lowpass_cutoff` if `filter_pass` equals "lowpass" and `highpass_cutoff` if `filter_pass` equals "highpass".
    lowpass_cutoff: float
        [optional, default is None, used only if `filter_pass` equals "lowpass", "bandpass" or "bandstop"]
        
        The low frequency cutoff when `filter_pass` equals "vertical", "bandpass", or "bandstop".
    highpass_cutoff: float
        [optional, default is None, used only if `filter_pass` equals "highpass", "bandpass" or "bandstop"]
        
        The high frequency cutoff when `filter_pass` equals "highpass", "bandpass", or "bandstop".
    window_len: int
        [optional, default is 3]
        
        Will soften the edges of the "fft" filter in the frequency domain.  The larger the number the softer the filter edges.  A value of 1 will have a brick wall step function which may introduce frequencies into the filtered output.
    butterworth_order: int
        [optional, default is 1]
        
        The order of the butterworth filter.
    reverse_second_stage: bool
        [optional, default is True]
        
        Will perform a second filter in reverse to eliminate shifting in time caused by the first filter.
    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {float_format}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {round_index}
    {source_units}
    {target_units}
    {print_input}
    {tablefmt}

    """
    tsutils.printiso(
        filter(
            filter_type,
            filter_pass,
            butterworth_order=butterworth_order,
            reverse_second_stage=reverse_second_stage,
            lowpass_cutoff=lowpass_cutoff,
            highpass_cutoff=highpass_cutoff,
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            print_input=print_input,
            cutoff_period=cutoff_period,
            window_len=window_len,
            source_units=source_units,
            target_units=target_units,
            round_index=round_index,
        ),
        float_format=float_format,
        tablefmt=tablefmt,
    )


@tsutils.validator(
    filter_type=[
        str,
        [
            "domain",
            [
                "flat",
                "hanning",
                "hamming",
                "bartlett",
                "blackman",
                "fft",
                "butterworth",
            ],
        ],
        1,
    ],
    filter_pass=[str, ["domain", ["lowpass", "highpass", "bandpass", "bandstop"],], 1,],
    window_len=[int, ["range", [1, None]], 1],
    cutoff_period=[float, ["range", [0, None]], 1],
    lowpass_cutoff=[float, ["range", [0, None]], 1],
    highpass_cutoff=[float, ["range", [0, None]], 1],
    butterworth_order=[int, ["range", [1, None]], 1],
    reverse_second_stage=[bool, ["domain", [True, False],], 1,],
)
def filter(
    filter_type,
    filter_pass,
    butterworth_order=1,
    reverse_second_stage=True,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    print_input=False,
    lowpass_cutoff=None,
    highpass_cutoff=None,
    cutoff_period=None,
    window_len=3,
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Apply different filters to the time-series."""
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    if len(tsd.values) < window_len:
        raise ValueError(
            tsutils.error_wrapper(
                """
Input vector (length={0}) needs to be bigger than window size ({1}).
""".format(
                    len(tsd.values), window_len
                )
            )
        )

    if filter_type in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        warnings.warn(
            tsutils.error_wrapper(
                """
DEPRECATED: The `filter_type`s "flat", "hanning", "hamming", "bartlett", and "blackman" are implemented with greater capabilities in the "rolling_window" function in tstoolbox.  Eventually they will be removed from the "filter" function."""
            )
        )
        # Inelegant - but works.  If any rolling_window filters then just set
        # lowpass_cutoff and highpass_cutoff to anything since it isn't used.
        if filter_pass == "lowpass":
            lowpass_cutoff = 1
            highpass_cutoff = None
        elif filter_pass == "highpass":
            lowpass_cutoff = None
            highpass_cutoff = 1
        else:
            lowpass_cutoff = 1
            highpass_cutoff = 1

    if cutoff_period is None:
        warnings.warn(
            tsutils.error_wrapper(
                """
The `cutoff_period` is deprecated in favor of using `lowpass_cutoff` if `filter_pass` is "lowpass", "bandpass" or "bandstop" and `highpass_cutoff` if `filter_pass` is "highpass", "bandpass", or "bandstop".  The `lowpass_cutoff` or `highpass_cutoff` options are set to `cutoff_period` according to `filter_pass`."""
            )
        )
        if filter_pass == "lowpass" and lowpass_cutoff is None:
            lowpass_cutoff = cutoff_period
        if filter_pass == "highpass" and highpass_cutoff is None:
            highpass_cutoff = cutoff_period
    if filter_pass in ["bandpass", "bandstop"]:
        # Need both low_cutoff and highpass_cutoff for "bandpass" and "bandstop".
        if lowpass_cutoff is None or highpass_cutoff is None:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The "bandpass" and "bandstop" options for `filter_pass` require values for the `lowpass_cutoff` and `highpass_cutoff` keywords.  You have "{lowpass_cutoff}" for `lowpass_cutoff` and "{highpass_cutoff}" for `highpass_cutoff`.""".format(
                        **locals()
                    )
                )
            )

    if filter_pass == "lowpass":
        if lowpass_cutoff is None:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The "lowpass" option for `filter_pass` requires a value for `lowpass_cutoff`.  You have "{lowpass_cutoff}".""".format(
                        **locals()
                    )
                )
            )
        if highpass_cutoff is not None:
            warnings.warn(
                tsutils.error_wrapper(
                    """
The `highpass_cutoff` value of {highpass_cutoff} is ignored it `filter_pass` is "lowpass".""".format(
                        **locals()
                    )
                )
            )

    if filter_pass == "highpass":
        if highpass_cutoff is None:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The "highpass" option for `filter_pass` requires a value for `highpass_cutoff`.  You have "{highpass_cutoff}".""".format(
                        **locals()
                    )
                )
            )
        if lowpass_cutoff is not None:
            warnings.warn(
                tsutils.error_wrapper(
                    """
The `lowpass_cutoff` value of {lowpass_cutoff} is ignored it `filter_pass` is "highpass".""".format(
                        **locals()
                    )
                )
            )

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    for col in tsd.columns:
        if filter_type == "fft":
            tsd[col].values[:] = _transform(
                tsd[col].values,
                filter_pass,
                lowpass_cutoff,
                highpass_cutoff,
                window_len,
            )
        elif filter_type == "butterworth":
            if filter_pass == "lowpass":
                wn = lowpass_cutoff
            elif filter_pass == "highpass":
                wn = highpass_cutoff
            else:
                wn = [lowpass_cutoff, highpass_cutoff]
            if reverse_second_stage is True:
                b, a = signal.butter(butterworth_order, wn, btype=filter_pass)
                tsd[col].values[:] = signal.filtfilt(b, a, tsd[col].values)
            else:
                sos = signal.butter(
                    butterworth_order, wn, btype=filter_pass, output="sos"
                )
                tsd[col].values[:] = signal.sosfilt(sos, tsd[col].values[:])
        elif filter_type in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            if window_len < 3:
                continue
            s = np.pad(tsd[col].values, window_len // 2, "reflect")

            if filter_type == "flat":  # moving average
                w = np.ones(window_len, "d")
            else:
                w = eval("np." + filter_type + "(window_len)")
            tsd[col].values[:] = np.convolve(w / w.sum(), s, mode="valid")

    return tsutils.return_input(print_input, otsd, tsd, "filter")


filter.__doc__ = filter_cli.__doc__
