# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
import numpy as np
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


warnings.filterwarnings("ignore")


def lpdes(fc, t, ns):
    """Subroutine LPDES evaluates the coefficients for a low pass filter."""
    a = np.zeros(ns)
    b = np.zeros(ns)
    c = np.zeros(ns)
    wcp = np.sin(fc * np.pi * t) / np.cos(fc * np.pi * t)
    for k in range(ns):
        cs = np.cos(float(2 * (k + ns) - 1) * np.pi / float(4 * ns))
        x = 1.0 / (1.0 + wcp * wcp - 2.0 * wcp * cs)
        a[k] = wcp * wcp * x
        b[k] = 2.0 * (wcp * wcp - 1.0) * x
        c[k] = (1.0 + wcp * wcp + 2.0 * wcp * cs) * x
    return a, b, c


def hpdes(fc, t, ns):
    """Subroutine HPDES evaluates the coefficients for a high pass filter."""
    a = np.zeros(ns)
    b = np.zeros(ns)
    c = np.zeros(ns)
    pi = 3.1415926536
    wcp = np.sin(fc * pi * t) / np.cos(fc * pi * t)
    for k in range(ns):
        cs = np.cos(float(2 * (k + ns) - 1) * pi / float(4 * ns))
        a[k] = 1.0 / (1.0 + wcp * wcp - 2.0 * wcp * cs)
        b[k] = 2.0 * (wcp * wcp - 1.0) * a[k]
        c[k] = (1.0 + wcp * wcp + 2.0 * wcp * cs) * a[k]
    return a, b, c


def bpdes(f1, f2, t, ns):
    """Subroutine BPDES evaluates the coefficients for a band pass filter."""
    a = np.zeros(ns)
    b = np.zeros(ns)
    c = np.zeros(ns)
    d = np.zeros(ns)
    e = np.zeros(ns)
    pi = 3.1415926536
    w1 = np.sin(f1 * pi * t) / np.cos(f1 * pi * t)
    w2 = np.sin(f2 * pi * t) / np.cos(f2 * pi * t)
    wc = w2 - w1
    q = wc * wc + 2.0 * w1 * w2
    s = w1 * w1 * w2 * w2
    for k in range(ns):
        cs = np.cos(float(2 * (k + ns) - 1) * pi / float(4 * ns))
        p = -2.0 * wc * cs
        r = p * w1 * w2
        x = 1.0 + p + q + r + s
        a[k] = wc * wc / x
        b[k] = (-4.0 - 2.0 * p + 2.0 * r + 4.0 * s) / x
        c[k] = (6.0 - 2.0 * q + 6.0 * s) / x
        d[k] = (-4.0 + 2.0 * p - 2.0 * r + 4.0 * s) / x
        e[k] = (1.0 - p + q - r + s) / x
        return a, b, c, d, e


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

    factor = np.pad(
        factor,
        window_len + 1,
        mode="edge",
    )

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
    lowpass_cutoff=None,
    highpass_cutoff=None,
    window_len=3,
    butterworth_stages=1,
    reverse_second_stage=True,
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    float_format="g",
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    round_index=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
):
    """Apply different filters to the time-series.

    Parameters
    ----------
    filter_type: str
        OneOf("fft", "butterworth")
        The "fft" and "butterworth" types are defined by cutoff frequencies.
        The "fft" is the Fast Fourier Transform filter in the frequency domain.
    filter_pass: str
        OneOf("lowpass", "highpass", "bandpass", "bandstop")
        Indicates what frequencies to block.
    lowpass_cutoff: float
        [optional, default is None, used only if `filter_pass` equals
         "lowpass", "bandpass" or "bandstop"]

        The low frequency cutoff when `filter_pass` equals "vertical",
        "bandpass", or "bandstop".
    highpass_cutoff: float
        [optional, default is None, used only if `filter_pass` equals
         "highpass", "bandpass" or "bandstop"]

        The high frequency cutoff when `filter_pass` equals "highpass",
        "bandpass", or "bandstop".
    window_len: int
        [optional, default is 3]

        Will soften the edges of the "fft" filter in the frequency domain.
        The larger the number the softer the filter edges.  A value of 1
        will have a brick wall step function which may introduce
        frequencies into the filtered output.
    butterworth_stages: int
        [optional, default is 1]

        The order of the butterworth filter.
    reverse_second_stage: bool
        [optional, default is True]

        Will perform a second filter in reverse to eliminate shifting
        in time caused by the first filter.
    ${input_ts}
    ${start_date}
    ${end_date}
    ${columns}
    ${float_format}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${round_index}
    ${source_units}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    tsutils.printiso(
        filter(
            filter_type,
            filter_pass,
            butterworth_stages=butterworth_stages,
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
            window_len=window_len,
            source_units=source_units,
            target_units=target_units,
            round_index=round_index,
        ),
        float_format=float_format,
        tablefmt=tablefmt,
    )


@typic.constrained(ge=1)
class IntGreaterThanOrEqualToOne(int):
    """Integer greater than zero."""


@typic.constrained(ge=1, le=3)
class IntBetweenOneAndThree(int):
    """Integer between 1 and 3 inclusive."""


@typic.constrained(ge=0)
class FloatGreaterThanOrEqualToZero(float):
    """Positive float value."""


@typic.al
@tsutils.copy_doc(filter_cli)
def filter(
    filter_type: Literal[
        "flat",
        "hanning",
        "hamming",
        "bartlett",
        "blackman",
        "fft",
        "butterworth",
    ],
    filter_pass: Literal["lowpass", "highpass", "bandpass", "bandstop"],
    butterworth_stages: IntBetweenOneAndThree = 1,
    butterworth_reverse_second_stage: bool = True,
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
    lowpass_cutoff: FloatGreaterThanOrEqualToZero = None,
    highpass_cutoff: FloatGreaterThanOrEqualToZero = None,
    window_len: IntGreaterThanOrEqualToOne = 3,
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Apply different filters to the time-series."""
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
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
Input vector (length={}) needs to be bigger than window size ({}).
""".format(
                    len(tsd.values), window_len
                )
            )
        )

    if filter_type in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        warnings.warn(
            tsutils.error_wrapper(
                """
DEPRECATED: The `filter_type`s "flat", "hanning", "hamming", "bartlett", and
"blackman" are implemented with greater capabilities in the "rolling_window"
function in tstoolbox.  Eventually they will be removed from the "filter" function."""
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

    if filter_pass in ["bandpass", "bandstop"]:
        # Need both low_cutoff and highpass_cutoff for "bandpass" and "bandstop".
        if lowpass_cutoff is None or highpass_cutoff is None:
            raise ValueError(
                tsutils.error_wrapper(
                    f"""
The "bandpass" and "bandstop" options for `filter_pass` require values
for the `lowpass_cutoff` and `highpass_cutoff` keywords.  You have
"{lowpass_cutoff}" for `lowpass_cutoff` and "{highpass_cutoff}" for
`highpass_cutoff`."""
                )
            )

    if filter_pass == "lowpass":
        if lowpass_cutoff is None:
            raise ValueError(
                tsutils.error_wrapper(
                    f"""
The "lowpass" option for `filter_pass` requires a value for
`lowpass_cutoff`.  You have "{lowpass_cutoff}"."""
                )
            )
        if highpass_cutoff is not None:
            warnings.warn(
                tsutils.error_wrapper(
                    f"""
The `highpass_cutoff` value of {highpass_cutoff} is ignored it
`filter_pass` is "lowpass"."""
                )
            )

    if filter_pass == "highpass":
        if highpass_cutoff is None:
            raise ValueError(
                tsutils.error_wrapper(
                    f"""
The "highpass" option for `filter_pass` requires a value for
`highpass_cutoff`.  You have "{highpass_cutoff}"."""
                )
            )
        if lowpass_cutoff is not None:
            warnings.warn(
                tsutils.error_wrapper(
                    f"""
The `lowpass_cutoff` value of {lowpass_cutoff} is ignored it
`filter_pass` is "highpass"."""
                )
            )

    if print_input is True:
        ntsd = tsutils.asbestfreq(tsd.copy())
    else:
        ntsd = tsutils.asbestfreq(tsd)

    tdelt = pd.to_timedelta(ntsd.index.freq).delta / 86400000000000

    for col in tsd.columns:
        if filter_type == "fft":
            ntsd[col] = _transform(
                tsd[col],
                filter_pass,
                lowpass_cutoff,
                highpass_cutoff,
                window_len,
            )
        elif filter_type == "butterworth":
            rval = np.zeros(9)
            if filter_pass == "lowpass":
                if lowpass_cutoff >= 0.5 / tdelt:
                    raise ValueError(
                        tsutils.error_wrapper(
                            """
The "lowpass_cutoff" must be greater than 0.5/interval_in_days."""
                        )
                    )
                a, b, c = lpdes(lowpass_cutoff, tdelt, butterworth_stages)
            elif filter_pass == "highpass":
                if highpass_cutoff >= 0.5 / tdelt:
                    raise ValueError(
                        tsutils.error_wrapper(
                            """
The "highpass_cutoff" must be greater than 0.5/interval_in_days."""
                        )
                    )
                a, b, c = hpdes(highpass_cutoff, tdelt, butterworth_stages)
            elif filter_pass == "bandpass":
                if lowpass_cutoff >= 0.5 / tdelt or highpass_cutoff >= 0.5 / tdelt:
                    raise ValueError(
                        tsutils.error_wrapper(
                            """
The "lowpass_cutoff" and "highpass_cutoff" must be greater than 0.5/interval_in_days."""
                        )
                    )
                a, b, c, d, e = bpdes(
                    lowpass_cutoff, highpass_cutoff, tdelt, butterworth_stages
                )

            rval = np.pad(tsd[col].values, (4, 0), mode="edge")
            for k in range(butterworth_stages):
                af = a[k]
                bf = b[k]
                cf = c[k]
                if (k == 1) and (butterworth_reverse_second_stage is True):
                    af = a[0]
                    bf = b[0]
                    cf = c[0]
                if filter_pass == "bandpass":
                    df = d[k]
                    ef = e[k]
                if filter_pass == "lowpass":
                    gval = rval
                else:
                    gval = rval
                    gval[:4] = 0.0
                if filter_pass == "lowpass":
                    gval[4:] = (
                        af * (rval[4:] + 2.0 * rval[3:-1] + rval[2:-2])
                        - bf * gval[3:-1]
                        - cf * gval[2:-2]
                    )
                elif filter_pass == "highpass":
                    gval[4:] = (
                        af * (rval[4:] - 2.0 * rval[3:-1] + rval[2:-2])
                        - bf * gval[3:-1]
                        - cf * gval[2:-2]
                    )
                elif filter_pass == "bandpass":
                    gval[4:] = (
                        af * (rval[4:] - 2.0 * rval[2:-2] + rval[:-4])
                        - bf * gval[3:-1]
                        - cf * gval[2:-2]
                        - df * gval[1:-3]
                        - ef * gval[:-4]
                    )
                gval = gval[4:]
                if k + 1 != butterworth_stages:
                    if k == 0 and butterworth_reverse_second_stage is True:
                        rval = np.flip(gval)
                    else:
                        rval = gval
                elif butterworth_reverse_second_stage is True:
                    gval = np.flip(gval)
            ntsd[col] = gval

        elif filter_type in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            if window_len < 3:
                continue
            s = np.pad(tsd[col].values, window_len // 2, "reflect")

            if filter_type == "flat":  # moving average
                w = np.ones(window_len, "d")
            else:
                w = eval("np." + filter_type + "(window_len)")
            ntsd[col] = np.convolve(w / w.sum(), s, mode="valid")

    return tsutils.return_input(print_input, tsd, ntsd, "filter")


if __name__ == "__main__":
    from tstoolbox import tstoolbox

    df = tstoolbox.read("../../tests/02325000_flow.csv")
    filt_fft_high = filter(
        "fft", "highpass", print_input=True, input_ts=df, highpass_cutoff=10
    )
    filt_fft_low = filter(
        "fft", "lowpass", print_input=True, input_ts=df, lowpass_cutoff=10
    )
    filt_butter_high = filter(
        "butterworth", "highpass", print_input=True, input_ts=df, highpass_cutoff=0.4
    )
    filt_butter_low = filter(
        "butterworth", "lowpass", print_input=True, input_ts=df, lowpass_cutoff=0.4
    )
