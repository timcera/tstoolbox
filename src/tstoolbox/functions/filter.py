"""Collection of functions for the manipulation of time series."""

import warnings
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt
from scipy import signal
from scipy.fft import irfft, rfft, rfftfreq
from scipy.optimize import leastsq
from typing_extensions import Annotated

from ..toolbox_utils.src.toolbox_utils import tsutils

try:
    from pydantic import validate_arguments
except ImportError:
    from pydantic import validate_call as validate_arguments

warnings.filterwarnings("ignore")


def _transform(vector, filter_pass, lowpass_cutoff, highpass_cutoff, window_len):
    """Private function used by FFT filtering.

    Parameters
    ----------
    vector : array_like, evenly spaced samples in time

    Returns
    -------
    vector of filtered values

    """
    vector = np.array(vector.ffill().bfill())
    result = rfft(vector)

    freq = np.abs(rfftfreq(len(vector)))
    factor = np.ones_like(freq)

    if filter_pass in ("lowpass", "bandpass"):
        factor[freq > float(lowpass_cutoff)] = 0.0
    if filter_pass in ("highpass", "bandpass"):
        factor[freq < (float(highpass_cutoff))] = 0.0
    if filter_pass == "bandstop":
        factor[(float(highpass_cutoff)) < freq < (float(lowpass_cutoff))] = 0.0

    factor = np.convolve(factor, [1.0 / window_len] * window_len, mode="same")

    result = result * factor

    rvector = irfft(result)

    return np.atleast_1d(rvector)


def delta_diff(elev, delta, start_index):
    """Calculate the difference between the elevation and the elevation delta hours ago."""
    bindex = delta
    if start_index > delta:
        bindex = start_index
    tmpe = elev[bindex:]
    return tmpe - elev[bindex - delta : bindex - delta + len(tmpe)]


def fft_lowpass(nelevation, low_bound, high_bound):
    """Performs a low pass filter on the nelevation series.
    low_bound and high_bound specifes the boundary of the filter.
    """
    nelevation = np.array(nelevation.ffill().bfill())
    if len(nelevation) % 2:
        result = rfft(nelevation, len(nelevation))
    else:
        result = rfft(nelevation)
    freq = rfftfreq(len(nelevation))[: len(nelevation) // 2 + 1]
    factor = np.ones_like(result)
    factor[freq > low_bound] = 0.0

    sl = np.logical_and(high_bound < freq, freq < low_bound)

    a = factor[sl]
    # Create float array of required length and reverse
    a = np.arange(len(a) + 2).astype(float)[::-1]

    # Ramp from 1 to 0 exclusive
    a = (a / a[0])[1:-1]

    # Insert ramp into factor
    factor[sl] = a

    result = result * factor
    relevation = irfft(result, len(nelevation))
    return relevation


FILTERS = (
    "bartlett",
    "blackman",
    "butterworth",
    # "cd",
    "fft",
    "flat",
    "hamming",
    "hanning",
    "kalman",
    "lecolazet1",
    "lecolazet2",
    # "tide_mstha",
    "tide_doodson",
    "tide_fft",
    "tide_usgs",
    # "wavelet",
)
Filters = Literal[FILTERS]

tsutils.docstrings["FILTERS"] = ", ".join(FILTERS)


@tsutils.transform_args(filter_types=tsutils.make_list)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def filter(
    filter_types: Union[Filters, List[Filters]],
    filter_pass: Literal["lowpass", "highpass", "bandpass", "bandstop"],
    butterworth_order: Annotated[int, Field(ge=1)] = 10,
    lowpass_cutoff: Optional[PositiveFloat] = None,
    highpass_cutoff: Optional[PositiveFloat] = None,
    window_len: PositiveInt = 3,
    pad_mode: Optional[
        Literal[
            "edge",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
        ]
    ] = "reflect",
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    print_input=False,
    source_units=None,
    target_units=None,
    round_index=None,
    float_format="g",
    tablefmt="csv",
):
    """
    Apply different filters to the time-series.

    Parameters
    ----------
    filter_types
        One or more of
        ${FILTERS}

        The "fft" and "butterworth" types are configured by cutoff frequencies
        `lowpass_cutoff`, and `highpass_cutoff`, by process defined in
        `filter_pass`. The "fft" is the Fast Fourier Transform filter in the
        frequency domain.

        Doodson filter

        The Doodson X0 filter is a simple filter designed to damp out
        the main tidal frequencies. It takes hourly values, 19 values
        either side of the central one. A weighted average is taken
        with the following weights

        (1010010110201102112 0 2112011020110100101)/30.

        In "Data Analysis and Methods in Oceanography":

        "The cosine-Lanczos filter, the transform filter, and the
        Butterworth filter are often preferred to the Godin filter,
        to earlier Doodson filter, because of their superior ability
        to remove tidal period variability from oceanic signals."
    filter_pass
        OneOf("lowpass", "highpass", "bandpass", "bandstop")
        Indicates what frequencies to block for the "fft" and "butterworth"
        filters.
    butterworth_order
        [optional, default is 10]

        The order of the butterworth filter.
    lowpass_cutoff
        [optional, default is None, used only if `filter` is "fft" or
        "butterworth" and required if `filter_pass` equals "lowpass",
        "bandpass" or "bandstop"]

        The low frequency cutoff when `filter_pass` equals "lowpass",
        "bandpass", or "bandstop".
    highpass_cutoff
        [optional, default is None, used only if `filter` is "fft" or
        "butterworth" and required if `filter_pass` equals "highpass",
        "bandpass" or "bandstop"]

        The high frequency cutoff when `filter_pass` equals "highpass",
        "bandpass", or "bandstop".
    window_len
        [optional, default is 3]

        "flat", "hanning", "hamming", "bartlett", "blackman"
        Time-series is padded by one half the window length on each end.  The
        `window_len` is then used for the length of the convolution kernel.

        "fft"
        Will soften the edges of the "fft" filter in the frequency domain.
        The larger the number the softer the filter edges.  A value of 1
        will have a brick wall step function which may introduce
        frequencies into the filtered output.

        "tide_usgs", "tide_doodson"
        The `window_len` is set to 33 for "tide_usgs" and 39 for "tide_doodson".
    pad_mode
        [optional, default is "reflect"]

        The method used to pad the time-series.  Uses some of the methods in
        numpy.pad.

        The pad methods "edge", "maximum", "mean", "median", "minimum",
        "reflect", "symmetric", "wrap" are available because they require no
        extra arguments.
    ${input_ts}
    ${start_date}
    ${end_date}
    ${columns}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${print_input}
    ${source_units}
    ${target_units}
    ${round_index}
    ${float_format}
    ${tablefmt}
    """
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
                f"""
                Input vector (length={len(tsd.values)}) needs to be bigger than
                window size ({window_len}).
                """
            )
        )

    if print_input is True:
        ttsd = tsutils.asbestfreq(tsd.copy())
    else:
        ttsd = tsutils.asbestfreq(tsd)

    ntsd = pd.DataFrame()
    for filter_type in filter_types:
        if filter_type == "tide_usgs":
            window_len = 33
        elif filter_type == "tide_doodson":
            window_len = 39

        if filter_type in ("flat", "hanning", "hamming", "bartlett", "blackman"):
            warnings.warn(
                tsutils.error_wrapper(
                    """
                    The `filter_type`s "flat", "hanning", "hamming",
                    "bartlett", and "blackman" are implemented with greater
                    capabilities in the "rolling_window" function in tstoolbox.
                    """
                )
            )
        if filter_type in ("fft", "butterworth"):
            if filter_pass in ("bandpass", "bandstop") and (
                lowpass_cutoff is None or highpass_cutoff is None
            ):
                raise ValueError(
                    tsutils.error_wrapper(
                        f"""
                        The "bandpass" and "bandstop" options for `filter_pass`
                        require values for the `lowpass_cutoff` and
                        `highpass_cutoff` keywords. You have "{lowpass_cutoff}"
                        for `lowpass_cutoff` and "{highpass_cutoff}" for
                        `highpass_cutoff`.
                        """
                    )
                )

            if filter_pass == "lowpass":
                if lowpass_cutoff is None:
                    raise ValueError(
                        tsutils.error_wrapper(
                            f"""
                            The "lowpass" option for `filter_pass` requires
                            a value for `lowpass_cutoff`.  You have
                            "{lowpass_cutoff}".
                            """
                        )
                    )
                if highpass_cutoff is not None:
                    warnings.warn(
                        tsutils.error_wrapper(
                            f"""
                            The `highpass_cutoff` value of {highpass_cutoff} is
                            ignored it `filter_pass` is "lowpass".
                            """
                        )
                    )

            if filter_pass == "highpass":
                if highpass_cutoff is None:
                    raise ValueError(
                        tsutils.error_wrapper(
                            f"""
                            The "highpass" option for `filter_pass` requires
                            a value for `highpass_cutoff`.  You have
                            "{highpass_cutoff}".
                            """
                        )
                    )
                if lowpass_cutoff is not None:
                    warnings.warn(
                        tsutils.error_wrapper(
                            f"""
                            The `lowpass_cutoff` value of {lowpass_cutoff} is
                            ignored it `filter_pass` is "highpass".
                            """
                        )
                    )

        for col in tsd.columns:
            ncol = col.split(":")
            if len(ncol) == 1:
                col_name = f"{ncol[0]}::{filter_type}"
            elif len(ncol) == 2:
                col_name = f"{ncol[0]}:{ncol[1]}:{filter_type}"
            else:
                col_name = f"{ncol[0]}:{ncol[1]}:{ncol[2:]}_{filter_type}"
            if filter_type == "fft":
                tmptsd = pd.DataFrame(
                    _transform(
                        ttsd[col],
                        filter_pass,
                        lowpass_cutoff,
                        highpass_cutoff,
                        window_len,
                    ),
                    index=ttsd.index,
                )
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")
            elif filter_type == "butterworth":
                if filter_pass == "lowpass":
                    wn = lowpass_cutoff
                elif filter_pass == "highpass":
                    wn = highpass_cutoff
                elif filter_pass in ["bandpass", "bandstop"]:
                    wn = [lowpass_cutoff, highpass_cutoff]
                sos = signal.butter(butterworth_order, wn, filter_pass, output="sos")
                tmptsd = pd.DataFrame(signal.sosfilt(sos, ttsd[col]), index=ttsd.index)
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")
            elif filter_type in (
                "flat",
                "hanning",
                "hamming",
                "bartlett",
                "blackman",
                "tide_usgs",
                "tide_doodson",
            ):
                if window_len < 3:
                    continue
                if filter_type == "flat":  # moving average
                    w = np.ones(window_len, "d")
                elif filter_type == "tide_usgs":
                    w = np.array(
                        [
                            -0.00027,
                            -0.00114,
                            -0.00211,
                            -0.00317,
                            -0.00427,
                            -0.00537,
                            -0.00641,
                            -0.00735,
                            -0.00811,
                            -0.00864,
                            -0.00887,
                            -0.00872,
                            -0.00816,
                            -0.00714,
                            -0.0056,
                            -0.00355,
                            -0.00097,
                            0.00213,
                            0.00574,
                            0.0098,
                            0.01425,
                            0.01902,
                            0.024,
                            0.02911,
                            0.03423,
                            0.03923,
                            0.04399,
                            0.04842,
                            0.05237,
                            0.05576,
                            0.0585,
                            0.06051,
                            0.06174,
                            0.06215,
                            0.06174,
                            0.06051,
                            0.0585,
                            0.05576,
                            0.05237,
                            0.04842,
                            0.04399,
                            0.03923,
                            0.03423,
                            0.02911,
                            0.024,
                            0.01902,
                            0.01425,
                            0.0098,
                            0.00574,
                            0.00213,
                            -0.00097,
                            -0.00355,
                            -0.0056,
                            -0.00714,
                            -0.00816,
                            -0.00872,
                            -0.00887,
                            -0.00864,
                            -0.00811,
                            -0.00735,
                            -0.00641,
                            -0.00537,
                            -0.00427,
                            -0.00317,
                            -0.00211,
                            -0.00114,
                            -0.00027,
                        ]
                    )
                elif filter_type == "tide_doodson":
                    w = np.array(
                        [
                            1,
                            0,
                            1,
                            0,
                            0,
                            1,
                            0,
                            1,
                            1,
                            0,
                            2,
                            0,
                            1,
                            1,
                            0,
                            2,
                            1,
                            1,
                            2,
                            0,
                            2,
                            1,
                            1,
                            2,
                            0,
                            1,
                            1,
                            0,
                            2,
                            0,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                            1,
                            0,
                            1,
                        ]
                    )
                else:
                    w = eval(f"np.{filter_type}(window_len)")
                tmptsd = pd.DataFrame(
                    np.convolve(w / w.sum(), ttsd[col], mode="same"), index=ttsd.index
                )
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")

            elif filter_type == "tide_fft":
                """
                The article:
                1981, 'Removing Tidal-Period Variations from Time-Series Data
                Using Low Pass Filters' by Roy Walters and Cythia Heston, in
                Physical Oceanography, Volume 12, pg 112.
                compared several filters and determined that the following
                order from best to worst:
                    1) FFT Transform ramp to 0 in frequency domain from 40 to
                       30 hours,
                    2) Godin
                    3) cosine-Lanczos squared filter
                    4) cosine-Lanczos filter
                """
                tmptsd = pd.DataFrame(
                    fft_lowpass(ttsd[col], 1 / 30.0, 1 / 40.0), index=ttsd.index
                )
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")

            elif filter_type == "kalman":
                # I threw this in from an example on scipy's web site.  I will
                # keep it here, but I can't see an immediate use for in in
                # tidal analysis.  It dampens out all frequencies.

                # Might be able to use it it fill missing values.

                # initial parameters
                sz = (len(ttsd[col]),)  # size of array

                Q = (max(ttsd[col]) - min(ttsd[col])) / 10000.0  # process variance
                Q = 1.0e-2

                # allocate space for arrays
                xhat = np.zeros(sz)  # a posteri estimate of x
                P = np.zeros(sz)  # a posteri error estimate
                xhatminus = np.zeros(sz)  # a priori estimate of x
                Pminus = np.zeros(sz)  # a priori error estimate
                K = np.zeros(sz)  # gain or blending factor

                R = (
                    np.var(ttsd[col]) ** 0.5
                )  # estimate of measurement variance, change to see effect

                # initial guesses
                xhat[0] = np.average(ttsd[col])
                P[0] = 1.0

                for k in range(1, len(ttsd[col])):
                    # time update
                    xhatminus[k] = xhat[k - 1]
                    Pminus[k] = P[k - 1] + Q

                    # measurement update
                    K[k] = Pminus[k] / (Pminus[k] + R)
                    xhat[k] = xhatminus[k] + K[k] * (ttsd[col][k] - xhatminus[k])
                    P[k] = (1 - K[k]) * Pminus[k]

                tmptsd = pd.DataFrame(xhat, index=ttsd.index)
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")

            elif filter_type == "lecolazet1":
                # 1/16 * ( S24 * S25 ) ** 2

                # The UNITS are important.  I think the 1/16 is for feet.  That
                # really makes things painful because I have always wanted
                # TAPPY to be unit blind.  I will have to think about whether
                # to implement this or not.

                # Available for testing but not documented in help.

                relevation = (
                    1.0
                    / 16.0
                    * (
                        delta_diff(ttsd[col], 24, 25)[25:]
                        * delta_diff(ttsd[col], 25, 25)[25:]
                    )
                    ** 2
                )
                tmptsd = pd.DataFrame(relevation, index=ttsd.index)
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")

            elif filter_type == "lecolazet2":
                # 1/64 * S1**3 * A3 * A6 ** 2

                # The UNITS are important.  I think the 1/64 is for feet.  That
                # really makes things painful because I have always wanted
                # TAPPY to be unit blind.  I will have to think about whether
                # to implement this or not.
                pass

            elif filter_type == "tide_mstha":
                blen = 12
                s_list = ["M2", "K1", "M3", "M4"]

                p0 = [1.0] * (len(s_list) * 2 + 2)
                p0[-2] = 0.0
                slope = []
                ntimes = np.arange(2 * blen + 1)
                for d in range(len(ttsd[col]))[blen:-blen]:
                    #      ntimes = (new_dates[d-12:d+12] - new_dates[d]) * 24
                    lsfit = leastsq(
                        lambda x, y: x - y, p0, args=(ttsd[col], ntimes, s_list)
                    )
                    slope.append(lsfit[0][-2])

                relevation = slope
                tmptsd = pd.DataFrame(relevation, index=ttsd.index)
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")

            elif filter_type == "wavelet":
                import pywt

                for wl in pywt.wavelist():
                    w = pywt.Wavelet(wl)

                    max_level = pywt.dwt_max_level(len(ttsd[col]), w.dec_len)
                    a = pywt.wavedec(ttsd[col], w, level=max_level, mode="sym")

                    for i in range(len(a))[1:]:
                        avg = np.average(a[i][:])
                        std = 2.0 * np.std(a[i][:])
                        a[i][(a[i][:] < (avg + std)) & (a[i][:] > (avg - std))] = 0.0

                    y = pywt.waverec(a, w, mode="sym")

                relevation = y
                tmptsd = pd.DataFrame(relevation, index=ttsd.index)
                tmptsd.columns = [col_name]
                ntsd = ntsd.join(tmptsd, how="outer")

            elif filter_type == "cd":
                print(
                    "Complex demodulation filter doesn't work right yet - still testing."
                )

                # kern = np.ones(25) * (1.0 / 25.0)

                # nslice = slice(0, len(ttsd[col]))
                # ns_amplitude = {}
                # ns_phase = {}
                # constituent_residual = {}
                # for key in self.key_list:
                #     ntimes_filled = np.arange(len(ttsd[col])) * 24
                #     yt = new_elev * np.exp(
                #         -1j * self.speed_dict[key]["speed"] * ntimes_filled
                #     )

                #     ns_amplitude[key] = np.absolute(yt)
                #     ns_amplitude[key] = yt.real
                #     ns_amplitude[key] = np.convolve(
                #         ns_amplitude[key], kern, mode="same"
                #     )
                #     print(key, np.average(ns_amplitude[key]))
                #     ns_amplitude[key] = np.convolve(ns_amplitude[key], kern, mode=1)

                #     ns_phase[key] = np.arctan2(yt.imag, yt.real) * rad2deg
                #     ns_phase[key] = np.convolve(ns_phase[key], kern, mode=1)

                #     new_list = [i for i in self.key_list if i != key]
                #     everything_but = self.sum_signals(
                #         new_list, ntimes_filled, self.speed_dict
                #     )
                #     constituent_residual[key] = new_elev - everything_but
                # relevation = everything_but
                # tmptsd = pd.DataFrame(relevation[nslice], index=ttsd.index)
                # tmptsd.columns = [col_name]
                # ntsd = ntsd.join(tmptsd, how="outer")

    ntsd.index.name = "Datetime"
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
