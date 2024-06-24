import os
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox
from tstoolbox.functions.filter import FILTERS, filter

test_sinwave = """Datetime,0::,0::peak,0::valley
2000-01-01 00:00:00,0.0,,
2000-01-01 01:00:00,0.258819045103,,
2000-01-01 02:00:00,0.5,,
2000-01-01 03:00:00,0.707106781187,,
2000-01-01 04:00:00,0.866025403784,,
2000-01-01 05:00:00,0.965925826289,,
2000-01-01 06:00:00,1.0,1.0,
2000-01-01 07:00:00,0.965925826289,,
2000-01-01 08:00:00,0.866025403784,,
2000-01-01 09:00:00,0.707106781187,,
2000-01-01 10:00:00,0.5,,
2000-01-01 11:00:00,0.258819045103,,
2000-01-01 12:00:00,1.22464679915e-16,,
2000-01-01 13:00:00,-0.258819045103,,
2000-01-01 14:00:00,-0.5,,
2000-01-01 15:00:00,-0.707106781187,,
2000-01-01 16:00:00,-0.866025403784,,
2000-01-01 17:00:00,-0.965925826289,,
2000-01-01 18:00:00,-1.0,,-1.0
2000-01-01 19:00:00,-0.965925826289,,
2000-01-01 20:00:00,-0.866025403784,,
2000-01-01 21:00:00,-0.707106781187,,
2000-01-01 22:00:00,-0.5,,
2000-01-01 23:00:00,-0.258819045103,,
"""


class TestFilter(TestCase):
    def setUp(self):
        self.ats = tstoolbox.read(os.path.join("tests", "data_sine.csv"))
        self.ats.index.name = "Datetime"
        self.ats.columns = ["Value"]

        self.flat_3 = self.ats.join(
            tstoolbox.read(os.path.join("tests", "data_filter_flat.csv"))
        )
        self.flat_3.columns = ["Value", "Value::flat_filter"]

        self.hanning = self.ats.join(
            tstoolbox.read(os.path.join("tests", "data_filter_hanning.csv"))
        )
        self.hanning.columns = ["Value", "Value::hanning_filter"]

        self.fft_lowpass = self.ats.join(
            tstoolbox.read(os.path.join("tests", "data_filter_fft_lowpass.csv"))
        )
        self.fft_lowpass.columns = ["Value", "Value::fft_filter"]

        self.fft_highpass = self.ats.copy()
        self.fft_highpass.columns = ["Value::filter"]
        self.fft_highpass = self.ats.join(self.fft_highpass)

    def test_filter_flat(self):
        out = pd.DataFrame(
            tstoolbox.filter(
                "flat",
                "lowpass",
                input_ts="tests/data_sine.csv",
                print_input=True,
                window_len=5,
            ).iloc[:, [0, -1]]
        )
        self.maxDiff = None
        assert_frame_equal(out, self.flat_3, check_column_type=False, check_names=False)

    def test_filter_hanning(self):
        out = pd.DataFrame(
            tstoolbox.filter(
                "hanning",
                "lowpass",
                input_ts="tests/data_sine.csv",
                print_input=True,
                window_len=5,
            ).iloc[:, [0, -1]]
        )
        self.maxDiff = None
        assert_frame_equal(
            out, self.hanning, check_column_type=False, check_names=False
        )

    @staticmethod
    def test_large_window_len():
        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.filter(
                "flat", "lowpass", input_ts="tests/data_sine.csv", window_len=1000
            )
        assert r"Input vector (length=" in str(e_info.value)

    @staticmethod
    def test_filter_type():
        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.filter("flatter", "lowpass", input_ts="tests/data_sine.csv")
        assert r"validation error" in str(e_info.value)

    @staticmethod
    def test_filter_all():
        _ = tstoolbox.filter(
            FILTERS,
            "lowpass",
            lowpass_cutoff=1 / 10,
            input_ts="tests/data_mayport_8720220_water_level.csv,1",
            window_len=5,
        )


test_series = pd.Series(
    np.random.randn(100), index=pd.date_range("2020-01-01", periods=100, freq="h")
)


@pytest.mark.parametrize(
    "filter_types, filter_pass, butterworth_order, input_ts, lowpass_cutoff, highpass_cutoff, window_len, expected_warning, expected_error, test_id",
    [
        # Happy path tests
        (
            "fft",
            "lowpass",
            1,
            test_series,
            0.1,
            None,
            3,
            None,
            None,
            "fft_lowpass",
        ),
        (
            "butterworth",
            "highpass",
            2,
            test_series,
            None,
            0.2,
            3,
            None,
            None,
            "butterworth_highpass",
        ),
        (
            "hanning",
            "lowpass",
            1,
            test_series,
            None,
            None,
            5,
            None,
            None,
            "hanning_lowpass",
        ),
        (
            "tide_usgs",
            "bandpass",
            1,
            test_series,
            0.05,
            0.1,
            33,
            None,
            None,
            "tide_usgs_bandpass",
        ),
        # Edge cases
        (
            "fft",
            "bandpass",
            1,
            test_series,
            0.05,
            0.1,
            3,
            None,
            None,
            "fft_bandpass_short_series",
        ),
        (
            "butterworth",
            "bandstop",
            3,
            test_series,
            0.05,
            0.1,
            3,
            None,
            None,
            "butterworth_bandstop_max_stages",
        ),
        # Error cases
        (
            "fft",
            "lowpass",
            1,
            test_series.iloc[:2],
            0.1,
            None,
            3,
            None,
            ValueError,
            "fft_lowpass_too_short_series",
        ),
        (
            "butterworth",
            "highpass",
            4,
            test_series,
            None,
            0.2,
            9,
            None,
            None,
            "butterworth_highpass_invalid_stages",
        ),
        (
            "hanning",
            "lowpass",
            1,
            test_series,
            None,
            None,
            -1,
            None,
            ValueError,
            "hanning_lowpass_invalid_window_len",
        ),
        (
            "tide_usgs",
            "bandpass",
            1,
            test_series,
            None,
            None,
            400000,
            None,
            ValueError,
            "tide_usgs_bandpass_missing_cutoffs",
        ),
    ],
)
def test_filter(
    filter_types,
    filter_pass,
    butterworth_order,
    input_ts,
    lowpass_cutoff,
    highpass_cutoff,
    window_len,
    expected_warning,
    expected_error,
    test_id,
):
    # Arrange
    if expected_warning:
        with pytest.warns(expected_warning):
            result = filter(
                filter_types=filter_types,
                filter_pass=filter_pass,
                butterworth_order=butterworth_order,
                input_ts=input_ts,
                lowpass_cutoff=lowpass_cutoff,
                highpass_cutoff=highpass_cutoff,
                window_len=window_len,
            )
    elif expected_error:
        with pytest.raises(expected_error):
            result = filter(
                filter_types=filter_types,
                filter_pass=filter_pass,
                butterworth_order=butterworth_order,
                input_ts=input_ts,
                lowpass_cutoff=lowpass_cutoff,
                highpass_cutoff=highpass_cutoff,
                window_len=window_len,
            )
    else:
        # Act
        result = filter(
            filter_types=filter_types,
            filter_pass=filter_pass,
            butterworth_order=butterworth_order,
            input_ts=input_ts,
            lowpass_cutoff=lowpass_cutoff,
            highpass_cutoff=highpass_cutoff,
            window_len=window_len,
        )

        # Assert
        assert (
            not result.empty
        ), f"Test failed for {test_id}. Result should not be empty."
