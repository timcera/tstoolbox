"""Collection of functions for the manipulation of time series."""

import warnings
from typing import Union

import pandas as pd
from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def createts(
    input_ts=None,
    freq: str = None,
    fillvalue: Union[float, int] = None,
    index_type="datetime",
    start_date=None,
    end_date=None,
):
    """Create empty time series, optionally fill with a value.

    Parameters
    ----------
    freq : str
        [optional, default is None]

        To use this form `start_date` and `end_date` must be supplied
        also.  The `freq` option is the pandas date offset code used to create
        the index.

        Python example::

            freq='A'

        Command line example::

            --freq='A'

        ${pandas_offset_codes}

    fillvalue
        [optional, default is None]

        The fill value for the time-series.  The default is None, which
        generates the date/time stamps only.

    ${input_ts}

    ${start_date}

    ${end_date}

    ${index_type}

    ${tablefmt}
    """
    if input_ts is None and (
        (start_date is None) or (end_date is None) or (freq is None)
    ):
        raise ValueError(
            tsutils.error_wrapper(
                f"""
If input_ts is None, then start_date, end_date, and freq must be supplied.

Instead you have:
start_date = {start_date},
end_date = {end_date},
freq = {freq}
"""
            )
        )

    if input_ts is not None:
        tsd = tsutils.common_kwds(
            input_ts,
            index_type=index_type,
            start_date=start_date,
            end_date=end_date,
        )
        tsd = pd.DataFrame([fillvalue] * len(tsd.index), index=tsd.index)
    else:
        tindex = pd.date_range(start=start_date, end=end_date, freq=freq)
        tsd = pd.DataFrame([fillvalue] * len(tindex), index=tindex)
    return tsd
