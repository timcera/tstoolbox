"""Collection of functions for the manipulation of time series."""

import warnings

import numpy as np
from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def add_trend(
    start_offset: float,
    end_offset: float,
    start_index: int = 0,
    end_index: int = -1,
    input_ts="-",
    columns=None,
    clean=False,
    start_date=None,
    end_date=None,
    dropna="no",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Add a trend.

    Adds a linear interpolated trend to the input data.  The trend
    values start at [`start_index`, `start_offset`] and end at
    [`end_index`, `end_offset`].

    Parameters
    ----------
    start_offset : float
        The starting value for the applied trend.  This is the starting
        value for the linear interpolation that will be added to the
        input data.

    end_offset : float
        The ending value for the applied trend.  This is the ending
        value for the linear interpolation that will be added to the
        input data.

    start_index : int
        [optional, default is 0, transformation]

        The starting index where `start_offset` will be initiated.  Rows
        prior to `start_index` will not be affected.

    end_index : int
        [optional, default is -1, transformation]

        The ending index where `end_offset` will be set.  Rows after
        `end_index` will not be affected.

    ${input_ts}

    ${columns}

    ${start_date}

    ${end_date}

    ${dropna}

    ${clean}

    ${round_index}

    ${skiprows}

    ${index_type}

    ${names}

    ${source_units}

    ${target_units}

    ${print_input}

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
    # Need it to be float since will be using np.nan
    ntsd = tsd.copy().astype("float64")

    ntsd.iloc[:, :] = np.nan
    ntsd.iloc[start_index, :] = start_offset
    ntsd.iloc[end_index, :] = end_offset
    ntsd = ntsd.interpolate(method="values")

    ntsd = ntsd + tsd

    ntsd = tsutils.memory_optimize(ntsd)
    return tsutils.return_input(print_input, tsd, ntsd, "trend")
