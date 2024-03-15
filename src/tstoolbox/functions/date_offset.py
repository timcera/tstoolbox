"""Collection of functions for the manipulation of time series."""

import pandas as pd

from ..toolbox_utils.src.toolbox_utils import tsutils

try:
    from pydantic import validate_arguments
except ImportError:
    from pydantic import validate_call as validate_arguments


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def date_offset(
    intervals: int,
    offset: str,
    columns=None,
    dropna="no",
    clean=False,
    skiprows=None,
    input_ts="-",
    start_date=None,
    end_date=None,
    names=None,
    index_type="datetime",
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Apply a date offset to a time-series index.

    If you want to adjust to a different time-zone then should use the
    "converttz" tstoolbox command.

    Parameters
    ----------
    intervals : int
        Number of intervals of `offset` to shift the time index.  A positive
        integer moves the index forward, negative moves it backwards.

    offset : str
        Any of the Pandas offset codes.  This is only the offset code
        and doesn't include a prefixed interval.

        ${pandas_offset_codes}

    ${input_ts}

    ${start_date}

    ${end_date}

    ${columns}

    ${round_index}

    ${dropna}

    ${clean}

    ${skiprows}

    ${index_type}

    ${source_units}

    ${target_units}

    ${names}

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

    if offset in {"A", "AS"}:
        tsd.index = tsd.index + pd.DateOffset(years=intervals)
    elif offset in {"M", "MS"}:
        tsd.index = tsd.index + pd.DateOffset(months=intervals)
    elif offset in {"W"}:
        tsd.index = tsd.index + pd.DateOffset(weeks=intervals)
    else:
        return tsd.shift(intervals, offset)
    return tsd
