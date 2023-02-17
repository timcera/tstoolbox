"""Collection of functions for the manipulation of time series."""

from typing import Literal, Optional

import pandas as pd
from pydantic import PositiveInt, conint, validate_arguments
from toolbox_utils import tsutils


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def pct_change(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    periods: PositiveInt = 1,
    fill_method: Literal["backfill", "bfill", "ffill", "pad"] = "pad",
    limit: Optional[conint(ge=0)] = None,
    freq: str = None,
    print_input=False,
    round_index=None,
    source_units=None,
    target_units=None,
):
    """Return the percent change between times.

    Parameters
    ----------
    periods : int
        [optional, default is 1]

        The number of intervals to calculate percent change across.

    fill_method : str
        [optional, defaults to 'pad']

        Fill method for NA.  Defaults to 'pad'.

    limit
        [optional, defaults to None]

        Is the minimum number of consecutive NA values where no more filling
        will be made.

    freq : str
        [optional, defaults to None]

        A pandas time offset string to represent the interval.

        ${pandas_offset_codes}

    ${input_ts}

    ${columns}

    ${start_date}

    ${end_date}

    ${dropna}

    ${skiprows}

    ${index_type}

    ${names}

    ${clean}

    ${source_units}

    ${target_units}

    ${print_input}

    ${float_format}

    ${round_index}

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

    # Trying to save some memory
    otsd = tsd.copy() if print_input else pd.DataFrame()
    return tsutils.return_input(
        print_input,
        otsd,
        tsd.pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq
        ),
        "pctchange",
    )
