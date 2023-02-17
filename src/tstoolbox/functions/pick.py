"""Collection of functions for the manipulation of time series."""

import warnings

from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@tsutils.doc(tsutils.docstrings)
def pick(
    columns,
    input_ts="-",
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    clean=False,
):
    """Will pick a column or list of columns from input [DEPRECATED].

    DEPRECATED: Effectively replaced by the "columns" keyword available
    in all other functions.

    Will be removed in a future version of `tstoolbox`.

    Can use column names or column numbers.  If using numbers, column
    number 1 is the first data column.

    Parameters
    ----------
    ${columns}
    ${input_ts}
    ${start_date}
    ${end_date}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${source_units}
    ${target_units}
    ${round_index}
    ${tablefmt}
    """
    warnings.warn(
        tsutils.error_wrapper(
            """
            DEPRECATED in favor of using the "columns" keyword available in all
            other functions.

            Will be removed in a future version of `tstoolbox`.
            """
        )
    )
    return tsutils.common_kwds(
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
