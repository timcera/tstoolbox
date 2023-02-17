"""Collection of functions for the manipulation of time series."""

import warnings

from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@tsutils.doc(tsutils.docstrings)
def date_slice(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    round_index=None,
    source_units=None,
    target_units=None,
):
    """Print out data to the screen between start_date and end_date.

    This isn't really useful anymore because "start_date" and "end_date"
    are available in all sub-commands.

    Parameters
    ----------
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${skiprows}
    ${index_type}
    ${names}
    ${float_format}
    ${source_units}
    ${target_units}
    ${round_index}
    ${tablefmt}
    """
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
