"""Collection of functions for the manipulation of time series."""

from typing import Optional

from pydantic import validate_arguments
from toolbox_utils import tsutils


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def converttz(
    fromtz: str,
    totz: Optional[str],
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
):
    """Convert the time zone of the index.

    Parameters
    ----------
    fromtz : str
        The time zone of the original time-series.  The 'EST', and
        'America/New_York' could in some sense be thought of as the
        same, however 'EST' would force the time index to have the same
        offset from UTC, regardless of daylight savings time, where
        'America/New_York' would implement the appropriate daylight
        savings offsets.
    totz : str
        The time zone of the converted time-series.  Same note applies
        as for `fromtz`.  Needs to be different from `fromtz`.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${round_index}
    ${dropna}
    ${clean}
    ${skiprows}
    ${index_type}
    ${names}
    ${source_units}
    ${target_units}
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

    # TODO Should test that 'fromtz' matches time zone that might be already
    # set in tsd.

    if totz == "None" or not totz:
        totz = None

    try:
        tsd = tsd.tz_localize(fromtz).tz_convert(totz)
    except TypeError:
        tsd = tsd.tz_convert(totz)
    tsd = tsutils.memory_optimize(tsd)
    return tsd
