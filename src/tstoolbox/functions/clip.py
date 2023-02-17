"""Collection of functions for the manipulation of time series."""

import warnings

from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def clip(
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    a_min: float = None,
    a_max: float = None,
    round_index=None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Return a time-series with values limited to [a_min, a_max].

    Parameters
    ----------
    a_min
        [optional, defaults to None, transformation]

        All values lower than this will be set to this value.
        Default is None.

    a_max
        [optional, defaults to None, transformation]

        All values higher than this will be set to this value.
        Default is None.

    ${input_ts}

    ${columns}

    ${start_date}

    ${end_date}

    ${dropna}

    ${clean}

    ${skiprows}

    ${index_type}

    ${print_input}

    ${names}

    ${source_units}

    ${target_units}

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
    ntsd = tsd.clip(lower=a_min, upper=a_max)

    return tsutils.return_input(print_input, tsd, ntsd, "clip")
