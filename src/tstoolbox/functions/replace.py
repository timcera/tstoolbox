"""Collection of functions for the manipulation of time series."""


import warnings
from typing import List, Optional, Union

from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@tsutils.transform_args(from_values=tsutils.make_list, to_values=tsutils.make_list)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def replace(
    from_values: Optional[List[Optional[Union[float, int, str]]]],
    to_values: Optional[List[Optional[Union[float, int, str]]]],
    round_index=None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Return a time-series replacing values with others.

    Parameters
    ----------
    from_values
        All values in this comma separated list are replaced with the
        corresponding value in to_values.  Use the string 'None' to
        represent a missing value.  If using 'None' as a from_value it
        might be easier to use the "fill" subcommand instead.
    to_values
        All values in this comma separated list are the replacement
        values corresponding one-to-one to the items in from_values.
        Use the string 'None' to represent a missing value.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${round_index}
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
    if from_values is None:
        from_values = [None]
    if to_values is None:
        to_values = [None]
    ntsd = tsd.replace(from_values, to_values)

    return tsutils.return_input(print_input, tsd, ntsd, "replace")
