"""Collection of functions for the manipulation of time series."""

# Standard library imports
import warnings
from typing import Literal

# Third party imports
import pandas as pd

# Local folder imports
from ..toolbox_utils.src.toolbox_utils import tsutils

try:
    # Third party imports
    from pydantic import validate_arguments
except ImportError:
    # Third party imports
    from pydantic import validate_call as validate_arguments

warnings.filterwarnings("ignore")


@validate_arguments
@tsutils.transform_args(
    statistic=tsutils.make_list,
    columns=tsutils.make_list,
    names=tsutils.make_list,
    source_units=tsutils.make_list,
    target_units=tsutils.make_list,
)
@tsutils.doc(tsutils.docstrings)
def accumulate(
    input_ts="-",
    columns: str | list | None = None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    statistic: str | list[Literal["sum", "max", "min", "prod"]] = "sum",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names: list | None = None,
    source_units: list | None = None,
    target_units: list | None = None,
    print_input=False,
):
    """
    Calculate accumulating statistics.

    Parameters
    ----------
    statistic : Union(str, list(str))
        [optional, default is "sum", transformation]

        OneOrMore("sum", "max", "min", "prod")

        Python example::
            statistic=["sum", "max"]

        Command line example::
            --statistic=sum,max
    ${input_ts}
    ${start_date}
    ${end_date}
    ${skiprows}
    ${names}
    ${columns}
    ${dropna}
    ${clean}
    ${source_units}
    ${target_units}
    ${round_index}
    ${index_type}
    ${print_input}
    ${tablefmt}
    """
    statistic = tsutils.make_list(statistic)
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
    ntsd = pd.DataFrame()

    for stat in statistic:
        tmptsd = eval(f"tsd.cum{stat}()")
        tmptsd.columns = [tsutils.renamer(i, stat) for i in tmptsd.columns]
        ntsd = ntsd.join(tmptsd, how="outer")
    return tsutils.return_input(print_input, tsd, ntsd)
