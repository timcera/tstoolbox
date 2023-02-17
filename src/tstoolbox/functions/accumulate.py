"""Collection of functions for the manipulation of time series."""

import warnings
from typing import List, Literal, Optional, Union

import pandas as pd
from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@tsutils.transform_args(
    statistic=tsutils.make_list,
    columns=tsutils.make_list,
    names=tsutils.make_list,
    source_units=tsutils.make_list,
    target_units=tsutils.make_list,
)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def accumulate(
    input_ts="-",
    columns: Optional[Union[str, List]] = None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    statistic: Union[str, List[Literal["sum", "max", "min", "prod"]]] = "sum",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names: Optional[List] = None,
    source_units: Optional[List] = None,
    target_units: Optional[List] = None,
    print_input=False,
):
    """Calculate accumulating statistics.

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
