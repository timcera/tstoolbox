# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings
from typing import List, Optional, Union

import mando
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


warnings.filterwarnings("ignore")


@mando.command("accumulate", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def accumulate_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    statistic="sum",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
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
    tsutils.printiso(
        accumulate(
            input_ts=input_ts,
            skiprows=skiprows,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            statistic=statistic,
            round_index=round_index,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


@tsutils.transform_args(
    statistic=tsutils.make_list,
    columns=tsutils.make_list,
    names=tsutils.make_list,
    source_units=tsutils.make_list,
    target_units=tsutils.make_list,
)
@typic.al
@tsutils.copy_doc(accumulate_cli)
def accumulate(
    input_ts="-",
    columns: Optional[Union[str, List]] = None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    statistic: List[Literal["sum", "max", "min", "prod"]] = ["sum"],
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names: Optional[List] = None,
    source_units: Optional[List] = None,
    target_units: Optional[List] = None,
    print_input=False,
):
    """Calculate accumulating statistics."""
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
        tmptsd = eval("tsd.cum{}()".format(stat))
        tmptsd.columns = [tsutils.renamer(i, stat) for i in tmptsd.columns]
        ntsd = ntsd.join(tmptsd, how="outer")
    return tsutils.return_input(print_input, tsd, ntsd)
