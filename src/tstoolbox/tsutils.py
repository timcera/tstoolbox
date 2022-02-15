# -*- coding: utf-8 -*-
"""A collection of functions used by tstoolbox, wdmtoolbox, ...etc."""

from __future__ import division, print_function

import bz2
import datetime
import gzip
import inspect
import os
import sys
from functools import reduce, wraps
from io import BytesIO, StringIO
from math import gcd
from string import Template
from textwrap import TextWrapper
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import dateparser
import numpy as np
import numpy.typing as npt
import pandas as pd
import pint_pandas
import typic
from _io import TextIOWrapper
from numpy import int64, ndarray
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.tseries.frequencies import to_offset
from scipy.stats.distributions import lognorm, norm
from tabulate import simple_separated_format
from tabulate import tabulate as tb

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@typic.al
def error_wrapper(estr: str) -> str:
    """Wrap estr into error format used by toolboxes."""
    wrapper = TextWrapper(initial_indent="*   ", subsequent_indent="*   ")
    nestr = ["", "*"]
    for paragraph in estr.split("\n\n"):
        nestr.append("\n".join(wrapper.wrap(paragraph.strip())))
        nestr.append("*")
    nestr.append("")
    return "\n".join(nestr)


_CODES = {}
_CODES["SUB_D"]: {
    "N": "Nanoseconds",
    "U": "microseconds",
    "L": "miLliseconds",
    "S": "Secondly",
    "T": "minuTely",
    "H": "Hourly",
}
_CODES["DAILY"]: {
    "D": "calendar Day",
    "B": "Business day",
    "C": "Custom business day (experimental)",
}
_CODES["WEEKLY"]: {
    "W": "Weekly",
    "W-SUN": "Weekly frequency (SUNdays)",
    "W-MON": "Weekly frequency (MONdays)",
    "W-TUE": "Weekly frequency (TUEsdays)",
    "W-WED": "Weekly frequency (WEDnesdays)",
    "W-THU": "Weekly frequency (THUrsdays)",
    "W-FRI": "Weekly frequency (FRIdays)",
    "W-SAT": "Weekly frequency (SATurdays)",
}
_CODES["MONTH"]: {
    "M": "Month end",
    "MS": "Month Start",
    "BM": "Business Month end",
    "BMS": "Business Month Start",
    "CBM": "Custom Business Month end",
    "CBMS": "Custom Business Month Start",
}
_CODES["QUARTERLY"]: {
    "Q": "Quarter end",
    "Q-JAN": "Quarterly, quarter ends end of JANuary",
    "Q-FEB": "Quarterly, quarter ends end of FEBruary",
    "Q-MAR": "Quarterly, quarter ends end of MARch",
    "Q-APR": "Quarterly, quarter ends end of APRil",
    "Q-MAY": "Quarterly, quarter ends end of MAY",
    "Q-JUN": "Quarterly, quarter ends end of JUNe",
    "Q-JUL": "Quarterly, quarter ends end of JULy",
    "Q-AUG": "Quarterly, quarter ends end of AUGust",
    "Q-SEP": "Quarterly, quarter ends end of SEPtember",
    "Q-OCT": "Quarterly, quarter ends end of OCTober",
    "Q-NOV": "Quarterly, quarter ends end of NOVember",
    "Q-DEC": "Quarterly, quarter ends end of DECember",
    "QS": "Quarter Start",
    "QS-JAN": "Quarterly, quarter Starts end of JANuary",
    "QS-FEB": "Quarterly, quarter Starts end of FEBruary",
    "QS-MAR": "Quarterly, quarter Starts end of MARch",
    "QS-APR": "Quarterly, quarter Starts end of APRil",
    "QS-MAY": "Quarterly, quarter Starts end of MAY",
    "QS-JUN": "Quarterly, quarter Starts end of JUNe",
    "QS-JUL": "Quarterly, quarter Starts end of JULy",
    "QS-AUG": "Quarterly, quarter Starts end of AUGust",
    "QS-SEP": "Quarterly, quarter Starts end of SEPtember",
    "QS-OCT": "Quarterly, quarter Starts end of OCTober",
    "QS-NOV": "Quarterly, quarter Starts end of NOVember",
    "QS-DEC": "Quarterly, quarter Starts end of DECember",
    "BQ": "Business Quarter end",
    "BQS": "Business Quarter Start",
}
_CODES["ANNUAL"]: {
    "A": "Annual end",
    "A-JAN": "Annual, year ends end of JANuary",
    "A-FEB": "Annual, year ends end of FEBruary",
    "A-MAR": "Annual, year ends end of MARch",
    "A-APR": "Annual, year ends end of APRil",
    "A-MAY": "Annual, year ends end of MAY",
    "A-JUN": "Annual, year ends end of JUNe",
    "A-JUL": "Annual, year ends end of JULy",
    "A-AUG": "Annual, year ends end of AUGust",
    "A-SEP": "Annual, year ends end of SEPtember",
    "A-OCT": "Annual, year ends end of OCTober",
    "A-NOV": "Annual, year ends end of NOVember",
    "A-DEC": "Annual, year ends end of DECember",
    "AS": "Annual Start",
    "AS-JAN": "Annual, year Starts end of JANuary",
    "AS-FEB": "Annual, year Starts end of FEBruary",
    "AS-MAR": "Annual, year Starts end of MARch",
    "AS-APR": "Annual, year Starts end of APRil",
    "AS-MAY": "Annual, year Starts end of MAY",
    "AS-JUN": "Annual, year Starts end of JUNe",
    "AS-JUL": "Annual, year Starts end of JULy",
    "AS-AUG": "Annual, year Starts end of AUGust",
    "AS-SEP": "Annual, year Starts end of SEPtember",
    "AS-OCT": "Annual, year Starts end of OCTober",
    "AS-NOV": "Annual, year Starts end of NOVember",
    "AS-DEC": "Annual, year Starts end of DECember",
    "BA": "Business Annual end",
    "BA-JAN": "Business Annual, business year ends end of JANuary",
    "BA-FEB": "Business Annual, business year ends end of FEBruary",
    "BA-MAR": "Business Annual, business year ends end of MARch",
    "BA-APR": "Business Annual, business year ends end of APRil",
    "BA-MAY": "Business Annual, business year ends end of MAY",
    "BA-JUN": "Business Annual, business year ends end of JUNe",
    "BA-JUL": "Business Annual, business year ends end of JULy",
    "BA-AUG": "Business Annual, business year ends end of AUGust",
    "BA-SEP": "Business Annual, business year ends end of SEPtember",
    "BA-OCT": "Business Annual, business year ends end of OCTober",
    "BA-NOV": "Business Annual, business year ends end of NOVember",
    "BA-DEC": "Business Annual, business year ends end of DECember",
    "BAS": "Business Annual Start",
    "BS-JAN": "Business Annual Start, business year starts end of JANuary",
    "BS-FEB": "Business Annual Start, business year starts end of FEBruary",
    "BS-MAR": "Business Annual Start, business year starts end of MARch",
    "BS-APR": "Business Annual Start, business year starts end of APRil",
    "BS-MAY": "Business Annual Start, business year starts end of MAY",
    "BS-JUN": "Business Annual Start, business year starts end of JUNe",
    "BS-JUL": "Business Annual Start, business year starts end of JULy",
    "BS-AUG": "Business Annual Start, business year starts end of AUGust",
    "BS-SEP": "Business Annual Start, business year starts end of SEPtember",
    "BS-OCT": "Business Annual Start, business year starts end of OCTober",
    "BS-NOV": "Business Annual Start, business year starts end of NOVember",
    "BS-DEC": "Business Annual Start, business year starts end of DECember",
}

docstrings = {
    "por": r"""por
        The `por` keyword adjusts the operation of `start_date` and `end_date`
        If "False" (the default) choose the indices in the time-series between
        `start_date` and `end_date`.  If "True" and if `start_date` or
        `end_date` is outside of the time-series will extend the time-series by
        inserting the exterior date(s).""",
    "lat": r"""lat
        The latitude of the point. North hemisphere is positive from 0 to 90. South
        hemisphere is negative from 0 to -90.""",
    "lon": r"""lon
        The longitude of the point.  Western hemisphere (west of Greenwich Prime
        Meridian) is negative 0 to -180.  The eastern hemisphere (east of the Greenwich
        Prime Meridian) is positive 0 to 180.""",
    "target_units": r"""target_units: str
        [optional, default is None, transformation]

        The purpose of this option is to specify target units for unit
        conversion.  The source units are specified in the header line
        of the input or using the 'source_units' keyword.

        The units of the input time-series or values are specified as
        the second field of a ':' delimited name in the header line of
        the input or in the 'source_units' keyword.

        Any unit string compatible with the 'pint' library can be used.

        This option will also add the 'target_units' string to the
        column names.""",
    "source_units": r"""source_units: str
        [optional, default is None, transformation]

        If unit is specified for the column as the second field of a ':'
        delimited column name, then the specified units and the
        'source_units' must match exactly.

        Any unit string compatible with the 'pint' library can be
        used.""",
    "names": r"""names: str
        [optional, default is None, transformation]

        If None, the column names are taken from the first row after
        'skiprows' from the input dataset.

        MUST include a name for all columns in the input dataset,
        including the index column.""",
    "index_type": r"""index_type : str
        [optional, default is 'datetime', output format]

        Can be either 'number' or 'datetime'.  Use 'number' with index
        values that are Julian dates, or other epoch reference.""",
    "input_ts": r"""input_ts : str
        [optional though required if using within Python, default is '-'
        (stdin)]

        Whether from a file or standard input, data requires a header of
        column names.  The default header is the first line of the
        input, but this can be changed using the 'skiprows' option.

        Most separators will be automatically detected. Most common date
        formats can be used, but the closer to ISO 8601 date/time
        standard the better.

        Command line:

            +-------------------------+------------------------+
            | Keyword Example         | Description            |
            +=========================+========================+
            | --input_ts=filename.csv | to read 'filename.csv' |
            +-------------------------+------------------------+
            | --input_ts='-'          | to read from standard  |
            |                         | input (stdin)          |
            +-------------------------+------------------------+

            In many cases it is better to use redirection rather that
            use `--input_ts=filename.csv`.  The following are identical:

            From a file:

                command subcmd --input_ts=filename.csv

            From standard input:

                command subcmd --input_ts=- < filename.csv

            The BEST way since you don't have to include `--input_ts=-`
            because that is the default:

                command subcmd < file.csv

            Can also combine commands by piping:

                command subcmd < filein.csv | command subcmd1 > fileout.csv

        As Python Library::

            You MUST use the `input_ts=...` option where `input_ts` can
            be one of a [pandas DataFrame, pandas Series, dict, tuple,
            list, StringIO, or file name].

            If result is a time series, returns a pandas DataFrame.""",
    "columns": r"""columns
        [optional, defaults to all columns, input filter]

        Columns to select out of input.  Can use column names from the
        first line header or column numbers.  If using numbers, column
        number 1 is the first data column.  To pick multiple columns;
        separate by commas with no spaces. As used in `tstoolbox pick`
        command.

        This solves a big problem so that you don't have to create
        a data set with a certain column order, you can rearrange
        columns when data is read in.""",
    "start_date": r"""start_date : str
        [optional, defaults to first date in time-series, input filter]

        The start_date of the series in ISOdatetime format, or 'None'
        for beginning.""",
    "end_date": r"""end_date : str
        [optional, defaults to last date in time-series, input filter]

        The end_date of the series in ISOdatetime format, or 'None' for
        end.""",
    "dropna": r"""dropna : str
        [optional, defauls it 'no', input filter]

        Set `dropna` to 'any' to have records dropped that have NA value
        in any column, or 'all' to have records dropped that have NA in
        all columns.  Set to 'no' to not drop any records.  The default
        is 'no'.""",
    "print_input": r"""print_input
        [optional, default is False, output format]

        If set to 'True' will include the input columns in the output
        table.""",
    "round_index": r"""round_index
        [optional, default is None which will do nothing to the index,
        output format]

        Round the index to the nearest time point.  Can significantly
        improve the performance since can cut down on memory and
        processing requirements, however be cautious about rounding to
        a very course interval from a small one.  This could lead to
        duplicate values in the index.""",
    "float_format": r"""float_format
        [optional, output format]

        Format for float numbers.""",
    "tablefmt": r"""tablefmt : str
        [optional, default is 'csv', output format]

        The table format.  Can be one of 'csv', 'tsv', 'plain',
        'simple', 'grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'latex',
        'latex_raw' and 'latex_booktabs'.""",
    "header": r"""header : str
        [optional, default is 'default', output format]

        This is if you want a different header than is the default for
        this output table.  Pass a list of strings for each column in
        the table.""",
    "pandas_offset_codes": r"""+-------+---------------+
        | Alias | Description   |
        +=======+===============+
        | N     | Nanoseconds   |
        +-------+---------------+
        | U     | microseconds  |
        +-------+---------------+
        | L     | milliseconds  |
        +-------+---------------+
        | S     | Secondly      |
        +-------+---------------+
        | T     | Minutely      |
        +-------+---------------+
        | H     | Hourly        |
        +-------+---------------+
        | D     | calendar Day  |
        +-------+---------------+
        | W     | Weekly        |
        +-------+---------------+
        | M     | Month end     |
        +-------+---------------+
        | MS    | Month Start   |
        +-------+---------------+
        | Q     | Quarter end   |
        +-------+---------------+
        | QS    | Quarter Start |
        +-------+---------------+
        | A     | Annual end    |
        +-------+---------------+
        | AS    | Annual Start  |
        +-------+---------------+

        Business offset codes.

        +-------+------------------------------------+
        | Alias | Description                        |
        +=======+====================================+
        | B     | Business day                       |
        +-------+------------------------------------+
        | BM    | Business Month end                 |
        +-------+------------------------------------+
        | BMS   | Business Month Start               |
        +-------+------------------------------------+
        | BQ    | Business Quarter end               |
        +-------+------------------------------------+
        | BQS   | Business Quarter Start             |
        +-------+------------------------------------+
        | BA    | Business Annual end                |
        +-------+------------------------------------+
        | BAS   | Business Annual Start              |
        +-------+------------------------------------+
        | C     | Custom business day (experimental) |
        +-------+------------------------------------+
        | CBM   | Custom Business Month end          |
        +-------+------------------------------------+
        | CBMS  | Custom Business Month Start        |
        +-------+------------------------------------+

        Weekly has the following anchored frequencies:

        +-------+-------------+-------------------------------+
        | Alias | Equivalents | Description                   |
        +=======+=============+===============================+
        | W-SUN | W           | Weekly frequency (SUNdays)    |
        +-------+-------------+-------------------------------+
        | W-MON |             | Weekly frequency (MONdays)    |
        +-------+-------------+-------------------------------+
        | W-TUE |             | Weekly frequency (TUEsdays)   |
        +-------+-------------+-------------------------------+
        | W-WED |             | Weekly frequency (WEDnesdays) |
        +-------+-------------+-------------------------------+
        | W-THU |             | Weekly frequency (THUrsdays)  |
        +-------+-------------+-------------------------------+
        | W-FRI |             | Weekly frequency (FRIdays)    |
        +-------+-------------+-------------------------------+
        | W-SAT |             | Weekly frequency (SATurdays)  |
        +-------+-------------+-------------------------------+

        Quarterly frequencies (Q, BQ, QS, BQS) and annual frequencies
        (A, BA, AS, BAS) replace the "x" in the "Alias" column to have
        the following anchoring suffixes:

        +-------+----------+-------------+----------------------------+
        | Alias | Examples | Equivalents | Description                |
        +=======+==========+=============+============================+
        | x-DEC | A-DEC    | A           | year ends end of DECember  |
        |       | Q-DEC    | Q           |                            |
        |       | AS-DEC   | AS          |                            |
        |       | QS-DEC   | QS          |                            |
        +-------+----------+-------------+----------------------------+
        | x-JAN |          |             | year ends end of JANuary   |
        +-------+----------+-------------+----------------------------+
        | x-FEB |          |             | year ends end of FEBruary  |
        +-------+----------+-------------+----------------------------+
        | x-MAR |          |             | year ends end of MARch     |
        +-------+----------+-------------+----------------------------+
        | x-APR |          |             | year ends end of APRil     |
        +-------+----------+-------------+----------------------------+
        | x-MAY |          |             | year ends end of MAY       |
        +-------+----------+-------------+----------------------------+
        | x-JUN |          |             | year ends end of JUNe      |
        +-------+----------+-------------+----------------------------+
        | x-JUL |          |             | year ends end of JULy      |
        +-------+----------+-------------+----------------------------+
        | x-AUG |          |             | year ends end of AUGust    |
        +-------+----------+-------------+----------------------------+
        | x-SEP |          |             | year ends end of SEPtember |
        +-------+----------+-------------+----------------------------+
        | x-OCT |          |             | year ends end of OCTober   |
        +-------+----------+-------------+----------------------------+
        | x-NOV |          |             | year ends end of NOVember  |
        +-------+----------+-------------+----------------------------+""",
    "plotting_position_table": r"""+------------+--------+----------------------+-----------------------+
        | Name       | a      | Equation             | Description           |
        |            |        | (i-a)/(n+1-2*a)      |                       |
        +============+========+======================+=======================+
        | weibull    | 0      | i/(n+1)              | mean of sampling      |
        | (default)  |        |                      | distribution          |
        +------------+--------+----------------------+-----------------------+
        | benard     | 0.3    | (i-0.3)/(n+0.4)      | approx. median of     |
        |            |        |                      | sampling distribution |
        +------------+--------+----------------------+-----------------------+
        | filliben   | 0.3175 | (i-0.3175)/(n+0.365) |                       |
        +------------+--------+----------------------+-----------------------+
        | yu         | 0.326  | (i-0.326)/(n+0.348)  |                       |
        +------------+--------+----------------------+-----------------------+
        | tukey      | 1/3    | (i-1/3)/(n+1/3)      | approx. median of     |
        |            |        |                      | sampling distribution |
        +------------+--------+----------------------+-----------------------+
        | blom       | 0.375  | (i-0.375)/(n+0.25)   |                       |
        +------------+--------+----------------------+-----------------------+
        | cunnane    | 2/5    | (i-2/5)/(n+1/5)      | subjective            |
        +------------+--------+----------------------+-----------------------+
        | gringorton | 0.44   | (1-0.44)/(n+0.12)    |                       |
        +------------+--------+----------------------+-----------------------+
        | hazen      | 1/2    | (i-1/2)/n            | midpoints of n equal  |
        |            |        |                      | intervals             |
        +------------+--------+----------------------+-----------------------+
        | larsen     | 0.567  | (i-0.567)/(n-0.134)  |                       |
        +------------+--------+----------------------+-----------------------+
        | gumbel     | 1      | (i-1)/(n-1)          | mode of sampling      |
        |            |        |                      | distribution          |
        +------------+--------+----------------------+-----------------------+
        | california | NA     | i/n                  |                       |
        +------------+--------+----------------------+-----------------------+

        Where 'i' is the sorted rank of the y value, and 'n' is the
        total number of values to be plotted.""",
    "clean": r"""clean
        [optional, default is False, input filter]

        The 'clean' command will repair a input index, removing
        duplicate index values and sorting.""",
    "skiprows": r"""skiprows: list-like or integer or callable
        [optional, default is None which will infer header from first
        line, input filter]

        Line numbers to skip (0-indexed) or number of lines to skip
        (int) at the start of the file.

        If callable, the callable function will be evaluated against the
        row indices, returning True if the row should be skipped and
        False otherwise.  An example of a valid callable argument would
        be

        ``lambda x: x in [0, 2]``.""",
    "groupby": r"""groupby: str
        [optional, default is None, transformation]

        The pandas offset code to group the time-series data into.
        A special code is also available to group 'months_across_years'
        that will group into twelve monthly categories across the entire
        time-series.""",
    "force_freq": r"""force_freq: str
        [optional, output format]

        Force this frequency for the output.  Typically you will only
        want to enforce a smaller interval where tstoolbox will insert
        missing values as needed.  WARNING: you may lose data if not
        careful with this option.  In general, letting the algorithm
        determine the frequency should always work, but this option will
        override.  Use PANDAS offset codes.""",
    "output_names": r"""output_names: str
        [optional, output_format]

        The tstoolbox will change the names of the output columns to include
        some record of the operations used on each column.  The `output_names`
        will override that feature.  Must be a list or tuple equal to the
        number of columns in the output data.""",
}

# Decided this was inelegant, but left here in case I figure out what I want
# and how I want it.
# ntables = {}
# for key in ["SUB_D", "DAILY", "WEEKLY", "QUATERLY", "ANNUAL"]:
#     ntables[key] = tb(_CODES[key].items(),
#                       tablefmt="grid",
#                       headers=["Alias", "Description"],)
#     ntables[key] = "        ".join(ntables[key].splitlines(True))
# codes_table = f"""{ntables["SUB_D"]}
#
#     {ntables["DAILY"]}
#     """
#
# docstrings["pandas_offset_codes"] = codes_table


@typic.constrained(gt=0, lt=1)
class FloatBetweenZeroAndOne(float):
    """0.0 < float < 1.0"""


@typic.constrained(ge=0, le=1)
class FloatBetweenZeroAndOneInclusive(float):
    """0.0 <= float <= 1.0"""


@typic.constrained(ge=0)
class FloatGreaterEqualToZero(float):
    """float >= 0.0"""


@typic.constrained(ge=1)
class FloatGreaterEqualToOne(float):
    """float >= 1.0"""


@typic.constrained(ge=0)
class IntGreaterEqualToZero(int):
    """int >= 0"""


@typic.constrained(ge=1)
class IntGreaterEqualToOne(int):
    """int >= 1"""


@typic.constrained(ge=1, le=3)
class IntBetweenOneAndThree(int):
    """1 <= int <= 3"""


def flatten(list_of_lists):
    if isinstance(list_of_lists, (list, tuple)):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], (list, tuple)):
            return list(flatten(list_of_lists[0])) + list(flatten(list_of_lists[1:]))
        return list(list_of_lists[:1]) + list(flatten(list_of_lists[1:]))
    return list_of_lists


@typic.al
def stride_and_unit(sunit: str) -> Tuple[str, int]:
    """Split a stride/unit combination into component parts."""
    if sunit is None:
        return sunit
    unit = sunit.lstrip("+-. 1234567890")
    stride = sunit[: sunit.index(unit)]
    if len(stride) > 0:
        stride = int(stride)
    else:
        stride = 1
    return unit, stride


@typic.al
def set_ppf(ptype: Optional[Literal["norm", "lognorm", "weibull"]]) -> Callable:
    """Return correct Percentage Point Function for `ptype`."""
    if ptype == "norm":
        ppf = norm.ppf
    elif ptype == "lognorm":
        ppf = lognorm.freeze(0.5, loc=0).ppf
    elif ptype == "weibull":

        def ppf(y: Union[List[float], npt.ArrayLike]) -> npt.ArrayLike:
            """Percentage Point Function for the weibull distibution."""
            return np.log(-np.log(1 - np.array(y)))

    elif ptype is None:

        def ppf(y: npt.ArrayLike) -> npt.ArrayLike:
            return y

    return ppf


PPDICT = {
    "weibull": 0,
    "benard": 0.3,
    "filliben": 0.3175,
    "yu": 0.326,
    "tukey": 1 / 3,
    "blom": 0.375,
    "cunnane": 2 / 5,
    "gringorton": 0.44,
    "hazen": 1 / 2,
    "larsen": 0.567,
    "gumbel": 1,
}


# @typic.al
def set_plotting_position(
    n: Union[int, int64],
    plotting_position: Union[
        float,
        Literal[
            "weibull",
            "benard",
            "bos-levenbach",
            "filliben",
            "yu",
            "tukey",
            "blom",
            "cunnane",
            "gringorton",
            "hazen",
            "larsen",
            "gumbel",
            "california",
        ],
    ] = "weibull",
) -> ndarray:
    """Create plotting position 1D array using linspace."""
    if plotting_position == "california":
        return np.linspace(1.0 / n, 1.0, n)
    try:
        a = PPDICT[plotting_position]
    except KeyError:
        a = float(plotting_position)
    i = np.arange(1, n + 1)
    return (i - a) / float(n + 1 - 2 * a)


@typic.al
def _handle_curly_braces_in_docstring(s: str, **kwargs) -> str:
    """Replace missing keys with a pattern."""
    RET = "{{{}}}"
    try:
        return s.format(**kwargs)
    except KeyError as e:
        keyname = e.args[0]
        return _handle_curly_braces_in_docstring(
            s, **{keyname: RET.format(keyname)}, **kwargs
        )


@typic.al
def copy_doc(source: Callable) -> Callable:
    """Copy docstring from source.

    Parameters
    ----------
    source : Callable
        Function to take doc string from.

    Examples
    --------
    >>> @decorit.copy_doc_params(func_to_copy_from)
    >>> def func(args):
    ...     pass  # function goes here
    # Function uses parameters of given func

    """

    @wraps(source)
    def wrapper_copy_doc(func: Callable) -> Callable:
        if source.__doc__:
            func.__doc__ = source.__doc__  # noqa: WPS125
        return func

    return wrapper_copy_doc


@typic.al
def doc(fdict: dict, **kwargs) -> Callable:
    """Return a decorator that formats a docstring."""

    def f(fn):
        fn.__doc__ = Template(fn.__doc__).safe_substitute(**fdict)

        # kwargs is currently always empty.
        # Could remove, but keeping in case useful in future.
        for attr in kwargs:
            setattr(fn, attr, kwargs[attr])
        return fn

    return f


@typic.al
def parsedate(
    dstr: Optional[str], strftime: Optional[Any] = None, settings: Optional[Any] = None
) -> Timestamp:
    """Use dateparser to parse a wide variety of dates.

    Used for start and end dates.
    """
    if dstr is None:
        return dstr

    # The API should boomerang a datetime.datetime instance and None.
    if isinstance(dstr, datetime.datetime):
        if strftime is None:
            return dstr
        return dstr.strftime(strftime)

    try:
        pdate = pd.to_datetime(dstr)
    except ValueError:
        pdate = dateparser.parse(dstr, settings=settings)

    if pdate is None:
        raise ValueError(
            error_wrapper(
                """
Could not parse date string '{}'.
""".format(
                    dstr
                )
            )
        )

    if strftime is None:
        return pdate
    return pdate.strftime(strftime)


@typic.al
def merge_dicts(*dict_args: dict) -> dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dict_args:
        result.update(d)
    return result


def about(name):
    """Return generic 'about' information used across all toolboxes."""
    import platform

    import pkg_resources

    namever = str(pkg_resources.get_distribution(name.split(".")[0]))
    print("package name = {}\npackage version = {}".format(*namever.split()))

    print("platform architecture = {}".format(platform.architecture()))
    print("platform machine = {}".format(platform.machine()))
    print("platform = {}".format(platform.platform()))
    print("platform processor = {}".format(platform.processor()))
    print("platform python_build = {}".format(platform.python_build()))
    print("platform python_compiler = {}".format(platform.python_compiler()))
    print("platform python branch = {}".format(platform.python_branch()))
    print(
        "platform python implementation = {}".format(platform.python_implementation())
    )
    print("platform python revision = {}".format(platform.python_revision()))
    print("platform python version = {}".format(platform.python_version()))
    print("platform release = {}".format(platform.release()))
    print("platform system = {}".format(platform.system()))
    print("platform version = {}".format(platform.version()))


def _round_index(ntsd: DataFrame, round_index: Optional[str] = None) -> DataFrame:
    """Round the index, typically time, to the nearest interval."""
    if round_index is None:
        return ntsd
    ntsd.index = ntsd.index.round(round_index)
    return ntsd


def _pick_column_or_value(tsd, var):
    """Return a keyword value or a time-series."""
    try:
        var = np.array([float(var)])
    except ValueError:
        var = tsd.loc[:, var].values
    return var


def make_list(*strorlist, **kwds: Any) -> Any:
    """Normalize strings, converting to numbers or lists."""
    try:
        n = kwds.pop("n")
    except KeyError:
        n = None
    if n is not None:
        n = int(n)

    try:
        sep = kwds.pop("sep")
    except KeyError:
        sep = ","

    try:
        kwdname = kwds.pop("kwdname")
    except KeyError:
        kwdname = ""

    try:
        flat = kwds.pop("flat")
    except KeyError:
        flat = True

    if isinstance(strorlist, (list, tuple)):
        # The following will fix ((tuples, in, a, tuple, problem),)
        if flat is True:
            strorlist = flatten(strorlist)

        if len(strorlist) == 1:
            # Normalize lists and tuples of length 1 to scalar for
            # further processing.
            strorlist = strorlist[0]

    if isinstance(strorlist, (list, tuple)) and n is not None and len(strorlist) != n:
        raise ValueError(
            error_wrapper(
                """
The list {0} for "{2}" should have {1} members according to function requirements.
""".format(
                    strorlist, n, kwdname
                )
            )
        )

    if isinstance(strorlist, pd.DataFrame):
        return [strorlist]

    if isinstance(strorlist, pd.Series):
        return [pd.DataFrame(strorlist)]

    if strorlist is None or isinstance(strorlist, (type(None))):
        # None -> None
        #
        return None

    if isinstance(strorlist, (int, float)):
        # 1      -> [1]
        # 1.2    -> [1.2]
        #
        return [strorlist]

    if isinstance(strorlist, (str, bytes)) and (
        strorlist in ["None", "", b"None", b""]
    ):
        # 'None' -> None
        # ''     -> None
        #
        return None

    if isinstance(strorlist, (str, bytes)):
        # '1'   -> [1]
        # '5.7' -> [5.7]

        # Anything other than a scalar int or float continues.
        #
        try:
            return [int(strorlist)]
        except ValueError:
            try:
                return [float(strorlist)]
            except ValueError:
                pass
        # Deal with a str or bytes.
        strorlist = strorlist.strip()

        if isinstance(strorlist, str):
            if "\r" in strorlist or "\n" in strorlist:
                return [StringIO(strorlist)]
            strorlist = strorlist.split(str(sep))

        if isinstance(strorlist, bytes):
            if b"\r" in strorlist or b"\n" in strorlist:
                return [BytesIO(strorlist)]
            strorlist = strorlist.split(bytes(sep, encoding="utf8"))

    if isinstance(strorlist, (StringIO, BytesIO)):
        return strorlist

    if n is None:
        n = len(strorlist)

    # At this point 'strorlist' variable should be a list or tuple.
    if len(strorlist) != n:
        raise ValueError(
            error_wrapper(
                """
The list {0} for "{2}" should have {1} members according to function requirements.
""".format(
                    strorlist, n, kwdname
                )
            )
        )

    # [1, 2, 3]          -> [1, 2, 3]
    # ['1', '2']         -> [1, 2]

    # [1, 'er', 5.6]     -> [1, 'er', 5.6]
    # [1,'er',5.6]       -> [1, 'er', 5.6]
    # ['1','er','5.6']   -> [1, 'er', 5.6]

    # ['1','','5.6']     -> [1, None, 5.6]
    # ['1','None','5.6'] -> [1, None, 5.6]

    ret = []
    for each in strorlist:
        if isinstance(each, (type(None), int, float, pd.DataFrame, pd.Series)):
            ret.append(each)
            continue
        if flat is False and isinstance(each, list):
            ret.append(each)
            continue
        if each is None or each.strip() == "" or each == "None":
            ret.append(None)
            continue
        try:
            ret.append(int(each))
        except ValueError:
            try:
                ret.append(float(each))
            except ValueError:
                ret.append(each)
    return ret


def make_iloc(columns, col_list):
    """Imitates the .ix option with subtracting one to convert."""
    # ["1", "Value2"]    ->    [0, "Value2"]
    # [1, 2, 3]          ->    [0, 1, 2]
    col_list = make_list(col_list)
    ntestc = []
    for i in col_list:
        try:
            ntestc.append(int(i) - 1)
        except ValueError:
            ntestc.append(columns.index(i))
    return ntestc


# NOT SET YET...
#
# Take `air_pressure` from df.loc[:, 1]
# Take `short_wave_rad` from df.loc[:, 'swaverad']
# The `temperature` keyword is set to 23.4 for all time periods
# The `wind_speed` keyword is set to 2.4 and 3.1 in turn
#
# Will output two columns, one with wind_speed equal to 2.4, the next
# with wind_speed equal to 3.1.
#
# API:
# testfunction(input_ts=df,
#              air_pressure='_1',
#              short_wave_rad='swaverad',
#              temperature=23.4,
#              wind_speed=[2.4, 3.1])
#             )
#
# CLI:
# mettoolbox testfunction --air_pressure=_1 \
#                         --short_wave_rad=swaverad \
#                         --temperature 23.4 \
#                         --wind_speed 2.4,3.1 < df.csv


def _normalize_units(
    ntsd: DataFrame,
    source_units: Optional[str],
    target_units: Optional[str],
    source_units_required: bool = False,
) -> DataFrame:
    """
    Following is aspirational and may not reflect the code.

    +--------------+--------------+--------------+--------------+--------------+
    | INPUT        | INPUT        | INPUT        | RETURN       | RETURN       |
    | ntsd.columns | source_units | target_units | source_units | target_units |
    +==============+==============+==============+==============+==============+
    | ["col1:cm",  | ["ft",       | ["m",        | ValueError   |              |
    |  "col2:cm"]  |  "cm"]       |  "cm"]       |              |              |
    +--------------+--------------+--------------+--------------+--------------+
    | ["col1:cm",  | ["cm"]       | ["ft"]       | ValueError   |              |
    |  "col2:cm"]  |              |              |              |              |
    +--------------+--------------+--------------+--------------+--------------+
    | ["col1:cm",  | ["cm"]       | ["ft"]       | ["cm",       | ["ft",       |
    |  "col2"]     |              |              |  ""]         |  ""]         |
    +--------------+--------------+--------------+--------------+--------------+
    | ["col1",     | ["", "cm"]   | ["", "ft"]   | ["",          | ["",         |
    |  "col2:cm"]  |              |              |  "cm"]       |  "ft"]       |
    +--------------+--------------+--------------+--------------+--------------+
    |              | ["cm"]       | ["ft"]       | ["cm"]       | ["ft"]       |
    +--------------+--------------+--------------+--------------+--------------+
    | ["cm"]       | None         | ["ft"]       | ["cm"]       | ["ft"]       |
    +--------------+--------------+--------------+--------------+--------------+

    """
    # Enforce DataFrame
    ntsd = pd.DataFrame(ntsd)

    target_units = make_list(target_units, n=len(ntsd.columns))
    if target_units is not None:
        target_units = ["" if i is None else i for i in target_units]
    isource_units = make_list(source_units, n=len(ntsd.columns))
    if isource_units is not None:
        isource_units = ["" if i is None else i for i in isource_units]

    # Create completely filled list of units from the column names.
    tsource_units = []
    for inx in list(range(len(ntsd.columns))):
        if isinstance(ntsd.columns[inx], (str, bytes)):
            words = ntsd.columns[inx].split(":")
            if len(words) >= 2:
                tsource_units.append(words[1])
            else:
                tsource_units.append("")
        else:
            tsource_units.append(ntsd.columns[inx])

    # Combine isource_units and tsource_units into su.
    su = []
    if isource_units is not None:
        for isource, tsource in zip(isource_units, tsource_units):
            if not tsource:
                tsource = isource
            if isource != tsource:
                raise ValueError(
                    error_wrapper(
                        f"""
The units specified by the "source_units" keyword and in the second ":"
delimited field in the DataFrame column name must match.

"source_unit" keyword is {isource_units}
Column name source units are {tsource_units}
                                                       """
                    )
                )
            su.append(tsource)
    else:
        su = [""] * len(ntsd.columns)

    if source_units_required is True and "" in su:
        raise ValueError(
            error_wrapper(
                f"""
Source units must be specified either using "source_units" keyword of in the
second ":" delimited field in the column name.  Instead you have {su}.
                                           """
            )
        )
    names = []
    for inx, unit in enumerate(su):
        if isinstance(ntsd.columns[inx], (str, bytes)):
            words = ntsd.columns[inx].split(":")
            if unit:
                tmpname = ":".join([words[0], unit])
                if len(words) > 2:
                    tmpname = tmpname + ":" + ":".join(words[2:])
                names.append(tmpname)
            else:
                names.append(":".join(words))
        else:
            names.append(ntsd.columns[inx])
    ntsd.columns = names

    if su is None and target_units is not None:
        raise ValueError(
            error_wrapper(
                f"""
To specify target_units, you must also specify source_units.  You can
specify source_units either by using the `source_units` keyword or placing
in the name of the data column as the second ':' separated field.

The `source_units` keyword must specify units that are convertible
to the `target_units`: {target_units}
"""
            )
        )

    # Convert source_units to target_units.
    if target_units is not None:
        ncolumns = []
        for inx, colname in enumerate(ntsd.columns):
            words = str(colname).split(":")
            if len(words) > 1:
                # convert words[1] to target_units[inx]
                try:
                    # Would be nice in the future to carry around units,
                    # however at the moment most tstoolbox functions will not
                    # work right with units specified.
                    # This single command uses pint to convert units and
                    # the "np.array(..., dtype=float)" command removes pint
                    # units from the converted pandas Series.
                    ntsd[str(colname)] = np.array(
                        pd.Series(
                            ntsd[str(colname)].astype(float), dtype=f"pint[{words[1]}]"
                        ).pint.to(target_units[inx]),
                        dtype=float,
                    )
                    words[1] = target_units[inx]
                except AttributeError:
                    raise ValueError(
                        error_wrapper(
                            """
No conversion between {} and {}.""".format(
                                words[1], target_units[inx]
                            )
                        )
                    )
            ncolumns.append(":".join(words))
        ntsd.columns = ncolumns

    return memory_optimize(ntsd)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def transform_args(**trans_func_for_arg):
    """
    Make a decorator that transforms function arguments before calling the
    function.  Works with plain functions and bounded methods.
    """

    def transform_args_decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def transform_args_wrapper(*args, **kwargs):
            # get a {argname: argval, ...} dict from *args and **kwargs
            # Note: Didn't really need an if/else here but I am assuming
            # that getcallargs gives us an overhead that can be avoided if
            # there's only keyword args.
            val_of_argname = sig.bind(*args, **kwargs)
            val_of_argname.apply_defaults()
            for argname, trans_func in trans_func_for_arg.items():
                val_of_argname.arguments[argname] = trans_func(
                    val_of_argname.arguments[argname]
                )
            # apply transform functions to argument values
            return func(*val_of_argname.args, **val_of_argname.kwargs)

        return transform_args_wrapper

    return transform_args_decorator


@transform_args(
    pick=make_list, names=make_list, source_units=make_list, target_units=make_list
)
@typic.al
def common_kwds(
    input_tsd=None,
    start_date=None,
    end_date=None,
    pick: Optional[List[Union[int, str]]] = None,
    force_freq: Optional[str] = None,
    groupby: Optional[str] = None,
    dropna: Optional[Literal["no", "any", "all"]] = "no",
    round_index: Optional[str] = None,
    clean: bool = False,
    target_units=None,
    source_units=None,
    source_units_required: bool = False,
    bestfreq: bool = True,
    parse_dates: bool = True,
    extended_columns: bool = False,
    skiprows: Optional[Union[int, List[int]]] = None,
    index_type: Literal["datetime", "number"] = "datetime",
    names: Optional[Union[str, List[str]]] = None,
    usecols: Optional[List[Union[int, str]]] = None,
    por: bool = False,
):
    """Process all common_kwds across sub-commands into single function.

    Parameters
    ----------
    input_tsd: DataFrame
        Input data which should be a DataFrame.

    Returns
    -------
    df: DataFrame
        DataFrame altered according to options.

    """
    # The "por" keyword is stupid, since it is a synonym for "dropna" equal
    # to "no".  Discovered this after implementation and should deprecate
    # and remove in the future.
    if por is True:
        dropna = "no"

    ntsd = read_iso_ts(
        input_tsd,
        parse_dates=parse_dates,
        extended_columns=extended_columns,
        dropna=dropna,
        force_freq=force_freq,
        skiprows=skiprows,
        index_type=index_type,
        usecols=usecols,
        clean=clean,
    )

    if names is not None:
        ntsd.columns = names

    ntsd = _pick(ntsd, pick)

    ntsd = _normalize_units(
        ntsd, source_units, target_units, source_units_required=source_units_required
    )

    if clean is True:
        ntsd = ntsd.sort_index()
        ntsd = ntsd[~ntsd.index.duplicated()]

    ntsd = _round_index(ntsd, round_index=round_index)

    if bestfreq is True:
        ntsd = asbestfreq(ntsd, force_freq=force_freq)

    ntsd = _date_slice(ntsd, start_date=start_date, end_date=end_date, por=por)

    if ntsd.index.inferred_type == "datetime64":
        ntsd.index.name = "Datetime"

    if dropna in ["any", "all"]:
        ntsd = ntsd.dropna(axis="index", how=dropna)
    else:
        try:
            if bestfreq is True:
                ntsd = asbestfreq(ntsd, force_freq=force_freq)
        except ValueError:
            pass

    if groupby is not None:
        if groupby == "months_across_years":
            return ntsd.groupby(lambda x: x.month)
        return ntsd.resample(groupby)

    return ntsd


def _pick(tsd: DataFrame, columns: Any) -> DataFrame:
    columns = make_list(columns)
    if not columns:
        return tsd
    ncolumns = []

    for i in columns:
        if i in tsd.columns:
            # if using column names
            ncolumns.append(tsd.columns.tolist().index(i))
            continue

        if i == tsd.index.name:
            # if wanting the index
            # making it -1 that will be evaluated later...
            ncolumns.append(-1)
            continue

        # if using column numbers
        try:
            target_col = int(i) - 1
        except ValueError:
            raise ValueError(
                error_wrapper(
                    """
The name {} isn't in the list of column names
{}.
""".format(
                        i, tsd.columns
                    )
                )
            )
        if target_col < -1:
            raise ValueError(
                error_wrapper(
                    """
The requested column "{}" must be greater than or equal to 0.
First data column is 1, index is column 0.
""".format(
                        i
                    )
                )
            )
        if target_col > len(tsd.columns):
            raise ValueError(
                error_wrapper(
                    """
The request column index {} must be less than or equal to the
number of columns {}.
""".format(
                        i, len(tsd.columns)
                    )
                )
            )

        # columns names or numbers or index organized into
        # numbers in ncolumns
        ncolumns.append(target_col)

    if len(ncolumns) == 1 and ncolumns[0] != -1:
        return pd.DataFrame(tsd[tsd.columns[ncolumns]])

    newtsd = pd.DataFrame()
    for index, col in enumerate(ncolumns):
        if col == -1:
            # Use the -1 marker to indicate index
            jtsd = pd.DataFrame(tsd.index, index=tsd.index)
        else:
            try:
                jtsd = pd.DataFrame(tsd.iloc[:, col], index=tsd.index)
            except IndexError:
                jtsd = pd.DataFrame(tsd.loc[:, col], index=tsd.index)

        newtsd = newtsd.join(jtsd, lsuffix="_{}".format(index), how="outer")
    return newtsd


def _date_slice(
    input_tsd: DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    por=False,
) -> DataFrame:
    """Private function to slice time series."""
    if input_tsd.index.inferred_type == "datetime64":
        if start_date is None:
            start_date = input_tsd.index[0]

        if end_date is None:
            end_date = input_tsd.index[-1]

        if input_tsd.index.tz is None:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        else:
            try:
                start_date = pd.Timestamp(start_date).tz_convert(input_tsd.index.tz)
            except TypeError:
                start_date = pd.Timestamp(start_date).tz_localize(input_tsd.index.tz)
            try:
                end_date = pd.Timestamp(end_date).tz_convert(input_tsd.index.tz)
            except TypeError:
                end_date = pd.Timestamp(end_date).tz_localize(input_tsd.index.tz)

        input_tsd = input_tsd.loc[
            (input_tsd.index >= start_date) & (input_tsd.index <= end_date), :
        ]

        if por is True:
            if start_date < input_tsd.index[0]:
                input_tsd = pd.DataFrame(index=[start_date]).append(input_tsd)
            if end_date > input_tsd.index[-1]:
                input_tsd = input_tsd.append(pd.DataFrame(index=[end_date]))
            input_tsd = asbestfreq(input_tsd)
    return input_tsd


_ANNUALS = {
    0: "DEC",
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}

_WEEKLIES = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}


def asbestfreq(data: DataFrame, force_freq: Optional[str] = None) -> DataFrame:
    """Test to determine best frequency to represent data.

    This uses several techniques.
    0.5.  If index is not DateTimeIndex, return
    1. If force_freq is set use .asfreq.
    2. If data.index.freq is not None, just return.
    3. If data.index.inferred_freq is set use .asfreq.
    4. Use pd.infer_freq - fails if any missing
    5. Use .is_* functions to establish A, AS, A-*, AS-*, Q, QS, M, MS
    6. Use minimum interval to establish the fixed time periods up to weekly
    7. Gives up returning None for PANDAS offset string

    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data

    if force_freq is not None:
        return data.asfreq(force_freq)

    ndata = data.sort_index()
    ndata = ndata[~ndata.index.duplicated()]

    ndiff = (
        ndata.index.values.astype("int64")[1:] - ndata.index.values.astype("int64")[:-1]
    )

    if np.any(ndiff <= 0):
        raise ValueError(
            error_wrapper(
                """
Duplicate or time reversal index entry at record {1} (start count at 0):
"{0}".

Perhaps use the "--clean" keyword on the CLI or "clean=True" if using Python or edit the
input data..
""".format(
                    ndata.index[:-1][ndiff <= 0][0], np.where(ndiff <= 0)[0][0] + 1
                )
            )
        )

    if data.index.freq is not None:
        return data

    # Since pandas doesn't set data.index.freq and data.index.freqstr when
    # using .asfreq, this function returns that PANDAS time offset alias code
    # also.  Not ideal at all.
    #
    # This gets most of the frequencies...
    if data.index.inferred_freq is not None:
        try:
            return data.asfreq(data.index.inferred_freq)
        except ValueError:
            pass

    # pd.infer_freq would fail if given a large dataset
    if len(data.index) > 1000:
        slic = slice(None, 999)
    else:
        slic = slice(None, None)
    try:
        infer_freq = pd.infer_freq(data.index[slic])
    except ValueError:
        infer_freq = None
    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # At this point pd.infer_freq failed probably because of missing values.
    # The following algorithm would not capture things like BQ, BQS
    # ...etc.
    if np.alltrue(data.index.is_year_end):
        infer_freq = "A"
    elif np.alltrue(data.index.is_year_start):
        infer_freq = "AS"
    elif np.alltrue(data.index.is_quarter_end):
        infer_freq = "Q"
    elif np.alltrue(data.index.is_quarter_start):
        infer_freq = "QS"
    elif np.alltrue(data.index.is_month_end):
        if np.all(data.index.month == data.index[0].month):
            # Actually yearly with different ends
            infer_freq = "A-{}".format(_ANNUALS[data.index[0].month])
        else:
            infer_freq = "M"
    elif np.alltrue(data.index.is_month_start):
        if np.all(data.index.month == data.index[0].month):
            # Actually yearly with different start
            infer_freq = "A-{}".format(_ANNUALS[data.index[0].month] - 1)
        else:
            infer_freq = "MS"

    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # Use the minimum of the intervals to test a new interval.
    # Should work for fixed intervals.
    ndiff = sorted(set(ndiff))
    mininterval = np.min(ndiff)
    if mininterval <= 0:
        raise ValueError
    if len(ndiff) == 1:
        ngcd = ndiff[0]
    else:
        ngcd = reduce(gcd, ndiff)
    if ngcd < 1000:
        infer_freq = "{}N".format(ngcd)
    elif ngcd < 1000000:
        infer_freq = "{}U".format(ngcd // 1000)
    elif ngcd < 1000000000:
        infer_freq = "{}L".format(ngcd // 1000000)
    elif ngcd < 60000000000:
        infer_freq = "{}S".format(ngcd // 1000000000)
    elif ngcd < 3600000000000:
        infer_freq = "{}T".format(ngcd // 60000000000)
    elif ngcd < 86400000000000:
        infer_freq = "{}H".format(ngcd // 3600000000000)
    elif ngcd < 604800000000000:
        infer_freq = "{}D".format(ngcd // 86400000000000)
    elif ngcd < 2419200000000000:
        infer_freq = "{}W".format(ngcd // 604800000000000)
        if np.all(data.index.dayofweek == data.index[0].dayofweek):
            infer_freq = infer_freq + "-{}".format(_WEEKLIES[data.index[0].dayofweek])
        else:
            infer_freq = "D"

    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # Give up
    return data


def dedupIndex(
    idx: List[str], fmt: Optional[Any] = None, ignoreFirst: bool = True
) -> Index:
    # fmt:          A string format that receives two arguments:
    #               name and a counter. By default: fmt='%s.%03d'
    # ignoreFirst:  Disable/enable postfixing of first element.
    idx = pd.Series(idx)
    duplicates = idx[idx.duplicated()].unique()
    fmt = "%s.%03d" if fmt is None else fmt
    for name in duplicates:
        dups = idx == name
        ret = [
            fmt % (name, i) if (i != 0 or not ignoreFirst) else name
            for i in range(dups.sum())
        ]
        idx.loc[dups] = ret
    return pd.Index(idx)


@typic.al
def renamer(xloc: str, suffix: Optional[str] = "") -> str:
    """Print the suffix into the third ":" separated field of the header."""
    if suffix is None:
        suffix = ""
    words = xloc.split(":")
    if len(words) == 1:
        words.append("")
        words.append(suffix)
    elif len(words) == 2:
        words.append(suffix)
    elif len(words) == 3 and suffix:
        if words[2]:
            words[2] = words[2] + "_" + suffix
        else:
            words[2] = suffix
    return ":".join(words)


# Utility
def print_input(
    iftrue,
    intds,
    output,
    suffix="",
    date_format=None,
    float_format="g",
    tablefmt="csv",
    showindex="never",
):
    """Print the input time series also."""
    return printiso(
        return_input(iftrue, intds, output, suffix=suffix),
        date_format=date_format,
        float_format=float_format,
        tablefmt=tablefmt,
        showindex=showindex,
    )


def return_input(
    iftrue: Union[bool, str],
    intds: DataFrame,
    output: DataFrame,
    suffix: Optional[str] = "",
    reverse_index: bool = False,
    output_names: List = None,
) -> DataFrame:
    """Print the input time series also."""
    if output_names is None:
        output_names = []
    output.columns = output_names or [renamer(i, suffix) for i in output.columns]
    if iftrue:
        return memory_optimize(
            intds.join(output, lsuffix="_1", rsuffix="_2", how="outer")
        )
    if reverse_index is True:
        return memory_optimize(output.iloc[::-1])
    return memory_optimize(output)


def _apply_across_columns(func, xtsd, **kwds):
    """Apply a function to each column in turn."""
    for col in xtsd.columns:
        xtsd[col] = func(xtsd[col], **kwds)
    return xtsd


def _printiso(
    tsd: DataFrame,
    date_format: Optional[Any] = None,
    sep: str = ",",
    float_format: str = "g",
    showindex: str = "never",
    headers: str = "keys",
    tablefmt: str = "csv",
) -> None:
    """Separate so can use in tests."""
    if isinstance(tsd, (pd.DataFrame, pd.Series)):
        if isinstance(tsd, pd.Series):
            tsd = pd.DataFrame(tsd)

        if tsd.columns.empty:
            tsd = pd.DataFrame(index=tsd.index)

        # Not perfectly true, but likely will use showindex for indices
        # that are not time stamps.
        if showindex is True:
            if not tsd.index.name:
                tsd.index.name = "UniqueID"
        else:
            if not tsd.index.name:
                tsd.index.name = "Datetime"

        print_index = True
        if tsd.index.inferred_type == "datetime64":
            if not tsd.index.name:
                tsd.index.name = "Datetime"
            # Someone made the decision about the name
            # This is how I include time zone info by tacking on to the
            # index.name.
            elif "datetime" not in tsd.index.name.lower():
                tsd.index.name = "Datetime"
        else:
            # This might be overkill, but tstoolbox is for time-series.
            # Revisit if necessary.
            print_index = False

        if tsd.index.name == "UniqueID":
            print_index = False

        if showindex in ["always", "default"]:
            print_index = True

    elif isinstance(tsd, (int, float, tuple, np.ndarray)):
        tablefmt = None

    ntablefmt = None
    if tablefmt in ["csv", "tsv", "csv_nos", "tsv_nos"]:
        sep = {"csv": ",", "tsv": "\t", "csv_nos": ",", "tsv_nos": "\t"}[tablefmt]
        if isinstance(tsd, pd.DataFrame):
            try:
                tsd.to_csv(
                    sys.stdout,
                    float_format="%{}".format(float_format),
                    date_format=date_format,
                    sep=sep,
                    index=print_index,
                )
                return
            except IOError:
                return
        else:
            ntablefmt = simple_separated_format(sep)

    if tablefmt is None:
        print(str(list(tsd))[1:-1])

    if ntablefmt is None:
        all_table = tb(
            tsd,
            tablefmt=tablefmt,
            showindex=showindex,
            headers=headers,
            floatfmt=float_format,
        )
    else:
        all_table = tb(
            tsd,
            tablefmt=ntablefmt,
            showindex=showindex,
            headers=headers,
            floatfmt=float_format,
        )

    if tablefmt in ["csv_nos", "tsv_nos"]:
        print(all_table.replace(" ", ""))
    else:
        print(all_table)


# Make _printiso public.  Keep both around until convert all toolboxes.
printiso = _printiso


def open_local(filein: str) -> TextIOWrapper:
    """
    Open the given input file.

    It can decode various formats too, such as gzip and bz2.

    """
    base, ext = os.path.splitext(os.path.basename(filein))
    if ext in [".gz", ".GZ"]:
        return gzip.open(filein, "rb"), base
    if ext in [".bz", ".bz2"]:
        return bz2.BZ2File(filein, "rb"), base
    return open(filein, "r"), os.path.basename(filein)


def reduce_mem_usage(props):
    """Kept here, but was too aggressive in terms of setting the dtype.

    Not used any longer.
    """
    for col in props.columns:
        try:
            if props[col].dtype == object:  # Exclude strings
                continue
        except AttributeError:
            continue

        # make variables for Int, max and min
        mx = props[col].max()
        mn = props[col].min()

        # test if column can be converted to an integer
        try:
            asint = props[col].astype(np.int64)
            result = all((props[col] - asint) == 0)
        except ValueError:
            # Want missing values to remain missing so
            # they need to remain float.
            result = False

        # Make Integer/unsigned Integer datatypes
        if result is True:
            if mn >= 0:
                if mx < np.iinfo(np.uint8).max:
                    props[col] = props[col].astype(np.uint8)
                elif mx < np.iinfo(np.uint16).max:
                    props[col] = props[col].astype(np.uint16)
                elif mx < np.iinfo(np.uint32).max:
                    props[col] = props[col].astype(np.uint32)
                else:
                    props[col] = props[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    props[col] = props[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    props[col] = props[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    props[col] = props[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    props[col] = props[col].astype(np.int64)

        else:
            # Put here whatever I come up with for float types.
            # Algorithm problem because just looking at limits doesn't help
            # with precision.
            pass

    return props


def memory_optimize(tsd: DataFrame) -> DataFrame:
    """Convert all columns to known types.

    "convert_dtypes" replaced some code here such that the "memory_optimize"
    function might go away.  Kept in case want to add additional
    optimizations.
    """
    tsd.index = pd.Index(tsd.index, dtype=None)
    tsd = tsd.convert_dtypes()
    try:
        tsd.index.freq = pd.infer_freq(tsd.index)
    except (TypeError, ValueError):
        # TypeError: Not datetime like index
        # ValueError: Less than three rows
        pass
    return tsd


def is_valid_url(url: Union[bytes, str], qualifying: Optional[Any] = None) -> bool:
    """Return whether "url" is valid."""
    min_attributes = ("scheme", "netloc")
    qualifying = min_attributes if qualifying is None else qualifying
    token = urlparse(url)
    return all(getattr(token, qualifying_attr) for qualifying_attr in qualifying)


# @typic.al
def read_iso_ts(
    *inindat,
    dropna: Literal["no", "any", "all"] = None,
    force_freq: Optional[str] = None,
    extended_columns: bool = False,
    parse_dates: bool = True,
    skiprows: Optional[Union[int, List[int]]] = None,
    index_type: Literal["datetime", "number"] = "datetime",
    names: Optional[str] = None,
    header: Optional[str] = "infer",
    sep: Optional[str] = ",",
    index_col=0,
    usecols=None,
    **kwds,
    # skiprows: Optional[Union[int, List[int]]] = None,
    # usecols: Optional[List[Union[int, str]]] = None,
) -> pd.DataFrame:
    """Read the format printed by 'printiso' and maybe other formats.

    Parameters
    ----------
    inindat: str, bytes, StringIO, file pointer, file name, DataFrame,
           Series, tuple, list, dict

        The input data.

    Returns
    -------
    df: DataFrame
        Returns a DataFrame.

    """
    # inindat
    #
    # CLI                              API                             TRANSFORM
    # ("-",)                           "-"                             [["-"]]
    #                                                                  all columns from standard input
    #
    # ("1,2,Value",)                   [1, 2, "Value"]                 [["-", 1, 2, "Value"]]
    #                                                                  extract columns from standard input
    #
    # ("fn.csv,skiprows=4",)           ["fn.csv", {"skiprow":4}]       [["fn.csv", "skiprows=4"]]
    #                                                                  all columns from "fn.csv"
    #                                                                  skipping the 5th row
    #
    # ("fn.csv,1,4,Val",)              ["fn.csv", 1, 4, "Val"]         [["fn.csv", 1, 4, "Val"]]
    #                                                                  extract columns from fn.csv
    #
    # ("fn.wdm,1002,20000",)           ["fn.wdm", 1002, 20000]         [["fn.wdm", 1002, 20000]]
    #                                                                  WDM files MUST have DSNs
    #                                                                  extract DSNs from "fn.wdm"
    #
    # ("fn.csv,Val1,Q,4 fn.wdm,201",)  [["fn.csv", "Val1", "Q", 4], ["fn.wdm", 101, 201]]
    #                                                                  extract columns from fn.csv
    #                                                                  extract DSNs from fn.wdm
    #                                  dataframe
    #                                  series                          [[series]]A
    newkwds = {}
    # Olde global way that applies to all sources.
    # usecols = kwds.get("usecols", None)
    # skiprows = kwds.get("skiprows", None)
    # index_col = kwds.get("index_col", 0)
    # parse_dates = kwds.get("parse_dates", True)
    # index_type = kwds.get("index_type", "datetime")
    # sep = kwds.get("sep", ",")
    # header = kwds.get("header", 0)
    clean = kwds.get("clean", False)
    names = kwds.get("names")

    if not inindat:
        inindat = "-"

    # Would want this to be more generic...
    na_values = []
    for spc in range(20)[1:]:
        spcs = " " * spc
        na_values.append(spcs)
        na_values.append(spcs + "nan")

    fstr = "{1}"
    if extended_columns is True:
        fstr = "{0}.{1}"

    if names is not None:
        header = 0
        names = make_list(names)

    if index_type == "number":
        parse_dates = False

    if (
        isinstance(inindat[0], (tuple, list, pd.DataFrame, pd.Series))
        and len(inindat) == 1
    ):
        inindat = inindat[0]
    sources = make_list(inindat, sep=" ", flat=False)
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    lresult_list = []
    zones = set()
    result = pd.DataFrame()
    stdin_df = pd.DataFrame()
    for source_index, source in enumerate(sources):
        res = pd.DataFrame()
        parameters = make_list(source, sep=",")
        if isinstance(parameters, list) and len(parameters) > 0:
            fname = parameters.pop(0)
        else:
            fname = parameters
            parameters = []
        # Python API
        if isinstance(fname, pd.DataFrame):
            if len(parameters) > 0:
                res = fname[parameters]
            else:
                res = fname

        elif isinstance(fname, (pd.Series, dict)):
            res = pd.DataFrame(inindat)

        elif isinstance(fname, (tuple, list, float)):
            res = pd.DataFrame({f"values{source_index}": fname}, index=[0])

        if len(res) == 0:
            # Store keywords for each source.
            newkwds = {}

            pars = [str(i) for i in parameters]
            pars = [i for i in pars if "=" in i]
            pars = [i.split("=") for i in pars]
            for key, val in pars:
                newkwds[key] = val

            parameters = [i for i in parameters if "=" not in str(i)]

            # Command line API
            # Uses wdmtoolbox, hspfbintoolbox, or pd.read_* functions.
            fpi = None
            if fname in ["-", b"-"]:
                # if from stdin format must be the tstoolbox standard
                # pandas read_csv supports file like objects
                if "header" not in kwds:
                    kwds["header"] = 0
                header = 0
                fpi = sys.stdin
            elif isinstance(fname, (StringIO, BytesIO)):
                fpi = fname
                header = 0
            elif os.path.exists(str(fname)):
                # a local file
                # Read all wdm, hdf5, and, xls* files here
                header = "infer"
                sep = ","
                index_col = 0
                usecols = None
                fpi = fname
                _, ext = os.path.splitext(fname)
                if ext.lower() == ".wdm":
                    from wdmtoolbox import wdmtoolbox

                    nres = []
                    for par in parameters:
                        nres.append(wdmtoolbox.extract(",".join([fname] + [str(par)])))
                    res = pd.concat(nres, axis="columns")
                elif ext.lower() == ".hdf5":
                    if len(parameters) == 0:
                        res = pd.read_hdf(fpi, **newkwds)
                    else:
                        res = pd.DataFrame()
                        for i in parameters:
                            res = res.join(
                                pd.read_hdf(fname, key=i, **newkwds), how="outer"
                            )
                elif ext.lower() in [
                    ".xls",
                    ".xlsx",
                    ".xlsm",
                    ".xlsb",
                    ".odf",
                    ".ods",
                    ".odt",
                ]:
                    if len(parameters) == 0:
                        sheet = [0]
                    elif len(parameters) == 1:
                        sheet = parameters[0]
                    else:
                        sheet = make_list(parameters)

                    res = pd.read_excel(
                        fname,
                        sheet_name=sheet,
                        keep_default_na=True,
                        header=header,
                        na_values=na_values,
                        index_col=index_col,
                        usecols=usecols,
                        parse_dates=parse_dates,
                        skiprows=skiprows,
                        **newkwds,
                    )
                    if isinstance(res, dict):
                        res = pd.concat(res, axis="columns")
                        # Collapse columns MultiIndex
                        fi = res.columns.to_flat_index()
                        fi = ["_".join((str(i[0]), str(i[1]))) for i in fi]
                        res.columns = fi

            elif is_valid_url(str(fname)):
                # a url?
                header = "infer"
                fpi = fname
            elif isinstance(fname, bytes):
                # Python API
                if b"\n" in fname or b"\r" in fname:
                    # a string?
                    header = 0
                    fpi = BytesIO(fname)
                else:
                    parameters.insert(0, fname)
                    fname = "-"
                    header = 0
                    fpi = sys.stdin
            elif isinstance(fname, str):
                # Python API
                if "\n" in fname or "\r" in fname:
                    # a string?
                    header = 0
                    fpi = StringIO(fname)
                else:
                    parameters.insert(0, fname)
                    header = 0
                    fpi = sys.stdin
                    fname = "-"
            elif isinstance(fname, (StringIO, BytesIO)):
                # Python API
                header = "infer"
                fpi = fname
            else:
                # Maybe fname and parameters are actual column names of standard input.
                parameters.insert(0, fname)
                fname = "-"
                header = 0
                fpi = sys.stdin

            if len(res) == 0:
                if fname == "-" and not stdin_df.empty:
                    res = stdin_df
                else:
                    res = pd.read_csv(
                        fpi,
                        engine="python",
                        infer_datetime_format=True,
                        keep_default_na=True,
                        skipinitialspace=True,
                        header=header,
                        sep=sep,
                        na_values=na_values,
                        index_col=index_col,
                        parse_dates=parse_dates,
                        skiprows=skiprows,
                        **newkwds,
                    )
                if fname == "-" and stdin_df.empty:
                    stdin_df = res
                res = _pick(res, parameters)

        lresult_list.append(res)
        try:
            zones.add(res.index.tzinfo)
        except AttributeError:
            pass

    first = []
    second = []
    rest = []
    for col in res.columns:
        words = [i.strip() for i in str(col).split(":")]
        nwords = [i.strip("0123456789") for i in words]

        first.append(fstr.format(fname, words[0].strip()))

        if len(words) > 1:
            second.append([nwords[1]])
        else:
            second.append([])
        if len(nwords) > 2:
            rest.append(nwords[2:])
        else:
            rest.append([])
    first = [[i.strip()] for i in dedupIndex(first)]
    res.columns = [":".join(i + j + k) for i, j, k in zip(first, second, rest)]

    # tmpc = res.columns.values
    # for index, i in enumerate(res.columns):
    #    if "Unnamed:" in i:
    #        words = i.split(":")
    #        tmpc[index] = words[0].strip() + words[1].strip()
    # res.columns = tmpc
    res = memory_optimize(res)

    if res.index.inferred_type != "datetime64":
        try:
            res.set_index(0, inplace=True)
        except KeyError:
            pass
    else:
        try:
            words = res.index.name.split(":")
        except AttributeError:
            words = ""
        if len(words) > 1:
            try:
                res.index = res.index.tz_localize(words[1])
            except TypeError:
                pass
            res.index.name = "Datetime:{}".format(words[1])
        else:
            res.index.name = "Datetime"

        # res = asbestfreq(res, force_freq=force_freq)

    if dropna in ["any", "all"]:
        res.dropna(how=dropna, inplace=True)

    if len(lresult_list) > 1:
        epoch = pd.to_datetime("2000-01-01")
        moffset = epoch + to_offset("A")
        offset_set = set()
        for res in lresult_list:
            if res.index.inferred_type != "datetime64":
                continue
            if len(zones) != 1:
                try:
                    res.index = res.index.tz_convert(None)
                except (TypeError, AttributeError):
                    pass

            # Remove duplicate times if hourly and daylight savings.
            if clean is True:
                res = res.sort_index()
                res = res[~res.index.duplicated()]

            res = asbestfreq(res)
            if res.index.inferred_freq is not None and moffset > epoch + to_offset(
                res.index.inferred_freq
            ):
                moffset = epoch + to_offset(res.index.inferred_freq)
                offset_set.add(moffset)

        result = pd.DataFrame()
        for df in lresult_list:
            if len(offset_set) < 2:
                result = result.join(df, how="outer", rsuffix="_r")
            else:
                result = result.join(
                    df.asfreq(moffset - epoch), how="outer", rsuffix="_r"
                )
    else:
        result = lresult_list[0]

    # Assign names to the index and columns.
    if names is not None:
        result.index.name = names.pop(0)
        result.columns = names

    result.sort_index(inplace=True)
    return result.convert_dtypes()
