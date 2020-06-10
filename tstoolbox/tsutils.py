"""A collection of functions used by tstoolbox, wdmtoolbox, ...etc."""

from __future__ import division, print_function

import bz2
import datetime
import gzip
import os
import sys
from textwrap import TextWrapper
from functools import reduce
from io import StringIO
from math import gcd
from urllib.parse import urlparse

import dateparser
import numpy as np
from pint import UnitRegistry
import pandas as pd
from scipy.stats.distributions import norm
from scipy.stats.distributions import lognorm
from tabulate import simple_separated_format
from tabulate import tabulate as tb

UREG = UnitRegistry()


WRAPPER = TextWrapper(initial_indent="*   ", subsequent_indent="*   ")


def error_wrapper(estr):
    """ Wrap estr into error format used by toolboxes. """
    nestr = ["", "*"]
    for paragraph in estr.split("\n\n"):
        nestr.append("\n".join(WRAPPER.wrap(paragraph.strip())))
        nestr.append("*")
    nestr.append("")
    return "\n".join(nestr)


docstrings = {
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
    "pandas_offset_codes": r"""+-------+------------------------------------+
        | Alias | Description                        |
        +=======+====================================+
        | B     | Business day                       |
        +-------+------------------------------------+
        | C     | Custom business day (experimental) |
        +-------+------------------------------------+
        | D     | calendar Day                       |
        +-------+------------------------------------+
        | W     | Weekly                             |
        +-------+------------------------------------+
        | M     | Month end                          |
        +-------+------------------------------------+
        | BM    | Business Month end                 |
        +-------+------------------------------------+
        | CBM   | Custom Business Month end          |
        +-------+------------------------------------+
        | MS    | Month Start                        |
        +-------+------------------------------------+
        | BMS   | Business Month Start               |
        +-------+------------------------------------+
        | CBMS  | Custom Business Month Start        |
        +-------+------------------------------------+
        | Q     | Quarter end                        |
        +-------+------------------------------------+
        | BQ    | Business Quarter end               |
        +-------+------------------------------------+
        | QS    | Quarter Start                      |
        +-------+------------------------------------+
        | BQS   | Business Quarter Start             |
        +-------+------------------------------------+
        | A     | Annual end                         |
        +-------+------------------------------------+
        | BA    | Business Annual end                |
        +-------+------------------------------------+
        | AS    | Annual Start                       |
        +-------+------------------------------------+
        | BAS   | Business Annual Start              |
        +-------+------------------------------------+
        | H     | Hourly                             |
        +-------+------------------------------------+
        | T     | Minutely                           |
        +-------+------------------------------------+
        | S     | Secondly                           |
        +-------+------------------------------------+
        | L     | milliseconds                       |
        +-------+------------------------------------+
        | U     | microseconds                       |
        +-------+------------------------------------+
        | N     | Nanoseconds                        |
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
}


def stride_and_unit(sunit):
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


def set_ppf(ptype):
    """Return correct Percentage Point Function for `ptype`."""
    if ptype == "norm":
        return norm.ppf
    elif ptype == "lognorm":
        return lognorm.freeze(0.5, loc=0).ppf
    elif ptype == "weibull":

        def ppf(y):
            """Percentage Point Function for the weibull distibution."""
            return np.log(-np.log((1 - np.array(y))))

        return ppf
    elif ptype is None:

        def ppf(y):
            return y

        return ppf


def _plotting_position_equation(i, n, a):
    """Parameterized, generic plotting position equation."""
    return (i - a) / float(n + 1 - 2 * a)


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


def set_plotting_position(n, plotting_position="weibull"):
    """Create plotting position 1D array using linspace."""
    if plotting_position == "california":
        return np.linspace(1.0 / n, 1.0, n)
    try:
        a = PPDICT[plotting_position]
    except KeyError:
        a = float(plotting_position)
    return _plotting_position_equation(np.arange(1, n + 1), n, a)


def b(s):
    """Make sure strings are correctly represented in Python 2 and 3."""
    try:
        return s.encode("utf-8")
    except AttributeError:
        return s


def doc(fdict, **kwargs):
    """Return a decorator that formats a docstring."""

    def f(fn):
        fn.__doc__ = fn.__doc__.format(**fdict)
        # kwargs is currently always empty.
        # Could remove, but keeping in case useful in future.
        for attr in kwargs:
            setattr(fn, attr, kwargs[attr])
        return fn

    return f


def convert_keyword_to_postional(keyword_name, *args, **kwargs):
    """ When complete will convert keyword_name from **kwargs to end of *args. """

    def f(fn):
        return fn

    return f


def parsedate(dstr, strftime=None, settings=None):
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

    if dstr is None:
        return dstr

    try:
        pdate = pd.to_datetime(dstr)
    except ValueError:
        pdate = dateparser.parse(dstr, settings=settings)

    if pdate is None:
        raise ValueError(
            error_wrapper(
                """
Could not parse date string '{0}'.
""".format(
                    dstr
                )
            )
        )

    if strftime is None:
        return pdate
    return pdate.strftime(strftime)


def merge_dicts(*dict_args):
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
    print("package name = {0}\npackage version = {1}".format(*namever.split()))

    print("platform architecture = {0}".format(platform.architecture()))
    print("platform machine = {0}".format(platform.machine()))
    print("platform = {0}".format(platform.platform()))
    print("platform processor = {0}".format(platform.processor()))
    print("platform python_build = {0}".format(platform.python_build()))
    print("platform python_compiler = {0}".format(platform.python_compiler()))
    print("platform python branch = {0}".format(platform.python_branch()))
    print(
        "platform python implementation = {0}".format(platform.python_implementation())
    )
    print("platform python revision = {0}".format(platform.python_revision()))
    print("platform python version = {0}".format(platform.python_version()))
    print("platform release = {0}".format(platform.release()))
    print("platform system = {0}".format(platform.system()))
    print("platform version = {0}".format(platform.version()))


def _round_index(ntsd, round_index=None):
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


def make_list(*strorlist, **kwds):
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

    if isinstance(strorlist, (list, tuple)):
        # The following will fix ((tuples, in, a, tuple, problem),)
        strorlist = list(pd.core.common.flatten(strorlist))
        if len(strorlist) == 1:
            # Normalize lists and tuples of length 1 to scalar for
            # further processing.
            strorlist = strorlist[0]

    if isinstance(strorlist, (list, tuple)):
        if n is not None:
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

    try:
        strorlist = strorlist.strip()
    except AttributeError:
        pass

    if strorlist is None or isinstance(strorlist, (type(None))):
        ### None -> None
        ###
        return None

    if isinstance(strorlist, (int, float)):
        ### 1      -> [1]
        ### 1.2    -> [1.2]
        ###
        return [strorlist]

    if isinstance(strorlist, (str, bytes)) and (strorlist in ["None", ""]):
        ### 'None' -> None
        ### ''     -> None
        ###
        return None

    if isinstance(strorlist, (str, bytes)):
        ### '1'   -> [1]
        ### '5.7' -> [5.7]

        ### Anything other than a scalar int or float continues.
        ###
        try:
            return [int(strorlist)]
        except ValueError:
            try:
                return [float(strorlist)]
            except ValueError:
                pass

    try:
        strorlist = strorlist.split(sep)
    except AttributeError:
        pass

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

    ### [1, 2, 3]          -> [1, 2, 3]
    ### ['1', '2']         -> [1, 2]

    ### [1, 'er', 5.6]     -> [1, 'er', 5.6]
    ### [1,'er',5.6]       -> [1, 'er', 5.6]
    ### ['1','er','5.6']   -> [1, 'er', 5.6]

    ### ['1','','5.6']     -> [1, None, 5.6]
    ### ['1','None','5.6'] -> [1, None, 5.6]

    ret = []
    for each in strorlist:
        if isinstance(each, (type(None), int, float)):
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


def Coerce(ntype, msg=None):
    """Coerce a value to a type.

    float:
        1     -> 1.0
        '1.1' -> 1.1
        '1,'  -> [1.0, None]
    int:
        1     -> 1
        '1'   -> 1
        '1,'  -> [1, None]
    str:
        1     -> '1'
        '1'   -> '1'
        '1,'  -> ['1', None]
    bool:
        True  -> True
        False -> False
        1     -> True
        0     -> False
        ''    -> False
        'a'   -> True
        '1,'  -> [True, False]
    """

    def f(v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            if "," in v:
                v = v.split(",")
        try:
            if isinstance(v, (list, tuple)):
                rl = []
                for i in v:
                    if i is None or i == "":
                        rl.append(i)
                    else:
                        rl.append(ntype(i))
                return rl
            return ntype(v)
        except ValueError:
            raise ValueError(msg or ("Cannot coerce {0} to {1}.".format(v, ntype)))

    return f


def _vhead(funcname, argname, nargs, nvar, vlen):
    if not isinstance(nvar, list):
        nvar = [nvar]
    if vlen is not None and len(nvar) != vlen:
        items = "item" if vlen == 1 else "items"
        raise ValueError(
            error_wrapper(
                """
The argument {argname} can only be {vlen} {items} long.

You gave {nvar}.
""".format(
                    **locals()
                )
            )
        )
    return nvar


def _vdomain(funcname, argname, nargs, nvar, vlen):
    nvar = _vhead(funcname, argname, nargs, nvar, vlen)
    for i in nvar:
        if i is None:
            continue
        if i not in nargs:
            raise ValueError(
                error_wrapper(
                    """
The argument "{argname}" should be one of the terms in {nargs}.

You gave "{i}".
""".format(
                        **locals()
                    )
                )
            )


def _vrange(funcname, argname, nargs, nvar, vlen):
    nvar = _vhead(funcname, argname, nargs, nvar, vlen)
    for i in nvar:
        if i is None:
            continue
        if nargs[0] is None:
            if i > nargs[1]:
                raise ValueError(
                    error_wrapper(
                        """
The argument "{1}" should be less than or equal to {4}.

You gave "{2}".
""".format(
                            funcname, argname, i, nargs[0], nargs[1]
                        )
                    )
                )
            continue
        if nargs[1] is None:
            if i < nargs[0]:
                raise ValueError(
                    error_wrapper(
                        """
The argument "{1}" should be greater than or equal to {3}.

You gave "{2}".
""".format(
                            funcname, argname, i, nargs[0], nargs[1]
                        )
                    )
                )
            continue
        if i < nargs[0] or i > nargs[1]:
            raise ValueError(
                error_wrapper(
                    """
The argument "{1}" should be between {3} to {4}, inclusive.

You gave "{2}".
""".format(
                        funcname, argname, i, nargs[0], nargs[1]
                    )
                )
            )


def _vpass(funcname, argname, nargs, nvar, vlen):
    pass


validator_func = {"domain": _vdomain, "range": _vrange, "pass": _vpass}


def validator(**argchecks):  # validate ranges for both+defaults
    def onDecorator(func):  # onCall remembers func and argchecks
        if not __debug__:  # True if "python -O main.py args.."
            return func  # wrap if debugging else use original
        code = func.__code__
        allargs = code.co_varnames[: code.co_argcount]
        funcname = func.__name__

        def onCall(*pargs, **kargs):
            # all pargs match first N args by position
            # the rest must be in kargs or omitted defaults
            positionals = list(allargs)
            positionals = positionals[: len(pargs)]

            for (argname, comb) in argchecks.items():
                collect_errors = []
                incomb = comb
                if callable(comb[0]):
                    incomb = [comb]
                for ctype, (valid, (nargs)), vlen in incomb:
                    # for all args to be checked
                    iffinally = True
                    if argname in kargs:
                        # was passed by name
                        cval = kargs[argname]
                    elif argname in positionals:
                        # was passed by position
                        position = positionals.index(argname)
                        cval = pargs[position]
                    else:
                        iffinally = False

                    if iffinally is True:
                        try:
                            nvar = Coerce(ctype)(cval)
                            validator_func[valid](funcname, argname, nargs, nvar, vlen)
                            collect_errors.append(None)
                            break
                        except ValueError as e:
                            collect_errors.append(str(e))
                if len(collect_errors) > 0 and all(collect_errors) is True:
                    raise ValueError("\n\n".join(collect_errors))

            return func(*pargs, **kargs)  # okay: run original call

        return onCall

    return onDecorator


def _normalize_units(ntsd, source_units, target_units):
    """

    The following is aspirational and may not reflect the code.

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

    target_units = make_list(target_units, n=len(ntsd.columns))
    source_units = make_list(source_units, n=len(ntsd.columns))

    if source_units is not None:
        names = []
        for inx in list(range(len(ntsd.columns))):
            words = ntsd.columns[inx].split(":")
            testunits = source_units[inx]
            if len(words) > 1:
                names.append(ntsd.columns[inx])
                if words[1] != testunits:
                    raise ValueError(
                        error_wrapper(
                            """
If 'source_units' specified must match units from column name.  Column
name units are specified as the second ':' delimited field.
You specified 'source_units' as {0}, but column units are {1}.
""".format(
                                source_units[inx], words[1]
                            )
                        )
                    )
            else:
                names.append("{0}:{1}".format(ntsd.columns[inx], testunits))
        ntsd.columns = names
    else:
        source_units = []
        for nu in ntsd.columns:
            try:
                source_units.append(nu.split(":")[1])
            except (AttributeError, IndexError):
                source_units.append("")

    if source_units is None and target_units is not None:
        raise ValueError(
            error_wrapper(
                """
To specify target_units, you must also specify source_units.  You can
specify source_units either by using the `source_units` keyword or placing
in the name of the data column as the second ':' separated field.

The `source_units` keyword must specify units that are convertible
to the `target_units`: {target_units}
""".format(
                    **locals()
                )
            )
        )

    # Convert source_units to target_units.
    if target_units is not None:
        ncolumns = []
        for inx, colname in enumerate(ntsd.columns):
            words = colname.split(":")
            if len(words) > 1:
                # convert words[1] to target_units[inx]
                Q_ = UREG.Quantity
                try:
                    ntsd[colname] = Q_(ntsd[colname].to_numpy(), UREG(words[1])).to(
                        target_units[inx]
                    )
                    words[1] = target_units[inx]
                except AttributeError:
                    raise ValueError(
                        error_wrapper(
                            """
No conversion between {0} and {1}.""".format(
                                words[1], target_units[inx]
                            )
                        )
                    )
            ncolumns.append(":".join(words))
        ntsd.columns = ncolumns

    return ntsd


@validator(
    start_date=[parsedate, ["pass", []], 1],
    end_date=[parsedate, ["pass", []], 1],
    force_freq=[str, ["pass", []], 1],
    groupby=[str, ["pass", []], 1],
    dropna=[str, ["domain", ["no", "any", "all"]], 1],
    round_index=[str, ["pass", []], 1],
    clean=[bool, ["domain", [True, False]], 1],
    target_units=[str, ["pass", []], None],
    source_units=[str, ["pass", []], None],
    bestfreq=[bool, ["domain", [True, False]], 1],
)
def common_kwds(
    input_tsd=None,
    start_date=None,
    end_date=None,
    pick=None,
    force_freq=None,
    groupby=None,
    dropna="no",
    round_index=None,
    clean=False,
    target_units=None,
    source_units=None,
    bestfreq=True,
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
    ntsd = input_tsd

    ntsd = _pick(ntsd, pick)

    ntsd = _normalize_units(ntsd, source_units, target_units)

    if clean is True:
        ntsd = ntsd.sort_index()
        ntsd = ntsd[~ntsd.index.duplicated()]

    ntsd = _round_index(ntsd, round_index=round_index)

    if bestfreq is True:
        ntsd = asbestfreq(ntsd, force_freq=force_freq)

    ntsd = _date_slice(ntsd, start_date=start_date, end_date=end_date)

    if ntsd.index.is_all_dates is True:
        ntsd.index.name = "Datetime"

    if dropna in ["any", "all"]:
        ntsd = ntsd.dropna(axis="index", how=dropna)
    else:
        try:
            ntsd = asbestfreq(ntsd)
        except ValueError:
            pass

    if groupby is not None:
        if groupby == "months_across_years":
            return ntsd.groupby(lambda x: x.month)
        return ntsd.resample(groupby)

    return ntsd


def _pick(tsd, columns):
    columns = make_list(columns)
    if columns is None:
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
The name {0} isn't in the list of column names
{1}.
""".format(
                        i, tsd.columns
                    )
                )
            )
        if target_col < -1:
            raise ValueError(
                error_wrapper(
                    """
The requested column "{0}" must be greater than or equal to 0.
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
The request column index {0} must be less than the
number of columns {1}.
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
            jtsd = pd.DataFrame(tsd[tsd.columns[col]], index=tsd.index)

        newtsd = newtsd.join(jtsd, lsuffix="_{0}".format(index), how="outer")
    return newtsd


def _date_slice(input_tsd, start_date=None, end_date=None):
    """Private function to slice time series."""
    if input_tsd.index.is_all_dates:
        accdate = []
        for testdate in [start_date, end_date]:
            if testdate is None:
                tdate = None
            else:
                if input_tsd.index.tz is None:
                    tdate = pd.Timestamp(testdate)
                else:
                    tdate = pd.Timestamp(testdate, tz=input_tsd.index.tz)
                # Is this comparison cheaper than the .join?
                if not np.any(input_tsd.index == tdate):
                    # Create a dummy column at the date I want, then delete
                    # Not the best, but...
                    row = pd.DataFrame([np.nan], index=[tdate])
                    row.columns = ["deleteme"]
                    input_tsd = input_tsd.join(row, how="outer")
                    input_tsd.drop("deleteme", inplace=True, axis=1)
            accdate.append(tdate)

        return input_tsd[slice(*accdate)]
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


def asbestfreq(data, force_freq=None):
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

    ndiff = (
        data.index.values.astype("int64")[1:] - data.index.values.astype("int64")[:-1]
    )
    if np.any(ndiff <= 0):
        raise ValueError(
            error_wrapper(
                """
Duplicate or time reversal index entry at
record {1} (start count at 0):
"{0}".
""".format(
                    data.index[:-1][ndiff <= 0][0], np.where(ndiff <= 0)[0][0] + 1
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
    if len(data.index) > 100:
        slic = slice(None, 99)
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
            infer_freq = "A-{0}".format(_ANNUALS[data.index[0].month])
        else:
            infer_freq = "M"
    elif np.alltrue(data.index.is_month_start):
        if np.all(data.index.month == data.index[0].month):
            # Actually yearly with different start
            infer_freq = "A-{0}".format(_ANNUALS[data.index[0].month] - 1)
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
        infer_freq = "{0}N".format(ngcd)
    elif ngcd < 1000000:
        infer_freq = "{0}U".format(ngcd // 1000)
    elif ngcd < 1000000000:
        infer_freq = "{0}L".format(ngcd // 1000000)
    elif ngcd < 60000000000:
        infer_freq = "{0}S".format(ngcd // 1000000000)
    elif ngcd < 3600000000000:
        infer_freq = "{0}T".format(ngcd // 60000000000)
    elif ngcd < 86400000000000:
        infer_freq = "{0}H".format(ngcd // 3600000000000)
    elif ngcd < 604800000000000:
        infer_freq = "{0}D".format(ngcd // 86400000000000)
    elif ngcd < 2419200000000000:
        infer_freq = "{0}W".format(ngcd // 604800000000000)
        if np.all(data.index.dayofweek == data.index[0].dayofweek):
            infer_freq = infer_freq + "-{0}".format(_WEEKLIES[data.index[0].dayofweek])
        else:
            infer_freq = "D"

    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # Give up
    return data


def dedupIndex(idx, fmt=None, ignoreFirst=True):
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


def renamer(xloc, suffix=""):
    """Print the suffix into the third ":" separated field of the header."""
    if suffix is None:
        suffix = ""
    try:
        words = xloc.split(":")
    except AttributeError:
        words = [str(xloc)]
    if len(words) == 1:
        words.append("")
        words.append(suffix)
    elif len(words) == 2:
        words.append(suffix)
    elif len(words) == 3:
        if suffix:
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
    return _printiso(
        return_input(iftrue, intds, output, suffix=suffix),
        date_format=date_format,
        float_format=float_format,
        tablefmt=tablefmt,
        showindex=showindex,
    )


def return_input(iftrue, intds, output, suffix="", reverse_index=False):
    """Print the input time series also."""
    output.columns = [renamer(i, suffix) for i in output.columns]
    if iftrue:
        return intds.join(output, lsuffix="_1", rsuffix="_2", how="outer")
    if reverse_index is True:
        return output.iloc[::-1]
    return output


def _apply_across_columns(func, xtsd, **kwds):
    """Apply a function to each column in turn."""
    for col in xtsd.columns:
        xtsd[col] = func(xtsd[col], **kwds)
    return xtsd


def _printiso(
    tsd,
    date_format=None,
    sep=",",
    float_format="g",
    showindex="never",
    headers="keys",
    tablefmt="csv",
):
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
        if tsd.index.is_all_dates is True:
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

    if tablefmt in ["csv", "tsv", "csv_nos", "tsv_nos"]:
        sep = {"csv": ",", "tsv": "\t", "csv_nos": ",", "tsv_nos": "\t"}[tablefmt]
        if isinstance(tsd, pd.DataFrame):
            try:
                tsd.to_csv(
                    sys.stdout,
                    float_format="%{0}".format(float_format),
                    date_format=date_format,
                    sep=sep,
                    index=print_index,
                )
                return
            except IOError:
                return
        else:
            tablefmt = simple_separated_format(sep)

    if tablefmt is None:
        print(str(list(tsd))[1:-1])

    all_table = tb(
        tsd,
        tablefmt=tablefmt,
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


def open_local(filein):
    """
    Open the given input file.

    It can decode various formats too, such as gzip and bz2.

    """
    ext = os.path.splitext(filein)[1]
    if ext in [".gz", ".GZ"]:
        return gzip.open(filein, "rb")
    if ext in [".bz", ".bz2"]:
        return bz2.BZ2File(filein, "rb")
    return open(filein, "r")


def memory_optimize(tsd):
    """Convert all columns to known types.

    "infer_objects" replaced some code here such that the "memory_optimize"
    function might go away.  Kept in case want to add additional
    optimizations.
    """
    tsd = tsd.infer_objects()
    return tsd


def is_valid_url(url, qualifying=None):
    """Return whether "url" is valid."""
    min_attributes = ("scheme", "netloc")
    qualifying = min_attributes if qualifying is None else qualifying
    token = urlparse(url)
    return all((getattr(token, qualifying_attr) for qualifying_attr in qualifying))


@validator(
    parse_dates=[bool, ["domain", [True, False]], 1],
    extended_columns=[bool, ["domain", [True, False]], 1],
    dropna=[str, ["domain", ["no", "any", "all"]], 1],
    force_freq=[str, ["pass", []], 1],
    index_type=[str, ["domain", ["datetime", "number"]], 1],
    names=[str, ["pass", []], 1],
    skiprows=[int, ["pass", []], 1],
)
def read_iso_ts(
    indat,
    parse_dates=True,
    extended_columns=False,
    dropna=None,
    force_freq=None,
    skiprows=None,
    index_type="datetime",
    names=None,
):
    """Read the format printed by 'printiso' and maybe other formats.

    Parameters
    ----------
    indat: str, bytes, StringIO, file pointer, file name, DataFrame,
           Series, tuple, list, dict

        The input data.

    Returns
    -------
    df: DataFrame

        Returns a DataFrame.

    """
    try:
        skiprows = int(skiprows)
    except (ValueError, TypeError):
        skiprows = make_list(skiprows)

    result = {}
    if isinstance(indat, (str, bytes, StringIO)):
        if indat in ["-", b"-"]:
            # if from stdin format must be the tstoolbox standard
            # pandas read_csv supports file like objects
            header = 0
            sep = None
            fpi = sys.stdin
            fname = "_"
        elif isinstance(indat, StringIO):
            header = "infer"
            sep = None
            fpi = indat
            fname = ""
        elif b"\n" in b(indat) or b"\r" in b(indat):
            # a string?
            header = "infer"
            sep = None
            fpi = StringIO(b(indat).decode())
            fname = ""
        elif os.path.exists(indat):
            # a local file
            header = "infer"
            sep = None
            fpi = open_local(indat)
            fname = os.path.splitext(os.path.basename(indat))[0]
        elif is_valid_url(indat):
            # a url?
            header = "infer"
            sep = None
            fpi = indat
            fname = ""
        else:
            raise ValueError(
                error_wrapper(
                    """
Can't figure out what "input_ts={0}" is.
I tested if it was a string or StringIO object, DataFrame, local file,
or an URL.  If you want to pull from stdin use "-" or redirection/piping.
""".format(
                        indat
                    )
                )
            )

        fstr = "{1}"
        if extended_columns is True:
            fstr = "{0}.{1}"

        index_col = 0
        if parse_dates is False:
            index_col = False

        # Would want this to be more generic...
        na_values = []
        for spc in range(20)[1:]:
            spcs = " " * spc
            na_values.append(spcs)
            na_values.append(spcs + "nan")

        if not result:
            if names is not None:
                header = 0
                names = make_list(names)
            if index_type == "number":
                parse_dates = False
            result = pd.io.parsers.read_csv(
                fpi,
                header=header,
                names=names,
                index_col=index_col,
                infer_datetime_format=True,
                parse_dates=parse_dates,
                na_values=na_values,
                keep_default_na=True,
                sep=sep,
                skipinitialspace=True,
                engine="python",
                skiprows=skiprows,
            )
            first = [i.split(":")[0] for i in result.columns]
            first = [fstr.format(fname, i) for i in first]
            first = [[i.strip()] for i in dedupIndex(first)]

            rest = [i.rstrip(".0123456789 ").split(":")[1:] for i in result.columns]

            result.columns = [":".join(i + j) for i, j in zip(first, rest)]

            tmpc = result.columns.values
            for index, i in enumerate(result.columns):
                if "Unnamed:" in i:
                    words = i.split(":")
                    tmpc[index] = words[0].strip() + words[1].strip()
            result.columns = tmpc

    elif isinstance(indat, pd.DataFrame):
        result = indat

    elif isinstance(indat, (pd.Series, dict)):
        result = pd.DataFrame(indat)

    elif isinstance(indat, (tuple, list)):
        result = pd.DataFrame({"values": indat})

    elif isinstance(indat, (int, float)):
        result = pd.DataFrame({"values": indat}, index=[0])

    else:
        raise ValueError(
            error_wrapper(
                """
Can't figure out what was passed to read_iso_ts.
You gave me {0}, of
{1}.
""".format(
                    indat, type(indat)
                )
            )
        )

    result = memory_optimize(result)

    if result.index.is_all_dates is False:
        try:
            result.set_index(0, inplace=True)
        except KeyError:
            pass

    if result.index.is_all_dates is True:
        try:
            words = result.index.name.split(":")
        except AttributeError:
            words = ""
        if len(words) > 1:
            try:
                result.index = result.index.tz_localize(words[1])
            except TypeError:
                pass
            result.index.name = "Datetime:{0}".format(words[1])
        else:
            result.index.name = "Datetime"

        try:
            return asbestfreq(result, force_freq=force_freq)
        except ValueError:
            return result

    if dropna in [b("any"), b("all")]:
        result.dropna(how=dropna, inplace=True)

    return result
