#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from builtins import range

import mando
from mando.rst_text_formatter import RSTHelpFormatter

# numpy imported like this so that things like 'sin' and 'cos' can be used in
# the equation.
from numpy import *

import pandas as pd

from .. import tsutils

warnings.filterwarnings('ignore')


def _parse_equation(equation_str):
    """Private function to parse the equation used in the calculations."""
    import re
    # Get rid of spaces
    nequation = equation_str.replace(' ', '')

    # Does the equation contain any x[t]?
    tsearch = re.search(r'\[.*?t.*?\]', nequation)

    # Does the equation contain any x1, x2, ...etc.?
    nsearch = re.search(r'x[1-9][0-9]*', nequation)

    # This beasty is so users can use 't' in their equations
    # Indices of 'x' are a function of 't' and can possibly be negative or
    # greater than the length of the DataFrame.
    # Correctly (I think) handles negative indices and indices greater
    # than the length by setting to nan
    # AssertionError happens when index negative.
    # IndexError happens when index is greater than the length of the
    # DataFrame.
    # UGLY!

    # testeval is just a list of the 't' expressions in the equation.
    # for example 'x[t]+0.6*max(x[t+1],x[t-1]' would have
    # testeval = ['t', 't+1', 't-1']
    testeval = set()
    # If there is both function of t and column terms x1, x2, ...etc.
    if tsearch and nsearch:
        testeval.update(re.findall(r'x[1-9][0-9]*\[(.*?t.*?)\]',
                                   nequation))
        # replace 'x1[t+1]' with 'x.iloc[t+1,1-1]'
        # replace 'x2[t+7]' with 'x.iloc[t+7,2-1]'
        # ...etc
        nequation = re.sub(r'x([1-9][0-9]*)\[(.*?t.*?)\]',
                           r'x.iloc[\2,\1-1]',
                           nequation)
        # replace 'x1' with 'x.iloc[t,1-1]'
        # replace 'x4' with 'x.iloc[t,4-1]'
        nequation = re.sub(r'x([1-9][0-9]*)',
                           r'x.iloc[t,\1-1]',
                           nequation)
    # If there is only a function of t, i.e. x[t]
    elif tsearch:
        testeval.update(re.findall(r'x\[(.*?t.*?)\]',
                                   nequation))
        nequation = re.sub(r'x\[(.*?t.*?)\]',
                           r'xxiloc[\1,:]',
                           nequation)
        # Replace 'x' with underlying equation, but not the 'x' in a word,
        # like 'maximum'.
        nequation = re.sub(r'(?<![a-zA-Z])x(?![a-zA-Z\[])',
                           r'xxiloc[t,:]',
                           nequation)
        nequation = re.sub(r'xxiloc',
                           r'x.iloc',
                           nequation)

    elif nsearch:
        nequation = re.sub(r'x([1-9][0-9]*)',
                           r'x.iloc[:,\1-1]',
                           nequation)

    try:
        testeval.remove('t')
    except KeyError:
        pass
    return tsearch, nsearch, testeval, nequation


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def equation(equation_str,
             input_ts='-',
             columns=None,
             start_date=None,
             end_date=None,
             dropna='no',
             skiprows=None,
             index_type='datetime',
             names=None,
             clean=False,
             print_input='',
             round_index=None,
             source_units=None,
             target_units=None,
             float_format='%g'):
    """Apply <equation_str> to the time series data.

    The <equation_str> argument is a string contained in single quotes
    with 'x' used as the variable representing the input.  For example,
    '(1 - x)*sin(x)'.

    Parameters
    ----------
    equation_str : str
        String contained in single quotes that defines the equation.

        There are four different types of equations that can be used.

        +-----------------------+-----------+-------------------------+
        | Description           | Variables | Examples                |
        +=======================+===========+=========================+
        | Equation applied to   | x         | x*0.3+4-x**2            |
        | all values in the     |           | sin(x)+pi*x             |
        | dataset.  Returns     |           |                         |
        | same number of        |           |                         |
        | columns as input.     |           |                         |
        +-----------------------+-----------+-------------------------+
        | Equation used time    | x and t   | 0.6*max(x[t-1],x[t+1])  |
        | relative to current   |           |                         |
        | record.  Applies      |           |                         |
        | equation to each      |           |                         |
        | column.  Returns same |           |                         |
        | number of columns as  |           |                         |
        | input.                |           |                         |
        +-----------------------+-----------+-------------------------+
        | Equation uses values  | x1, x2,   | x1+x2                   |
        | from different        | x3, ...   |                         |
        | columns.  Returns a   | xN        |                         |
        | single column.        |           |                         |
        +-----------------------+-----------+-------------------------+
        | Equation uses values  | x1, x2,   | x1[t-1]+x2+x3[t+1]      |
        | from different        | x3,       |                         |
        | columns and values    | ...xN, t  |                         |
        | from different rows.  |           |                         |
        | Returns a single      |           |                         |
        | column.               |           |                         |
        +-----------------------+-----------+-------------------------+

        Mathematical functions in the 'np' (numpy) name space can be
        used.  Additional examples::

            'x*4 + 2',
            'x**2 + cos(x)', and
            'tan(x*pi/180)'

        are all valid <equation> strings.  The variable 't' is special
        representing the time at which 'x' occurs.  This means you can
        do things like::

            'x[t] + max(x[t-1], x[t+1])*0.6'

        to add to the current value 0.6 times the maximum adjacent
        value.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {clean}
    {print_input}
    {names}
    {float_format}
    {source_units}
    {target_units}
    {round_index}

    """
    x = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                skiprows=skiprows,
                                                names=names,
                                                index_type=index_type),
                            start_date=start_date,
                            end_date=end_date,
                            pick=columns,
                            round_index=round_index,
                            dropna=dropna,
                            source_units=source_units,
                            target_units=target_units,
                            clean=clean)

    def returnval(t, x, testeval, nequation):
        for tst in testeval:
            tvalue = eval(tst)
            if tvalue < 0 or tvalue >= len(x):
                return pd.np.nan
        return eval(nequation)

    tsearch, nsearch, testeval, nequation = _parse_equation(equation_str)
    if tsearch and nsearch:
        y = pd.DataFrame(pd.Series(index=x.index),
                         columns=['_'],
                         dtype='float64')
        for t in range(len(x)):
            y.iloc[t, 0] = returnval(t, x, testeval, nequation)
    elif tsearch:
        y = x.copy()
        for t in range(len(x)):
            y.iloc[t, :] = returnval(t, x, testeval, nequation)
    elif nsearch:
        y = pd.DataFrame(pd.Series(index=x.index),
                         columns=['_'],
                         dtype='float64')
        try:
            y.iloc[:, 0] = eval(nequation)
        except IndexError:
            raise IndexError("""
*
*   There are {0} columns, but the equation you are trying to apply is trying
*   to access a column greater than that.
*
""".format(y.shape[1]))

    else:
        y = eval(equation_str)

    y = tsutils.memory_optimize(y)

    return tsutils.print_input(print_input,
                               x,
                               y,
                               '_equation',
                               float_format=float_format)
