#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import itertools
import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import numpy as np
import pandas as pd

from .. import tsutils

warnings.filterwarnings("ignore")

MARKER_LIST = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "D",
    "d",
    "|",
    "_",
]

COLOR_LIST = ["b", "g", "r", "c", "m", "y", "k"]

LINE_LIST = ["-", "--", "-.", ":"]


def _know_your_limits(xylimits, axis="arithmetic"):
    """Establish axis limits.

    This defines the xlim and ylim as lists rather than strings.
    Might prove useful in the future in a more generic spot.  It
    normalizes the different representations.
    """
    nlim = tsutils.make_list(xylimits, n=2)

    if axis == "normal":
        if nlim is None:
            nlim = [None, None]
        if nlim[0] is None:
            nlim[0] = 0.01
        if nlim[1] is None:
            nlim[1] = 0.99
        if nlim[0] < 0 or nlim[0] > 1 or nlim[1] < 0 or nlim[1] > 1:
            raise ValueError(
                tsutils.error_wrapper(
                    """
Both limits must be between 0 and 1 for the 'normal', 'lognormal', or 'weibull'
axis.

Instead you have {0}.
""".format(
                        nlim
                    )
                )
            )

    if nlim is None:
        return nlim

    if nlim[0] is not None and nlim[1] is not None:
        if nlim[0] >= nlim[1]:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The second limit must be greater than the first.

You gave {0}.
""".format(
                        nlim
                    )
                )
            )

    if axis == "log":
        if (nlim[0] is not None and nlim[0] <= 0) or (
            nlim[1] is not None and nlim[1] <= 0
        ):
            raise ValueError(
                tsutils.error_wrapper(
                    """
If log plot cannot have limits less than or equal to 0.

You have {0}.
""".format(
                        nlim
                    )
                )
            )

    return nlim


@mando.command("plot", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def plot_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    ofilename="plot.png",
    type="time",
    xtitle="",
    ytitle="",
    title="",
    figsize="10,6.0",
    legend=None,
    legend_names=None,
    subplots=False,
    sharex=True,
    sharey=False,
    colors="auto",
    linestyles="auto",
    markerstyles=" ",
    style="auto",
    logx=False,
    logy=False,
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    secondary_y=False,
    mark_right=True,
    scatter_matrix_diagonal="kde",
    bootstrap_size=50,
    bootstrap_samples=500,
    norm_xaxis=False,
    norm_yaxis=False,
    lognorm_xaxis=False,
    lognorm_yaxis=False,
    xy_match_line="",
    grid=False,
    label_rotation=None,
    label_skip=1,
    force_freq=None,
    drawstyle="default",
    por=False,
    invert_xaxis=False,
    invert_yaxis=False,
    round_index=None,
    plotting_position="weibull",
    prob_plot_sort_values="descending",
    source_units=None,
    target_units=None,
    lag_plot_lag=1,
):
    r"""Plot data.

    Parameters
    ----------
    {input_ts}
    ofilename : str
        [optional, defaults to 'plot.png']

        Output filename for the plot.  Extension defines
        the type, for example 'filename.png' will create a PNG file.

        If used within Python, and `ofilename` is None will return the
        Matplotlib figure that can then be changed or added to as
        needed.
    type : str
        [optional, defaults to 'time']

        The plot type.

        Can be one of the following:

        time
            Standard time series plot.  Time is the index, and plots each
            column of data.
        xy
            An (x,y) plot, also know as a scatter plot.  Data must be organized
            as x1,y1,x2,y2,x3,y3,....  The "columns" keyword can be used to
            duplicate or change the column order.
        double_mass
            An (x,y) plot of the cumulative sum of x and y.  Data must be
            organized as x1,y1,x2,y2,x3,y3,....  The "columns" keyword can be
            used to duplicate or change the column order.
        boxplot
            Box extends from lower to upper quartile, with line at the
            median.  Depending on the statistics, the wiskers represent
            the range of the data or 1.5 times the inter-quartile range
            (Q3 - Q1).  Data should be organized as y1,y2,y3,....  The
            "columns" keyword can be used to duplicate or change the column
            order.
        scatter_matrix
            Plots all columns against each other in a matrix, with the diagonal
            plots either histogram or KDE probability distribution
            depending on `scatter_matrix_diagonal` keyword.
        lag_plot
            Indicates structure in the data.  Only available for a single
            time-series.
        autocorrelation
            Plot autocorrelation.  Only available for a single time-series.
        bootstrap
            Visually assess aspects of a data set by plotting random
            selections of values.  Only available for a single time-series.
        histogram
            Calculate and create a histogram plot.  See 'kde' for a smooth
            representation of a histogram.
        kde
            This plot is an estimation of the probability density function
            based on the data called kernel density estimation (KDE).
        kde_time
            This plot is an estimation of the probability density function
            based on the data called kernel density estimation (KDE) combined
            with a time-series plot.
        bar
            Column plot.
        barh
            A horizontal bar plot.
        bar_stacked
            A stacked column plot.
        barh_stacked
            A horizontal stacked bar plot.
        heatmap
            Create a 2D heatmap of daily data, day of year x-axis, and year for
            y-axis.  Only available for a single, daily time-series.
        norm_xaxis
            Sort, calculate probabilities, and plot data against an
            x axis normal distribution.
        norm_yaxis
            Sort, calculate probabilities, and plot data against an
            y axis normal distribution.
        lognorm_xaxis
            Sort, calculate probabilities, and plot data against an
            x axis lognormal distribution.
        lognorm_yaxis
            Sort, calculate probabilities, and plot data against an
            y axis lognormal distribution.
        weibull_xaxis
            Sort, calculate and plot data against an x axis weibull
            distribution.
        weibull_yaxis
            Sort, calculate and plot data against an y axis weibull
            distribution.
        taylor
            Creates a taylor diagram that compares three goodness of fit
            statistics on one plot.  The three goodness of fit statistics
            calculated and displayed are standard deviation, correlation
            coefficient, and centered root mean square deviation.  The data
            columns have to be organized as
            'observed,simulated1,simulated2,simulated3,...etc.'
        target
            Creates a target diagram that compares three goodness of fit
            statistics on one plot.  The three goodness of fit statistics
            calculated and displayed are bias, root mean square deviation, and
            centered root mean square deviation.  The data columns have to be
            organized as 'observed,simulated1,simulated2,simulated3,...etc.'
    lag_plot_lag
        [optional, default to 1]

        The lag used if ``type`` "lag_plot" is chosen.
    xtitle : str
        [optional, default depends on ``type``]

        Title of x-axis.
    ytitle : str
        [optional, default depends on ``type``]

        Title of y-axis.
    title : str
        [optional, defaults to '']

        Title of chart.
    figsize : str
        [optional, defaults to '10,6.5']

        The 'width,height' of plot in inches.
    legend
        [optional, defaults to True]

        Whether to display the legend.
    legend_names : str
        [optional, defaults to None]

        Legend would normally use the time-series names associated with
        the input data.  The 'legend_names' option allows you to
        override the names in the data set.  You must supply a comma
        separated list of strings for each time-series in the data set.
    subplots
        [optional, defaults to False]

        Make separate subplots for each time series.
    sharex
        [optional, default to True]

        In case subplots=True, share x axis.
    sharey
        [optional, default to False]

        In case subplots=True, share y axis.
    colors
        [optional, default is 'auto']

        The default 'auto' will cycle through matplotlib colors.  Otherwise at
        command line supply a comma separated matplotlib color codes, or
        within Python a list of color code strings.

        Separated 'colors', 'linestyles', and 'markerstyles' instead of using
        the 'style' keyword.

        +------+---------+
        | Code | Color   |
        +======+=========+
        | b    | blue    |
        +------+---------+
        | g    | green   |
        +------+---------+
        | r    | red     |
        +------+---------+
        | c    | cyan    |
        +------+---------+
        | m    | magenta |
        +------+---------+
        | y    | yellow  |
        +------+---------+
        | k    | black   |
        +------+---------+

        +---------+-----------+
        | Number  | Color     |
        +=========+===========+
        | 0.75    | 0.75 gray |
        +---------+-----------+
        | ...etc. |           |
        +---------+-----------+

        +------------------+
        | HTML Color Names |
        +==================+
        | red              |
        +------------------+
        | burlywood        |
        +------------------+
        | chartreuse       |
        +------------------+
        | ...etc.          |
        +------------------+

        Color reference:
        http://matplotlib.org/api/colors_api.html
    linestyles
        [optional, default to 'auto']

        If 'auto' will iterate through the available matplotlib line types.
        Otherwise on the command line a comma separated list, or a list of
        strings if using the Python API.

        To not display lines use a space (' ') as the linestyle code.

        Separated 'colors', 'linestyles', and 'markerstyles' instead of using
        the 'style' keyword.

        +------+--------------+
        | Code | Lines        |
        +======+==============+
        | \-   | solid        |
        +------+--------------+
        | --   | dashed       |
        +------+--------------+
        | -.   | dash_dot     |
        +------+--------------+
        | :    | dotted       |
        +------+--------------+
        | None | draw nothing |
        +------+--------------+
        | ' '  | draw nothing |
        +------+--------------+
        | ''   | draw nothing |
        +------+--------------+

        Line reference:
        http://matplotlib.org/api/artist_api.html
    markerstyles
        [optional, default to ' ']

        The default ' ' will not plot a marker.  If 'auto' will iterate through
        the available matplotlib marker types.  Otherwise on the command line
        a comma separated list, or a list of strings if using the Python API.

        Separated 'colors', 'linestyles', and 'markerstyles' instead of using
        the 'style' keyword.

        +------+----------------+
        | Code | Markers        |
        +======+================+
        | .    | point          |
        +------+----------------+
        | o    | circle         |
        +------+----------------+
        | v    | triangle down  |
        +------+----------------+
        | ^    | triangle up    |
        +------+----------------+
        | <    | triangle left  |
        +------+----------------+
        | >    | triangle right |
        +------+----------------+
        | 1    | tri_down       |
        +------+----------------+
        | 2    | tri_up         |
        +------+----------------+
        | 3    | tri_left       |
        +------+----------------+
        | 4    | tri_right      |
        +------+----------------+
        | 8    | octagon        |
        +------+----------------+
        | s    | square         |
        +------+----------------+
        | p    | pentagon       |
        +------+----------------+
        | \*   | star           |
        +------+----------------+
        | h    | hexagon1       |
        +------+----------------+
        | H    | hexagon2       |
        +------+----------------+
        | \+   | plus           |
        +------+----------------+
        | x    | x              |
        +------+----------------+
        | D    | diamond        |
        +------+----------------+
        | d    | thin diamond   |
        +------+----------------+
        | _    | hline          |
        +------+----------------+
        | None | nothing        |
        +------+----------------+
        | ' '  | nothing        |
        +------+----------------+
        | ''   | nothing        |
        +------+----------------+

        Marker reference:
        http://matplotlib.org/api/markers_api.html
    style
        [optional, default is None]

        Still available, but if None is replaced by 'colors', 'linestyles', and
        'markerstyles' options.  Currently the 'style' option will override the
        others.

        Comma separated matplotlib style strings per time-series.  Just
        combine codes in 'ColorMarkerLine' order, for example 'r*--' is
        a red dashed line with star marker.
    logx
        DEPRECATED: use '--xaxis="log"' instead.
    logy
        DEPRECATED: use '--yaxis="log"' instead.
    xlim
        [optional, default is based on range of x values]

        Comma separated lower and upper limits for the x-axis of the
        plot.  For example, '--xlim 1,1000' would limit the plot from
        1 to 1000, where '--xlim ,1000' would base the lower limit on
        the data and set the upper limit to 1000.
    ylim
        [optional, default is based on range of y values]

        Comma separated lower and upper limits for the y-axis of the
        plot.  See `xlim` for examples.
    xaxis : str
        [optional, default is 'arithmetic']

        Defines the type of the xaxis.  One of 'arithmetic', 'log'.
    yaxis : str
        [optional, default is 'arithmetic']

        Defines the type of the yaxis.  One of 'arithmetic', 'log'.
    secondary_y
        [optional, default is False]

        Whether to plot on the secondary y-axis. If a list/tuple, which
        time-series to plot on secondary y-axis.
    mark_right
        [optional, default is True]

        When using a secondary_y axis, should the legend label the axis of the
        various time-series automatically.
    scatter_matrix_diagonal : str
        [optional, defaults to 'kde']

        If plot type is 'scatter_matrix', this specifies the plot along the
        diagonal.  One of 'kde' for Kernel Density Estimation or 'hist'
        for a histogram.
    bootstrap_size : int
        [optional, defaults to 50]

        The size of the random subset for 'bootstrap' plot.
    bootstrap_samples
        [optional, defaults to 500]

        The number of random subsets of 'bootstrap_size'.
    norm_xaxis
        DEPRECATED: use '--type="norm_xaxis"' instead.
    norm_yaxis
        DEPRECATED: use '--type="norm_yaxis"' instead.
    lognorm_xaxis
        DEPRECATED: use '--type="lognorm_xaxis"' instead.
    lognorm_yaxis
        DEPRECATED: use '--type="lognorm_yaxis"' instead.
    xy_match_line : str
        [optional, defaults is '']

        Will add a match line where x == y. Set to a line style code.
    grid
        [optional, default is False]

        Whether to plot grid lines on the major ticks.
    label_rotation : int
        [optional]

        Rotation for major labels for bar plots.
    label_skip : int
        [optional]

        Skip for major labels for bar plots.
    drawstyle : str
        [optional, default is 'default']

        'default' connects the points with lines. The
        steps variants produce step-plots. 'steps' is equivalent to 'steps-pre'
        and is maintained for backward-compatibility.

        ACCEPTS::

         ['default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post']

    por
        [optional]

        Plot from first good value to last good value.  Strips NANs
        from beginning and end.
    {force_freq}
    invert_xaxis
        [optional, default is False]

        Invert the x-axis.
    invert_yaxis
        [optional, default is False]

        Invert the y-axis.
    plotting_position : str
        [optional, default is 'weibull']

        {plotting_position_table}

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis.
    prob_plot_sort_values : str
        [optional, default is 'descending']

        How to sort the values for the probability plots.

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis.
    {columns}
    {start_date}
    {end_date}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {round_index}

    """
    plt = plot(
        input_ts=input_ts,
        columns=columns,
        start_date=start_date,
        end_date=end_date,
        clean=clean,
        skiprows=skiprows,
        index_type=index_type,
        names=names,
        ofilename=ofilename,
        type=type,
        xtitle=xtitle,
        ytitle=ytitle,
        title=title,
        figsize=figsize,
        legend=legend,
        legend_names=legend_names,
        subplots=subplots,
        sharex=sharex,
        sharey=sharey,
        colors=colors,
        linestyles=linestyles,
        markerstyles=markerstyles,
        style=style,
        logx=logx,
        logy=logy,
        xaxis=xaxis,
        yaxis=yaxis,
        xlim=xlim,
        ylim=ylim,
        secondary_y=secondary_y,
        mark_right=mark_right,
        scatter_matrix_diagonal=scatter_matrix_diagonal,
        bootstrap_size=bootstrap_size,
        bootstrap_samples=bootstrap_samples,
        norm_xaxis=norm_xaxis,
        norm_yaxis=norm_yaxis,
        lognorm_xaxis=lognorm_xaxis,
        lognorm_yaxis=lognorm_yaxis,
        xy_match_line=xy_match_line,
        grid=grid,
        label_rotation=label_rotation,
        label_skip=label_skip,
        force_freq=force_freq,
        drawstyle=drawstyle,
        por=por,
        invert_xaxis=invert_xaxis,
        invert_yaxis=invert_yaxis,
        round_index=round_index,
        plotting_position=plotting_position,
        prob_plot_sort_values=prob_plot_sort_values,
        source_units=source_units,
        target_units=target_units,
        lag_plot_lag=lag_plot_lag,
    )


@tsutils.validator(
    ofilename=[str, ["pass", []], 1],
    type=[
        str,
        [
            "domain",
            [
                "time",
                "xy",
                "double_mass",
                "boxplot",
                "scatter_matrix",
                "lag_plot",
                "autocorrelation",
                "bootstrap",
                "histogram",
                "kde",
                "kde_time",
                "bar",
                "barh",
                "bar_stacked",
                "barh_stacked",
                "heatmap",
                "norm_xaxis",
                "norm_yaxis",
                "lognorm_yaxis",
                "lognorm_xaxis",
                "weibull_xaxis",
                "weibull_yaxis",
                "taylor",
                "target",
                "probability_density",
            ],
        ],
        1,
    ],
    lag_plot_lag=[int, ["range", [1, None]], 1],
    xtitle=[str, ["pass", []], 1],
    ytitle=[str, ["pass", []], 1],
    title=[str, ["pass", []], 1],
    figsize=[float, ["range", [0, None]], 2],
    legend=[bool, ["domain", [True, False]], 1],
    legend_names=[str, ["pass", []], 1],
    subplots=[bool, ["domain", [True, False]], 1],
    sharex=[bool, ["domain", [True, False]], 1],
    sharey=[bool, ["domain", [True, False]], 1],
    colors=[str, ["pass", []], None],
    linestyles=[str, ["pass", []], None],
    markerstyles=[str, ["pass", []], None],
    style=[str, ["pass", []], None],
    xlim=[float, ["pass", []], 2],
    ylim=[float, ["pass", []], 2],
    xaxis=[str, ["domain", ["arithmetic", "log"]], 1],
    yaxis=[str, ["domain", ["arithmetic", "log"]], 1],
    secondary_y=[bool, ["domain", [True, False]], 1],
    mark_right=[bool, ["domain", [True, False]], 1],
    scatter_matrix_diagonal=[str, ["domain", ["kde", "hist"]], 1],
    bootstrap_size=[int, ["range", [0, None]], 1],
    xy_match_line=[str, ["pass", []], 1],
    grid=[bool, ["domain", [True, False]], 1],
    label_rotation=[float, ["pass", []], 1],
    label_skip=[int, ["range", [1, None]], 1],
    drawstyle=[str, ["pass", []], 1],
    por=[bool, ["domain", [True, False]], 1],
    invert_xaxis=[bool, ["domain", [True, False]], 1],
    invert_yaxis=[bool, ["domain", [True, False]], 1],
    plotting_position=[
        str,
        [
            "domain",
            ["weibull", "benard", "tukey", "gumbel", "hazen", "cunnane", "california"],
        ],
        1,
    ],
    prob_plot_sort_values=[str, ["domain", ["ascending", "descending"]], 1],
)
def plot(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    ofilename="plot.png",
    type="time",
    xtitle="",
    ytitle="",
    title="",
    figsize="10,6.0",
    legend=None,
    legend_names=None,
    subplots=False,
    sharex=True,
    sharey=False,
    colors="auto",
    linestyles="auto",
    markerstyles=" ",
    style="auto",
    logx=False,
    logy=False,
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    secondary_y=False,
    mark_right=True,
    scatter_matrix_diagonal="kde",
    bootstrap_size=50,
    bootstrap_samples=500,
    norm_xaxis=False,
    norm_yaxis=False,
    lognorm_xaxis=False,
    lognorm_yaxis=False,
    xy_match_line="",
    grid=False,
    label_rotation=None,
    label_skip=1,
    force_freq=None,
    drawstyle="default",
    por=False,
    invert_xaxis=False,
    invert_yaxis=False,
    round_index=None,
    plotting_position="weibull",
    prob_plot_sort_values="descending",
    source_units=None,
    target_units=None,
    lag_plot_lag=1,
):
    r"""Plot data."""
    # Need to work around some old option defaults with the implementation of
    # mando
    legend = bool(legend == "" or legend == "True" or legend is None)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna="all",
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    if type in ["bootstrap", "heatmap", "autocorrelation", "lag_plot"]:
        if len(tsd.columns) != 1:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The '{1}' plot can only work with 1 time-series in the DataFrame.
The DataFrame that you supplied has {0} time-series.
""".format(
                        len(tsd.columns), type
                    )
                )
            )

    if por is True:
        tsd = tsutils.common_kwds(
            tsutils.read_iso_ts(tsd),
            start_date=start_date,
            end_date=end_date,
            round_index=round_index,
            dropna="no",
        )

    # This is to help pretty print the frequency
    try:
        try:
            pltfreq = str(tsd.index.freq, "utf-8").lower()
        except TypeError:
            pltfreq = str(tsd.index.freq).lower()
        if pltfreq.split(" ")[0][1:] == "1":
            beginstr = 3
        else:
            beginstr = 1
        if pltfreq == "none":
            short_freq = ""
        else:
            # short freq string (day) OR (2 day)
            short_freq = "({0})".format(pltfreq[beginstr:-1])
    except AttributeError:
        short_freq = ""

    if legend_names:
        lnames = tsutils.make_list(legend_names)
        if len(lnames) != len(set(lnames)):
            raise ValueError(
                tsutils.error_wrapper(
                    """
Each name in legend_names must be unique.
"""
                )
            )
        if len(tsd.columns) == len(lnames):
            renamedict = dict(list(zip(tsd.columns, lnames)))
        elif type == "xy" and len(tsd.columns) // 2 == len(lnames):
            renamedict = dict(list(zip(tsd.columns[2::2], lnames[1:])))
            renamedict[tsd.columns[1]] = lnames[0]
        else:
            raise ValueError(
                tsutils.error_wrapper(
                    """
For 'legend_names' you must have the same number of comma
separated names as columns in the input data.  The input
data has {0} where the number of 'legend_names' is {1}.

If 'xy' type you need to have legend names as x,y1,y2,y3,...
""".format(
                        len(tsd.columns), len(lnames)
                    )
                )
            )
        tsd.rename(columns=renamedict, inplace=True)
    else:
        lnames = tsd.columns

    if colors == "auto":
        colors = COLOR_LIST
    else:
        colors = tsutils.make_list(colors)

    if linestyles == "auto":
        linestyles = LINE_LIST
    else:
        linestyles = tsutils.make_list(linestyles)

    if markerstyles == "auto":
        markerstyles = MARKER_LIST
    else:
        markerstyles = tsutils.make_list(markerstyles)
        if markerstyles is None:
            markerstyles = " "

    if style != "auto":

        nstyle = tsutils.make_list(style)
        if len(nstyle) != len(tsd.columns):
            raise ValueError(
                tsutils.error_wrapper(
                    """
You have to have the same number of style strings as time-series to plot.
You supplied '{0}' for style which has {1} style strings,
but you have {2} time-series.
""".format(
                        style, len(nstyle), len(tsd.columns)
                    )
                )
            )
        colors = []
        markerstyles = []
        linestyles = []
        for st in nstyle:
            colors.append(st[0])
            if len(st) == 1:
                markerstyles.append(" ")
                linestyles.append("-")
                continue
            if st[1] in MARKER_LIST:
                markerstyles.append(st[1])
                try:
                    linestyles.append(st[2:])
                except IndexError:
                    linestyles.append(" ")
            else:
                markerstyles.append(" ")
                linestyles.append(st[1:])
    if linestyles is None:
        linestyles = [" "]
    else:
        linestyles = [" " if i == "  " else i for i in linestyles]
    markerstyles = [" " if i is None else i for i in markerstyles]

    icolors = itertools.cycle(colors)
    imarkerstyles = itertools.cycle(markerstyles)
    ilinestyles = itertools.cycle(linestyles)

    style = [
        "{0}{1}{2}".format(next(icolors), next(imarkerstyles), next(ilinestyles))
        for i in list(range(len(tsd.columns)))
    ]

    # reset to beginning of iterator
    icolors = itertools.cycle(colors)
    imarkerstyles = itertools.cycle(markerstyles)
    ilinestyles = itertools.cycle(linestyles)

    if (
        logx is True
        or logy is True
        or norm_xaxis is True
        or norm_yaxis is True
        or lognorm_xaxis is True
        or lognorm_yaxis is True
    ):
        warnings.warn(
            """
*
*   The --logx, --logy, --norm_xaxis, --norm_yaxis, --lognorm_xaxis, and
*   --lognorm_yaxis options are deprecated.
*
*   For --logx use --xaxis="log"
*   For --logy use --yaxis="log"
*   For --norm_xaxis use --type="norm_xaxis"
*   For --norm_yaxis use --type="norm_yaxis"
*   For --lognorm_xaxis use --type="lognorm_xaxis"
*   For --lognorm_yaxis use --type="lognorm_yaxis"
*
"""
        )

    if xaxis == "log":
        logx = True
    if yaxis == "log":
        logy = True

    if type in ["norm_xaxis", "lognorm_xaxis", "weibull_xaxis"]:
        xaxis = "normal"
        if logx is True:
            logx = False
            warnings.warn(
                """
*
*   The --type={1} cannot also have the xaxis set to {0}.
*   The {0} setting for xaxis is ignored.
*
""".format(
                    xaxis, type
                )
            )

    if type in ["norm_yaxis", "lognorm_yaxis", "weibull_yaxis"]:
        yaxis = "normal"
        if logy is True:
            logy = False
            warnings.warn(
                """
*
*   The --type={1} cannot also have the yaxis set to {0}.
*   The {0} setting for yaxis is ignored.
*
""".format(
                    yaxis, type
                )
            )

    xlim = _know_your_limits(xlim, axis=xaxis)
    ylim = _know_your_limits(ylim, axis=yaxis)

    figsize = tsutils.make_list(figsize, n=2)

    if not isinstance(tsd.index, pd.DatetimeIndex):
        if type == "time":
            raise ValueError(
                tsutils.error_wrapper(
                    """
The index is not a datetime index and cannot be plotted as a time-series.
Instead of `type="time"` you might want `type="xy"` or change the index to
a datetime index.
"""
                )
            )

    if type in ["xy", "double_mass"]:
        if tsd.shape[1] % 2 != 0:
            raise AttributeError(
                tsutils.error_wrapper(
                    """
The 'xy' and 'double_mass' types must have an even number of columns arranged
as x,y pairs.  You supplied {0} columns.
""".format(
                        tsd.shape[1]
                    )
                )
            )
        colcnt = tsd.shape[1] // 2
    elif type in [
        "norm_xaxis",
        "norm_yaxis",
        "lognorm_xaxis",
        "lognorm_yaxis",
        "weibull_xaxis",
        "weibull_yaxis",
    ]:
        colcnt = tsd.shape[1]

    _, ax = plt.subplots(figsize=figsize)
    if type in [
        "xy",
        "double_mass",
        "norm_xaxis",
        "norm_yaxis",
        "lognorm_xaxis",
        "lognorm_yaxis",
        "weibull_xaxis",
        "weibull_yaxis",
        "heatmap",
    ]:
        plotdict = {
            (False, True): ax.semilogy,
            (True, False): ax.semilogx,
            (True, True): ax.loglog,
            (False, False): ax.plot,
        }

    if type == "time":
        ax = tsd.plot(
            legend=legend,
            subplots=subplots,
            sharex=sharex,
            sharey=sharey,
            style=None,
            logx=logx,
            logy=logy,
            xlim=xlim,
            ylim=ylim,
            secondary_y=secondary_y,
            mark_right=mark_right,
            figsize=figsize,
            drawstyle=drawstyle,
        )
        for index, line in enumerate(ax.lines):
            plt.setp(line, color=style[index][0])
            plt.setp(line, marker=style[index][1])
            plt.setp(line, linestyle=style[index][2:])
        xtitle = xtitle or "Time"
        if legend is True:
            plt.legend(loc="best")
    elif type in ["taylor"]:
        from ..skill_metrics import centered_rms_dev
        from ..skill_metrics import taylor_diagram

        ref = tsd.iloc[:, 0]
        std = [np.std(ref)]
        ccoef = [1.0]
        crmsd = [0.0]
        for col in range(1, len(tsd.columns)):
            std.append(np.std(tsd.iloc[:, col]))
            ccoef.append(np.corrcoef(tsd.iloc[:, col], ref)[0][1])
            crmsd.append(centered_rms_dev(tsd.iloc[:, col].values, ref.values))
        taylor_diagram(np.array(std), np.array(crmsd), np.array(ccoef))
    elif type in ["target"]:
        from ..skill_metrics import centered_rms_dev
        from ..skill_metrics import rmsd
        from ..skill_metrics import bias
        from ..skill_metrics import target_diagram

        biases = []
        rmsds = []
        crmsds = []
        ref = tsd.iloc[:, 0].values
        for col in range(1, len(tsd.columns)):
            biases.append(bias(tsd.iloc[:, col].values, ref))
            crmsds.append(centered_rms_dev(tsd.iloc[:, col].values, ref))
            rmsds.append(rmsd(tsd.iloc[:, col].values, ref))
        target_diagram(np.array(biases), np.array(crmsds), np.array(rmsds))
    elif type in ["xy", "double_mass"]:
        # PANDAS was not doing the right thing with xy plots
        # if you wanted lines between markers.
        # Fell back to using raw matplotlib.
        # Boy I do not like matplotlib.

        for colindex in range(colcnt):
            ndf = tsd.iloc[:, colindex * 2 : colindex * 2 + 2]
            if type == "double_mass":
                ndf = ndf.dropna().cumsum()
            oxdata = np.array(ndf.iloc[:, 0])
            oydata = np.array(ndf.iloc[:, 1])

            plotdict[(logx, logy)](
                oxdata,
                oydata,
                linestyle=next(ilinestyles),
                color=next(icolors),
                marker=next(imarkerstyles),
                label=lnames[colindex],
                drawstyle=drawstyle,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend is True:
            ax.legend(loc="best")

        if type == "double_mass":
            xtitle = xtitle or "Cumulative {0}".format(tsd.columns[0])
            ytitle = ytitle or "Cumulative {0}".format(tsd.columns[1])

    elif type in [
        "norm_xaxis",
        "norm_yaxis",
        "lognorm_xaxis",
        "lognorm_yaxis",
        "weibull_xaxis",
        "weibull_yaxis",
    ]:
        ppf = tsutils.set_ppf(type.split("_")[0])
        ys = tsd.iloc[:, :]

        for colindex in range(colcnt):
            oydata = np.array(ys.iloc[:, colindex].dropna())
            if prob_plot_sort_values == "ascending":
                oydata = np.sort(oydata)
            elif prob_plot_sort_values == "descending":
                oydata = np.sort(oydata)[::-1]
            n = len(oydata)
            norm_axis = ax.xaxis
            oxdata = ppf(tsutils.set_plotting_position(n, plotting_position))
            if type in ["norm_yaxis", "lognorm_yaxis", "weibull_yaxis"]:
                oxdata, oydata = oydata, oxdata
                norm_axis = ax.yaxis

            plotdict[(logx, logy)](
                oxdata,
                oydata,
                linestyle=next(ilinestyles),
                color=next(icolors),
                marker=next(imarkerstyles),
                label=lnames[colindex],
                drawstyle=drawstyle,
            )

        # Make it pretty
        xtmaj = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        xtmaj_str = ["1", "10", "50", "90", "99"]
        xtmin = np.concatenate(
            [
                np.linspace(0.001, 0.01, 10),
                np.linspace(0.01, 0.1, 10),
                np.linspace(0.1, 0.9, 9),
                np.linspace(0.9, 0.99, 10),
                np.linspace(0.99, 0.999, 10),
            ]
        )
        xtmaj = ppf(xtmaj)
        xtmin = ppf(xtmin)

        norm_axis.set_major_locator(FixedLocator(xtmaj))
        norm_axis.set_minor_locator(FixedLocator(xtmin))

        if type in ["norm_xaxis", "lognorm_xaxis", "weibull_xaxis"]:
            ax.set_xticklabels(xtmaj_str)
            ax.set_ylim(ylim)
            ax.set_xlim(ppf(xlim))

        elif type in ["norm_yaxis", "lognorm_yaxis", "weibull_yaxis"]:
            ax.set_yticklabels(xtmaj_str)
            ax.set_xlim(xlim)
            ax.set_ylim(ppf(ylim))

        if type in ["norm_xaxis", "norm_yaxis"]:
            xtitle = xtitle or "Normal Distribution"
            ytitle = ytitle or tsd.columns[0]
        elif type in ["lognorm_xaxis", "lognorm_yaxis"]:
            xtitle = xtitle or "Log Normal Distribution"
            ytitle = ytitle or tsd.columns[0]
        elif type in ["weibull_xaxis", "weibull_yaxis"]:
            xtitle = xtitle or "Weibull Distribution"
            ytitle = ytitle or tsd.columns[0]

        if type in ["norm_yaxis", "lognorm_yaxis", "weibull_yaxis"]:
            xtitle, ytitle = ytitle, xtitle

        if legend is True:
            ax.legend(loc="best")

    elif type in ["kde", "probability_density"]:
        ax = tsd.plot(
            kind="kde",
            legend=legend,
            subplots=subplots,
            sharex=sharex,
            sharey=sharey,
            style=None,
            logx=logx,
            logy=logy,
            xlim=xlim,
            ylim=ylim,
            secondary_y=secondary_y,
            figsize=figsize,
        )
        for index, line in enumerate(ax.lines):
            plt.setp(line, color=style[index][0])
            plt.setp(line, marker=style[index][1])
            plt.setp(line, linestyle=style[index][2:])
        ytitle = ytitle or "Density"
        if legend is True:
            plt.legend(loc="best")
    elif type == "kde_time":
        from scipy.stats.kde import gaussian_kde

        _, (ax0, ax1) = plt.subplots(
            nrows=1,
            ncols=2,
            sharey=True,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 4]},
        )
        tsd.plot(
            legend=legend,
            subplots=subplots,
            sharex=sharex,
            sharey=sharey,
            style=None,
            logx=logx,
            logy=logy,
            xlim=xlim,
            ylim=ylim,
            secondary_y=secondary_y,
            mark_right=mark_right,
            figsize=figsize,
            drawstyle=drawstyle,
            ax=ax1,
        )
        for index, line in enumerate(ax1.lines):
            plt.setp(line, color=style[index][0])
            plt.setp(line, marker=style[index][1])
            plt.setp(line, linestyle=style[index][2:])
        xtitle = xtitle or "Time"
        ylimits = ax1.get_ylim()
        ny = np.linspace(ylimits[0], ylimits[1], 1000)
        for col in range(len(tsd.columns)):
            xvals = tsd.iloc[:, col].dropna().values
            pdf = gaussian_kde(xvals)
            ax0.plot(
                pdf(ny),
                ny,
                linestyle=style[col][2:],
                color=style[col][0],
                marker=style[col][1],
                label=tsd.columns[col],
                drawstyle=drawstyle,
            )
        ax0.set(xlabel="Probability Density", ylabel=ytitle)
    elif type == "boxplot":
        tsd.boxplot(figsize=figsize)
    elif type == "scatter_matrix":
        from pandas.plotting import scatter_matrix

        if scatter_matrix_diagonal == "probablity_density":
            scatter_matrix_diagonal = "kde"
        scatter_matrix(tsd, diagonal=scatter_matrix_diagonal, figsize=figsize)
    elif type == "lag_plot":
        from pandas.plotting import lag_plot

        lag_plot(tsd, lag=lag_plot_lag, ax=ax)
        xtitle = xtitle or "y(t)"
        ytitle = ytitle or "y(t+{0})".format(short_freq or 1)
    elif type == "autocorrelation":
        from pandas.plotting import autocorrelation_plot

        autocorrelation_plot(tsd, ax=ax)
        xtitle = xtitle or "Time Lag {0}".format(short_freq)
    elif type == "bootstrap":
        from pandas.plotting import bootstrap_plot

        bootstrap_plot(
            tsd, size=bootstrap_size, samples=bootstrap_samples, color="gray"
        )
    elif type == "heatmap":
        # Find beginning and end years
        byear = tsd.index[0].year
        eyear = tsd.index[-1].year
        tsd = tsutils.asbestfreq(tsd)
        if tsd.index.freqstr != "D":
            raise ValueError(
                tsutils.error_wrapper(
                    """
The "heatmap" plot type can only work with daily time series.
"""
                )
            )
        dr = pd.date_range(
            "{0}-01-01".format(byear), "{0}-12-31".format(eyear), freq="D"
        )
        ntsd = tsd.reindex(index=dr)
        groups = ntsd.iloc[:, 0].groupby(pd.Grouper(freq="A"))
        years = pd.DataFrame()
        for name, group in groups:
            ngroup = group.values
            if len(group.values) == 365:
                ngroup = np.append(group.values, [np.nan])
            years[name.year] = ngroup
        years = years.T
        plt.imshow(years, interpolation=None, aspect="auto")
        plt.colorbar()
        yticks = list(range(byear, eyear + 1))
        skip = len(yticks) // 20 + 1
        plt.yticks(range(0, len(yticks), skip), yticks[::skip])
        mnths = [0, 30, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        mnths_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        plt.xticks(mnths, mnths_labels)
        grid = False
    elif type in ("bar", "bar_stacked", "barh", "barh_stacked"):
        stacked = False
        if type[-7:] == "stacked":
            stacked = True
        kind = "bar"
        if type[:4] == "barh":
            kind = "barh"
        ax = tsd.plot(
            kind=kind,
            legend=legend,
            stacked=stacked,
            style=style,
            logx=logx,
            logy=logy,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
        )
        for index, line in enumerate(ax.lines):
            plt.setp(line, color=style[index][0])
            plt.setp(line, marker=style[index][1])
            plt.setp(line, linestyle=style[index][2:])
        freq = tsutils.asbestfreq(tsd, force_freq=force_freq).index.freqstr
        if freq is not None:
            if "A" in freq:
                endchar = 4
            elif "M" in freq:
                endchar = 7
            elif "D" in freq:
                endchar = 10
            elif "H" in freq:
                endchar = 13
            else:
                endchar = None
            nticklabels = []
            if kind == "bar":
                taxis = ax.xaxis
            else:
                taxis = ax.yaxis
            for index, i in enumerate(taxis.get_majorticklabels()):
                if index % label_skip:
                    nticklabels.append(" ")
                else:
                    nticklabels.append(i.get_text()[:endchar])
            taxis.set_ticklabels(nticklabels)
            plt.setp(taxis.get_majorticklabels(), rotation=label_rotation)
        if legend is True:
            plt.legend(loc="best")
    elif type == "histogram":
        tsd.hist(figsize=figsize)
    if xy_match_line:
        if isinstance(xy_match_line, str):
            xymsty = xy_match_line
        else:
            xymsty = "g--"
        nxlim = ax.get_xlim()
        nylim = ax.get_ylim()
        maxt = max(nxlim[1], nylim[1])
        mint = min(nxlim[0], nylim[0])
        ax.plot([mint, maxt], [mint, maxt], xymsty, zorder=1)
        ax.set_ylim(nylim)
        ax.set_xlim(nxlim)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if invert_xaxis is True:
        plt.gca().invert_xaxis()
    if invert_yaxis is True:
        plt.gca().invert_yaxis()

    plt.grid(grid)

    plt.title(title)
    plt.tight_layout()
    if ofilename is not None:
        plt.savefig(ofilename)
    return plt


plot.__doc__ = plot_cli.__doc__
