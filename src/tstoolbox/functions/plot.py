# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

import itertools
import os
import warnings
from typing import List, Optional, Tuple, Union

import cltoolbox
import numpy as np
import pandas as pd
import typic
from cltoolbox.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


warnings.filterwarnings("ignore")

ldocstrings = tsutils.docstrings
ldocstrings[
    "xydata"
] = """If the input 'x,y' dataset(s) is organized as
            'index,x1,y1,x2,y2,x3,y3,...,xN,yN' then the 'index' is ignored.
            If there is one 'x,y' dataset then it can be organized as 'index,y'
            where 'index' is used for 'x'.  The "columns" keyword can be used
            to duplicate or change the order of all the data columns."""
ldocstrings[
    "ydata"
] = """Data must be organized as 'index,y1,y2,y3,...,yN'.  The 'index' is
            ignored and all data columns are plotted.  The "columns" keyword
            can be used to duplicate or change the order of all the data
            columns."""
ldocstrings[
    "yone"
] = """Data must be organized as 'index,y1'.  Can only plot one series."""
ldocstrings[
    "secondary_axis"
] = """[optional, default is False]

        * list/tuple: Give the column numbers or names to plot on secondary
          y-axis.
        * (string, string): The first string is the units of the primary axis,
          the second string is the units of the secondary axis if you want just
          unit conversion.  Use any units or combination thereof from the
          "pint" library.
        * (callable, callable): Functions relating relationship between
          primary and secondary axis.  First function will be given the values
          on primary axis and returns valueis on secondary axis.  Second function
          will be do the inverse.  Python API only.
        * string: One of pre-built (callable, callable) combinations.  Can be
          one of "period"."""

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

LINE_LIST = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]

HATCH_LIST = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


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
                    f"""
Both limits must be between 0 and 1 for the 'normal', 'lognormal', or 'weibull'
axis.

Instead you have {nlim}.
"""
                )
            )

    if nlim is None:
        return nlim

    if nlim[0] is not None and nlim[1] is not None and nlim[0] >= nlim[1]:
        raise ValueError(
            tsutils.error_wrapper(
                f"""
The second limit must be greater than the first.

You gave {nlim}.
"""
            )
        )

    if (
        axis == "log"
        and (nlim[0] is not None and nlim[0] <= 0)
        or (nlim[1] is not None and nlim[1] <= 0)
    ):
        raise ValueError(
            tsutils.error_wrapper(
                f"""
If log plot cannot have limits less than or equal to 0.

You have {nlim}.
"""
            )
        )

    return nlim


@cltoolbox.command("plot", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def plot_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    dropna="no",
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
    bar_hatchstyles="auto",
    style="auto",
    logx=False,
    logy=False,
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    secondary_y=False,
    secondary_x=False,
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
    plot_styles="bright",
    hlines_y=None,
    hlines_xmin=None,
    hlines_xmax=None,
    hlines_colors="auto",
    hlines_linestyles="auto",
    vlines_x=None,
    vlines_ymin=None,
    vlines_ymax=None,
    vlines_colors="auto",
    vlines_linestyles="auto",
):
    r"""Plot data.

    Parameters
    ----------
    ${input_ts}

    ofilename : str, optional
        [optional, defaults to 'plot.png']

        Output filename for the plot.  Extension defines
        the type, for example 'filename.png' will create a PNG file.

        If used within Python, and `ofilename` is None will return the
        Matplotlib figure that can then be changed or added to as
        needed.
    type : {'time', 'xy', 'double_mass', 'bloxplot', 'scatter_matrix',
        'lag_plot', 'autocorrelation', 'bootstrap', 'histogram', 'kde',
        'kde_time', 'bar', 'barh', 'bar_stacked', 'barh_stacked',
        'heatmap', 'norm_xaxis', 'norm_yaxis', 'lognorm_xaxis',
        'lognorm_yaxis', 'weibull_xaxis', 'weibull_yaxis', 'taylor',
        'target'}, optional
        [optional, defaults to 'time']

        The plot type.

        Can be one of the following:

        time
            Standard time series plot.

            Data must be organized as 'index,y1,y2,y3,...,yN'.  The 'index'
            must be a date/time and all data columns are plotted.  Legend names
            are taken from the column names in the first row unless over-ridden
            by the `legend_names` keyword.

        xy
            An 'x,y' plot, also know as a scatter plot.

            ${xydata}
        double_mass
            An 'x,y' plot of the cumulative sum of x and y.

            ${xydata}
        boxplot
            Box extends from lower to upper quartile, with line at the
            median.  Depending on the statistics, the wiskers represent
            the range of the data or 1.5 times the inter-quartile range
            (Q3 - Q1).

            ${ydata}
        scatter_matrix
            Plots all columns against each other in a matrix, with the diagonal
            plots either histogram or KDE probability distribution
            depending on `scatter_matrix_diagonal` keyword.

            ${ydata}
        lag_plot
            Indicates structure in the data.

            ${yone}
        autocorrelation
            Plot autocorrelation.  Only available for a single time-series.

            ${yone}
        bootstrap
            Visually assess aspects of a data set by plotting random
            selections of values.  Only available for a single time-series.

            ${yone}
        histogram
            Calculate and create a histogram plot.  See 'kde' for a smooth
            representation of a histogram.
        kde
            This plot is an estimation of the probability density function
            based on the data called kernel density estimation (KDE).

            ${ydata}
        kde_time
            This plot is an estimation of the probability density function
            based on the data called kernel density estimation (KDE) combined
            with a time-series plot.

            ${ydata}
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
    lag_plot_lag : int, optional
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

        The default 'auto' will cycle through matplotlib colors in the chosen
        style.

        At the command line supply a comma separated matplotlib
        color codes, or within Python a list of color code strings.

        Can identify colors in four different ways.

        1. Use 'CN' where N is a number from 0 to 9 that gets the Nth color
        from the current style.

        2. Single character code from the table below.

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

        3. Number between 0 and 1 that represents the level of gray, where 0 is
        white an 1 is black.

        4. Any of the HTML color names.

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

        +---------+--------------+
        | Code    | Lines        |
        +=========+==============+
        | ``-``   | solid        |
        +---------+--------------+
        | --      | dashed       |
        +---------+--------------+
        | -.      | dash_dot     |
        +---------+--------------+
        | :       | dotted       |
        +---------+--------------+
        | None    | draw nothing |
        +---------+--------------+
        | ' '     | draw nothing |
        +---------+--------------+
        | ''      | draw nothing |
        +---------+--------------+

        Line reference:
        http://matplotlib.org/api/artist_api.html
    markerstyles
        [optional, default to ' ']

        The default ' ' will not plot a marker.  If 'auto' will iterate through
        the available matplotlib marker types.  Otherwise on the command line
        a comma separated list, or a list of strings if using the Python API.

        Separated 'colors', 'linestyles', and 'markerstyles' instead of using
        the 'style' keyword.

        +-------+----------------+
        | Code  | Markers        |
        +=======+================+
        | .     | point          |
        +-------+----------------+
        | o     | circle         |
        +-------+----------------+
        | v     | triangle down  |
        +-------+----------------+
        | ^     | triangle up    |
        +-------+----------------+
        | <     | triangle left  |
        +-------+----------------+
        | >     | triangle right |
        +-------+----------------+
        | 1     | tri_down       |
        +-------+----------------+
        | 2     | tri_up         |
        +-------+----------------+
        | 3     | tri_left       |
        +-------+----------------+
        | 4     | tri_right      |
        +-------+----------------+
        | 8     | octagon        |
        +-------+----------------+
        | s     | square         |
        +-------+----------------+
        | p     | pentagon       |
        +-------+----------------+
        | ``*`` | star           |
        +-------+----------------+
        | h     | hexagon1       |
        +-------+----------------+
        | H     | hexagon2       |
        +-------+----------------+
        | ``+`` | plus           |
        +-------+----------------+
        | x     | x              |
        +-------+----------------+
        | D     | diamond        |
        +-------+----------------+
        | d     | thin diamond   |
        +-------+----------------+
        | _     | hlines_y       |
        +-------+----------------+
        | None  | nothing        |
        +-------+----------------+
        | ' '   | nothing        |
        +-------+----------------+
        | ''    | nothing        |
        +-------+----------------+

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
    bar_hatchstyles
        [optional, default to "auto", only used if type equal to "bar", "barh",
        "bar_stacked", and "barh_stacked"]

        If 'auto' will iterate through the available matplotlib hatch types.
        Otherwise on the command line a comma separated list, or a list of
        strings if using the Python API.

        +-----------------+-------------------+
        | bar_hatchstyles | Description       |
        +=================+===================+
        | /               | diagonal hatching |
        +-----------------+-------------------+
        | ``\``           | back diagonal     |
        +-----------------+-------------------+
        | ``|``           | vertical          |
        +-----------------+-------------------+
        | -               | horizontal        |
        +-----------------+-------------------+
        | +               | crossed           |
        +-----------------+-------------------+
        | x               | crossed diagonal  |
        +-----------------+-------------------+
        | o               | small circle      |
        +-----------------+-------------------+
        | O               | large circle      |
        +-----------------+-------------------+
        | .               | dots              |
        +-----------------+-------------------+
        | *               | stars             |
        +-----------------+-------------------+
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
        ${secondary_axis}
    secondary_x
        ${secondary_axis}
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
    ${force_freq}
    invert_xaxis
        [optional, default is False]

        Invert the x-axis.
    invert_yaxis
        [optional, default is False]

        Invert the y-axis.
    plotting_position : str
        [optional, default is 'weibull']

        ${plotting_position_table}

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis.
    prob_plot_sort_values : str
        [optional, default is 'descending']

        How to sort the values for the probability plots.

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis.
    ${columns}
    ${start_date}
    ${end_date}
    ${clean}
    ${skiprows}
    ${dropna}
    ${index_type}
    ${names}
    ${source_units}
    ${target_units}
    ${round_index}
    plot_styles: str
        [optional, default is "default"]

        Set the style of the plot.  One or more of Matplotlib styles "classic",
        "Solarize_Light2", "bmh", "dark_background", "fast", "fivethirtyeight",
        "ggplot", "grayscale", "seaborn", "seaborn-bright",
        "seaborn-colorblind", "seaborn-dark", "seaborn-dark-palette",
        "seaborn-darkgrid", "seaborn-deep", "seaborn-muted",
        "seaborn-notebook", "seaborn-paper", "seaborn-pastel",
        "seaborn-poster", "seaborn-talk", "seaborn-ticks", "seaborn-white",
        "seaborn-whitegrid", "tableau-colorblind10", and

        SciencePlots styles "science", "grid", "ieee", "scatter", "notebook",
        "high-vis", "bright", "vibrant", "muted", and "retro".

        If multiple styles then each over rides some or all of the
        characteristics of the previous.

        Color Blind Appropriate Styles

        The styles "seaborn-colorblind", "tableau-colorblind10", "bright",
        "vibrant", and "muted" are all styles that are setup to be able to be
        distinguished by someone with color blindness.

        Black, White, and Gray Styles

        The "ieee" style is appropriate for black, white, and gray, however the
        "ieee" also will change the chart size to fit in a column of the "IEEE"
        journal.

        The "grayscale" is another style useful for photo-copyable black,
        white, nd gray.

        Matplotlib styles:
            https://matplotlib.org/3.3.1/gallery/style_sheets/style_sheets_reference.html

        SciencePlots styles:
            https://github.com/garrettj403/SciencePlots
    hlines_y:
        [optional, defaults to None]

        Number or list of y values where to place a horizontal line.
    hlines_xmin:
        [optional, defaults to None]

        List of minimum x values to start the horizontal line.  If a list must
        be same length as `hlines_y`.  If a single number will be used as the
        minimum x values for all horizontal lines.  A missing value or None
        will start at the minimum x value for the entire plot.
    hlines_xmax:
        [optional, defaults to None]

        List of maximum x values to end each horizontal line.  If a list must
        be same length as `hlines_y`.  If a single number will be the maximum
        x value for all horizontal lines.  A missing value or None will end at
        the maximum x value for the entire plot.
    hlines_colors:
        [optional, defaults to None]

        List of colors for the horizontal lines.  If a single color then will
        be used as the color for all horizontal lines.  If a list must be same
        length as `hlines_y`.  If None will take from the color pallette in the
        current plot style.
    hlines_linestyles:
        [optional, defaults to None]

        List of linestyles for the horizontal lines.  If a single linestyle
        then will be used as the linestyle for all horizontal lines.  If a list
        must be same length as `hlines_y`.  If None will take for the standard
        linestyles list.
    vlines_x:
        [optional, defaults to None]

        List of x values where to place a vertical line.
    vlines_ymin:
        [optional, defaults to None]

        List of minimum y values to start the vertical line.  If a list must be
        same length as `vlines_x`.  If a single number will be used as the
        minimum x values for all vertical lines.  A missing value or None will
        start at the minimum x value for the entire plot.
    vlines_ymax:
        [optional, defaults to None]

        List of maximum x values to end each vertical line.  If a list must be
        same length as `vlines_x`.  If a single number will be the maximum
        x value for all vertical lines.  A missing value or None will end at
        the maximum x value for the entire plot.
    vlines_colors:
        [optional, defaults to None]

        List of colors for the vertical lines.  If a single color then will be
        used as the color for all vertical lines.  If a list must be same
        length as `vlines_x`.  If None will take from the color pallette in the
        current plot style.
    vlines_linestyles:
        [optional, defaults to None]

        List of linestyles for the vertical lines.  If a single linestyle then
        will be used as the linestyle for all vertical lines.  If a list must
        be same length as `vlines_x`.  If None will take for the standard
        linestyles list.
    """
    pltr = plot(
        input_ts=input_ts,
        columns=columns,
        start_date=start_date,
        end_date=end_date,
        clean=clean,
        skiprows=skiprows,
        dropna=dropna,
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
        bar_hatchstyles=bar_hatchstyles,
        style=style,
        logx=logx,
        logy=logy,
        xaxis=xaxis,
        yaxis=yaxis,
        xlim=xlim,
        ylim=ylim,
        secondary_y=secondary_y,
        secondary_x=secondary_x,
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
        plot_styles=plot_styles,
        hlines_y=hlines_y,
        hlines_xmin=hlines_xmin,
        hlines_xmax=hlines_xmax,
        hlines_colors=hlines_colors,
        hlines_linestyles=hlines_linestyles,
        vlines_x=vlines_x,
        vlines_ymin=vlines_ymin,
        vlines_ymax=vlines_ymax,
        vlines_colors=vlines_colors,
        vlines_linestyles=vlines_linestyles,
    )
    return pltr


@tsutils.transform_args(
    xlim=tsutils.make_list,
    ylim=tsutils.make_list,
    legend_names=tsutils.make_list,
    markerstyles=tsutils.make_list,
    colors=tsutils.make_list,
    linestyles=tsutils.make_list,
    bar_hatchstyles=tsutils.make_list,
    style=tsutils.make_list,
    figsize=tsutils.make_list,
    hlines_y=tsutils.make_list,
    hlines_xmin=tsutils.make_list,
    hlines_xmax=tsutils.make_list,
    hlines_colors=tsutils.make_list,
    hlines_linestyles=tsutils.make_list,
    vlines_x=tsutils.make_list,
    vlines_ymin=tsutils.make_list,
    vlines_ymax=tsutils.make_list,
    vlines_colors=tsutils.make_list,
    vlines_linestyles=tsutils.make_list,
    plot_styles=tsutils.make_list,
)
@typic.al
@tsutils.copy_doc(plot_cli)
def plot(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    dropna="no",
    index_type="datetime",
    names=None,
    ofilename: Optional[str] = "plot.png",
    type: Literal[
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
    ] = "time",
    xtitle: str = "",
    ytitle: str = "",
    title: str = "",
    figsize: Union[Tuple[float, float], List[float], str] = "10,6.0",
    legend: Optional[bool] = None,
    legend_names: Optional[List[str]] = None,
    subplots: bool = False,
    sharex: bool = True,
    sharey: bool = False,
    colors: Optional[List[Optional[str]]] = "auto",
    linestyles: Optional[List[Optional[str]]] = "auto",
    markerstyles: Optional[List[Optional[str]]] = " ",
    bar_hatchstyles: Optional[List[Optional[str]]] = "auto",
    style: Optional[List[str]] = "auto",
    logx: bool = False,
    logy: bool = False,
    xaxis: Literal["arithmetic", "log"] = "arithmetic",
    yaxis: Literal["arithmetic", "log"] = "arithmetic",
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    secondary_y=False,
    secondary_x=False,
    mark_right: bool = True,
    scatter_matrix_diagonal: Literal["kde", "hist"] = "kde",
    bootstrap_size: tsutils.IntGreaterEqualToOne = 50,
    bootstrap_samples: tsutils.IntGreaterEqualToOne = 500,
    norm_xaxis: bool = False,
    norm_yaxis: bool = False,
    lognorm_xaxis: bool = False,
    lognorm_yaxis: bool = False,
    xy_match_line: str = "",
    grid: bool = False,
    label_rotation: Optional[float] = None,
    label_skip: tsutils.IntGreaterEqualToOne = 1,
    force_freq: Optional[str] = None,
    drawstyle: str = "default",
    por: bool = False,
    invert_xaxis: bool = False,
    invert_yaxis: bool = False,
    round_index=None,
    plotting_position: Literal[
        "weibull", "benard", "tukey", "gumbel", "hazen", "cunnane", "california"
    ] = "weibull",
    prob_plot_sort_values: Literal["ascending", "descending"] = "descending",
    source_units=None,
    target_units=None,
    lag_plot_lag: tsutils.IntGreaterEqualToOne = 1,
    plot_styles: List[
        Literal[
            "classic",
            "Solarize_Light2",
            "bmh",
            "dark_background",
            "fast",
            "fivethirtyeight",
            "ggplot",
            "grayscale",
            "seaborn",
            "seaborn-bright",
            "seaborn-colorblind",
            "seaborn-dark",
            "seaborn-dark-palette",
            "seaborn-darkgrid",
            "seaborn-deep",
            "seaborn-muted",
            "seaborn-notebook",
            "seaborn-paper",
            "seaborn-pastel",
            "seaborn-poster",
            "seaborn-talk",
            "seaborn-ticks",
            "seaborn-white",
            "seaborn-whitegrid",
            "tableau-colorblind10",
            "science",
            "grid",
            "ieee",
            "scatter",
            "notebook",
            "high-vis",
            "bright",
            "vibrant",
            "muted",
            "retro",
        ]
    ] = "bright",
    hlines_y: Optional[List[float]] = None,
    hlines_xmin: Optional[List[float]] = None,
    hlines_xmax: Optional[List[float]] = None,
    hlines_colors: List[str] = None,
    hlines_linestyles: List[Optional[str]] = "-",
    vlines_x: Optional[List[float]] = None,
    vlines_ymin: Optional[List[float]] = None,
    vlines_ymax: Optional[List[float]] = None,
    vlines_colors: List[str] = None,
    vlines_linestyles: List[Optional[str]] = "-",
):
    r"""Plot data."""
    # Need to work around some old option defaults with the implementation of
    # cltoolbox
    legend = bool(legend == "" or legend == "True" or legend is None)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from plottoolbox import plottoolbox

    tsd = tsutils.common_kwds(
        input_tsd=input_ts,
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
        por=por,
    )

    if (
        type in ["bootstrap", "heatmap", "autocorrelation", "lag_plot"]
        and len(tsd.columns) != 1
    ):
        raise ValueError(
            tsutils.error_wrapper(
                f"""
The '{type}' plot can only work with 1 time-series in the DataFrame.
The DataFrame that you supplied has {len(tsd.columns)} time-series.
"""
            )
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
        if pltfreq == None:
            short_freq = ""
        else:
            # short freq string (day) OR (2 day)
            short_freq = f"({pltfreq[beginstr:-1]})"
    except AttributeError:
        short_freq = ""

    if legend_names:
        lnames = legend_names
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
        elif type in ["xy", "double_mass"] and (
            len(tsd.columns) // 2 == len(lnames) or len(tsd.columns) == 1
        ):
            renamedict = dict(list(zip(tsd.columns[2::2], lnames[1:])))
            renamedict[tsd.columns[1]] = lnames[0]
        else:
            raise ValueError(
                tsutils.error_wrapper(
                    f"""
For 'legend_names' and most plot types you must have the same number of comma
separated names as columns in the input data.  The input data has {len(tsd.columns)} where the
number of 'legend_names' is {len(lnames)}.

If `type` is 'xy' or 'double_mass' you need to have legend names as
l1,l2,l3,...  where l1 is the legend for x1,y1, l2 is the legend for x2,y2,
...etc.
"""
                )
            )
        tsd.rename(columns=renamedict, inplace=True)
    else:
        lnames = tsd.columns

    if colors is not None and "auto" in colors:
        colors = None

    if linestyles is None:
        linestyles = " "
    elif "auto" in linestyles:
        linestyles = LINE_LIST

    if bar_hatchstyles is None:
        bar_hatchstyles = " "
    elif "auto" in bar_hatchstyles:
        bar_hatchstyles = HATCH_LIST

    if markerstyles is None:
        markerstyles = " "
    elif "auto" in markerstyles:
        markerstyles = MARKER_LIST

    if "auto" not in style:

        nstyle = style
        if len(nstyle) != len(tsd.columns):
            raise ValueError(
                tsutils.error_wrapper(
                    f"""
You have to have the same number of style strings as time-series to plot.
You supplied '{style}' for style which has {len(nstyle)} style strings,
but you have {len(tsd.columns)} time-series.
"""
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
        linestyles = [" " if i in ["  ", None] else i for i in linestyles]
    markerstyles = [" " if i is None else i for i in markerstyles]

    if colors is not None:
        icolors = itertools.cycle(colors)
    else:
        icolors = None
    imarkerstyles = itertools.cycle(markerstyles)
    ilinestyles = itertools.cycle(linestyles)

    # Only for bar, barh, bar_stacked, and barh_stacked.
    ibar_hatchstyles = itertools.cycle(bar_hatchstyles)

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
                tsutils.error_wrapper(
                    """
The --type={1} cannot also have the yaxis set to {0}.
The {0} setting for yaxis is ignored.
""".format(
                        yaxis, type
                    )
                )
            )

    xlim = _know_your_limits(xlim, axis=xaxis)
    ylim = _know_your_limits(ylim, axis=yaxis)
    plot_styles = plot_styles + ["no-latex"]
    style_loc = os.path.join(
        os.path.dirname(__file__), os.pardir, "SciencePlots_styles"
    )
    plot_styles = [
        os.path.join(style_loc, i + ".mplstyle")
        if os.path.exists(os.path.join(style_loc, i + ".mplstyle"))
        else i
        for i in plot_styles
    ]
    plt.style.use(plot_styles)

    _, ax = plt.subplots(figsize=figsize)

    if not isinstance(tsd.index, pd.DatetimeIndex) and type == "time":
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
        if tsd.shape[1] > 1 and tsd.shape[1] % 2 != 0:
            raise AttributeError(
                tsutils.error_wrapper(
                    f"""
The 'xy' and 'double_mass' types must have an even number of columns arranged
as x,y pairs or an x-index and one y data column.  You supplied {tsd.shape[1]} columns.
"""
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

    if type in [
        "xy",
        "double_mass",
        "norm_xaxis",
        "norm_yaxis",
        "lognorm_xaxis",
        "lognorm_yaxis",
        "weibull_xaxis",
        "weibull_yaxis",
    ]:
        plotdict = {
            (False, True): ax.semilogy,
            (True, False): ax.semilogx,
            (True, True): ax.loglog,
            (False, False): ax.plot,
        }

    if type == "autocorrelation":
        pltr = plottoolbox.autocorrelation(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
        )
    elif type == "bar":
        pltr = plottoolbox.bar(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            bar_hatchstyles=bar_hatchstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            label_rotation=label_rotation,
            label_skip=label_skip,
            force_freq=force_freq,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "bar_stacked":
        pltr = plottoolbox.bar_stacked(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            bar_hatchstyles=bar_hatchstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            label_rotation=label_rotation,
            label_skip=label_skip,
            force_freq=force_freq,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "barh":
        pltr = plottoolbox.barh(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            bar_hatchstyles=bar_hatchstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            label_rotation=label_rotation,
            label_skip=label_skip,
            force_freq=force_freq,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "barh_stacked":
        pltr = plottoolbox.barh_stacked(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            bar_hatchstyles=bar_hatchstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            label_rotation=label_rotation,
            label_skip=label_skip,
            force_freq=force_freq,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "bootstrap":
        pltr = plottoolbox.bootstrap(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            bootstrap_size=bootstrap_size,
            bootstrap_samples=bootstrap_samples,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "boxplot":
        pltr = plottoolbox.boxplot(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "double_mass":
        pltr = plottoolbox.double_mass(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "heatmap":
        pltr = plottoolbox.heatmap(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "histogram":
        pltr = plottoolbox.histogram(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            sharex=sharex,
            sharey=sharey,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "kde":
        pltr = plottoolbox.kde(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            secondary_y=secondary_y,
            secondary_x=secondary_x,
        )
    elif type == "kde_time":
        pltr = plottoolbox.kde_time(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            secondary_y=secondary_y,
            secondary_x=secondary_x,
        )
    elif type == "lag_plot":
        pltr = plottoolbox.lag_plot(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            lag_plot_lag=lag_plot_lag,
        )
    elif type == "lognorm_xaxis":
        pltr = plottoolbox.lognorm_xaxis(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
        )
    elif type == "lognorm_yaxis":
        pltr = plottoolbox.lognorm_yaxis(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
        )
    elif type == "norm_xaxis":
        pltr = plottoolbox.norm_xaxis(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
        )
    elif type == "norm_yaxis":
        pltr = plottoolbox.norm_yaxis(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
        )
    elif type == "probability_density":
        pltr = plottoolbox.probability_density(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            secondary_y=secondary_y,
            secondary_x=secondary_x,
        )
    elif type == "scatter_matrix":
        pltr = plottoolbox.scatter_matrix(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            scatter_matrix_diagonal=scatter_matrix_diagonal,
        )
    elif type == "target":
        pltr = plottoolbox.target(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "taylor":
        pltr = plottoolbox.taylor(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    elif type == "time":
        pltr = plottoolbox.time(
            tsd,
            legend=legend,
            subplots=subplots,
            sharex=sharex,
            sharey=sharey,
            colors=colors,
            markerstyles=markerstyles,
            linestyles=linestyles,
            logx=logx,
            logy=logy,
            xlim=xlim,
            ylim=ylim,
            mark_right=mark_right,
            figsize=figsize,
            drawstyle=drawstyle,
            secondary_y=secondary_y,
            secondary_x=secondary_x,
        )
    elif type == "weibull_xaxis":
        pltr = plottoolbox.weibull_xaxis(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
        )
    elif type == "weibull_yaxis":
        pltr = plottoolbox.weibull_yaxis(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            grid=grid,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
        )
    elif type == "xy":
        pltr = plottoolbox.xy(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            style=style,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            xy_match_line=xy_match_line,
            grid=grid,
            drawstyle=drawstyle,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
    return pltr
