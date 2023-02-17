"""Collection of functions for the manipulation of time series."""

import warnings
from typing import List, Literal, Optional, Tuple, Union

from plottoolbox import (
    autocorrelation,
    bar,
    bar_stacked,
    barh,
    barh_stacked,
    bootstrap,
    boxplot,
    double_mass,
    heatmap,
    histogram,
    kde,
    kde_time,
    lag_plot,
    lognorm_xaxis,
    lognorm_yaxis,
    norm_xaxis,
    norm_yaxis,
    probability_density,
    scatter_matrix,
    target,
    taylor,
    time,
    weibull_xaxis,
    weibull_yaxis,
    xy,
)
from pydantic import PositiveInt, validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@tsutils.transform_args(
    xlim=tsutils.make_list,
    ylim=tsutils.make_list,
    legend_names=tsutils.make_list,
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
)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
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
    colors: Optional[Union[str, List[Optional[str]]]] = "auto",
    linestyles: Optional[Union[str, List[Optional[str]]]] = "auto",
    markerstyles: Optional[Union[str, List[Optional[str]]]] = " ",
    bar_hatchstyles: Optional[Union[str, List[Optional[str]]]] = "auto",
    style: Optional[Union[str, List[str]]] = "auto",
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
    bootstrap_size: PositiveInt = 50,
    bootstrap_samples: PositiveInt = 500,
    xy_match_line: str = "",
    grid: bool = False,
    label_rotation: Optional[float] = None,
    label_skip: PositiveInt = 1,
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
    lag_plot_lag: PositiveInt = 1,
    plot_styles: Union[
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
        ],
        List[
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
        ],
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
    r"""Plot data.

    Parameters
    ----------
    ${input_ts}

    ${columns}

    ${start_date}

    ${end_date}

    ${clean}

    ${skiprows}

    ${dropna}

    ${index_type}

    ${names}

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

    xaxis : str
        [optional, default is 'arithmetic']

        Defines the type of the xaxis.  One of 'arithmetic', 'log'.

    yaxis : str
        [optional, default is 'arithmetic']

        Defines the type of the yaxis.  One of 'arithmetic', 'log'.

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

    ${force_freq}

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

    invert_xaxis
        [optional, default is False]

        Invert the x-axis.

    invert_yaxis
        [optional, default is False]

        Invert the y-axis.

    ${round_index}

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

    ${source_units}

    ${target_units}

    lag_plot_lag : int, optional
        [optional, default to 1]

        The lag used if ``type`` "lag_plot" is chosen.

    plot_styles : str
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
    # Need to work around some old option defaults with the implementation of
    # mando (predecessor to cltoolbox)
    legend = legend in ("", "True", None, True)

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

    if type == "autocorrelation":
        pltr = autocorrelation(
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
        pltr = bar(
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
        pltr = bar_stacked(
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
        pltr = barh(
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
        pltr = barh_stacked(
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
        pltr = bootstrap(
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
        pltr = boxplot(
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
        pltr = double_mass(
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
        pltr = heatmap(
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
        pltr = histogram(
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
        pltr = kde(
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
        pltr = kde_time(
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
        pltr = lag_plot(
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
        pltr = lognorm_xaxis(
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
        pltr = lognorm_yaxis(
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
        pltr = norm_xaxis(
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
        pltr = norm_yaxis(
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
        pltr = probability_density(
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
        pltr = scatter_matrix(
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
        pltr = target(
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
        pltr = taylor(
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
        pltr = time(
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
            plot_styles=plot_styles,
        )
    elif type == "weibull_xaxis":
        pltr = weibull_xaxis(
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
        pltr = weibull_yaxis(
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
        pltr = xy(
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
