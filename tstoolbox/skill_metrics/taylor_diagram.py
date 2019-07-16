from array import array

import numpy as np

from . import (
    check_taylor_stats,
    get_taylor_diagram_axes,
    get_taylor_diagram_options,
    overlay_taylor_diagram_circles,
    overlay_taylor_diagram_lines,
    plot_pattern_diagram_colorbar,
    plot_pattern_diagram_markers,
    plot_taylor_axes,
    plot_taylor_obs,
)


def taylor_diagram(*args, **kwargs):
    """
    Plot a Taylor diagram from statistics of different series.

    taylor_diagram(stds,rmss,cors,keyword=value)

    The first 3 arguments must be the inputs as described below followed by
    keywords in the format OPTION = value. An example call to the function
    would be:

    taylor_diagram(stds,rmss,cors,markerdisplayed='marker')

    INPUTS:
    stds: Standard deviations
    rmss: Centered Root Mean Square Difference
    cors: Correlation

    Each of these inputs are one-dimensional with the same length. First
    index corresponds to the reference series for the diagram. For
    example stds[1] is the standard deviation of the reference series
    and stds[2:N] are the standard deviations of the other series. Note
    that only the latter are plotted.

    Note that by definition the following relation must be true for all
    series i:

     rmss(i) = sqrt(stds(i).^2 + stds(1)^2 - 2*stds(i)*stds(1).*cors(i))

    This relation is checked if the checkStats option is used, and if not
    verified an error message is sent. This relation is not checked by
    default. Please see Taylor's JGR article for more informations about
    this relation.

    OUTPUTS:
    None.

    LIST OF OPTIONS:
    For an exhaustive list of options to customize your diagram, call the
    function without arguments at a Python command line:
    % python
    >> from tstoolbox.skill_metrics.taylor_diagram import taylor_diagram
    >> taylor_diagram()

    Reference:

    Taylor, K. E. (2001), Summarizing multiple aspects of model
      performance in a single diagram, J. Geophys. Res., 106(D7),
      7183-7192, doi:10.1029/2000JD900719.

    Author: Peter A. Rochford
            Symplectic, LLC
            www.thesymplectic.com
            prochford@thesymplectic.com

    Created on Dec 3, 2016
    """
    # Check for number of arguments
    nargin = len(args)
    stds, rmss, cors = _get_taylor_diagram_arguments(*args, **kwargs)
    if nargin == 0:
        return

    # Get options
    option = get_taylor_diagram_options(cors, **kwargs)

    # Check the input statistics if requested.
    if option["checkstats"] == "on":
        check_taylor_stats(stds[1:], rmss[1:], cors[1:], 0.01)

    # Express statistics in polar coordinates.
    rho = stds
    theta = np.arccos(cors)

    #  Get axis values for plot
    axes, cax = get_taylor_diagram_axes(rho, option)

    # Plot axes for target diagram
    if option["overlay"] == "off":
        # Draw circles about origin
        overlay_taylor_diagram_circles(axes, cax, option)

        # Draw lines emanating from origin
        overlay_taylor_diagram_lines(axes, cax, option)

        # Plot axes for Taylor diagram
        ax = plot_taylor_axes(axes, cax, option)

        # Plot marker on axis indicating observation STD
        plot_taylor_obs(ax, stds[0], axes, option)

    # Plot data points. Note that only rho[1:N] and theta[1:N] are
    # plotted.
    x = np.multiply(rho[1:], np.cos(theta[1:]))
    y = np.multiply(rho[1:], np.sin(theta[1:]))

    lowcase = option["markerdisplayed"].lower()
    if lowcase == "marker":
        plot_pattern_diagram_markers(x, y, option)
    elif lowcase == "colorbar":
        plot_pattern_diagram_colorbar(x, y, rmss[1:], option)
    else:
        raise ValueError("Unrecognized option: option['markerdisplayed']")


def _get_taylor_diagram_arguments(*args, **kwargs):
    """
    Get arguments for taylor_diagram function.

    Retrieves the arguments supplied to the TAYLOR_DIAGRAM function as
    arguments and displays the optional arguments if none are supplied.
    Otherwise, tests the first 3 arguments are numeric quantities and
    returns their values.

    INPUTS:
    args : variable-length input argument list

    OUTPUTS:
    stds: Standard deviations
    rmss: Centered Root Mean Square Difference
    cors: Correlation
    """
    import numbers

    stds = []
    rmss = []
    cors = []

    nargin = len(args)
    if nargin == 0:
        # Display options list
        _display_taylor_diagram_options()
        return stds, rmss, cors
    elif nargin != 3:
        raise ValueError("Must supply 3 arguments.")

    stds = args[0]
    rmss = args[1]
    cors = args[2]

    # Test the above are numeric quantities
    if isinstance(stds, array):
        stds = np.array(stds)
    if isinstance(stds, numbers.Number):
        stds = np.array(stds, ndmin=1)
    if not isinstance(stds, np.ndarray):
        raise ValueError("Argument stds is not a numeric array")

    if isinstance(rmss, array):
        rmss = np.array(rmss)
    if isinstance(rmss, numbers.Number):
        rmss = np.array(rmss, ndmin=1)
    if not isinstance(rmss, np.ndarray):
        raise ValueError("Argument rmss is not a numeric array")

    if isinstance(cors, array):
        cors = np.array(cors)
    if isinstance(cors, numbers.Number):
        cors = np.array(cors, ndmin=1)
    if not isinstance(cors, (np.ndarray, array)):
        raise ValueError("Argument cors is not a numeric array")

    return stds, rmss, cors


def _display_taylor_diagram_options():
    """Display available options for TAYLOR_DIAGRAM function."""
    _disp("General options:")
    _dispopt(
        "'numberPanels'",
        """1 or 2: Panels to display
(1 for positive correlations, 2 for positive and negative correlations).
    Default value depends on correlations (cors)""",
    )
    _dispopt(
        "'overlay'",
        """'on' / 'off' (default):
        Switch to overlay current statistics on Taylor diagram.
        Only markers will be displayed.""",
    )
    _dispopt(
        "'alpha'",
        """Blending of symbol face color (0.0 transparent through 1.0 opaque)
(Default: 1.0)""",
    )
    _dispopt("'axismax'", "Maximum for the radial contours")
    _dispopt(
        "'colormap'",
        """'on'/ 'off' (default):
        Switch to map color shading of markers to colormap ('on')
        or min to max range of RMSDz values ('off').
        Set to same value as option['nonRMSDz'].""",
    )
    _disp("")

    _disp("Marker options:")
    _dispopt(
        "'MarkerDisplayed'",
        """'marker' (default):
        Experiments are represented by individual symbols
        "'colorBar':
             Experiments are represented by a color described in a colorbar""",
    )
    _disp("OPTIONS when 'MarkerDisplayed' == 'marker'")
    _dispopt("'markerLabel'", "Labels for markers")
    _dispopt("'markerLabelColor'", "Marker label color (Default: black)")
    _dispopt("'markerColor'", "Single color to use for all markers (Default: red)")
    _dispopt("'markerLegend'", "'on' / 'off' (default): Use legend for markers")
    _dispopt("'markerSize'", "Marker size (Default: 10)")

    _disp("OPTIONS when MarkerDisplayed' == 'colorbar'")
    _dispopt(
        "'nonRMSDz'",
        """'on'/ 'off' (default):
        Values in RMSDz do not correspond to total RMS Differences.
        (Used to make range of RMSDz values appear above color bar.)""",
    )
    _dispopt("'titleColorBar'", "Title of the colorbar.")
    _disp("")

    _disp("RMS axis options:")
    _dispopt("'tickRMS'", "RMS values to plot grid circles from observation point")
    _dispopt("'rincRMS'", "axis tick increment for RMS values")
    _dispopt("'colRMS'", "RMS grid and tick labels color. (Default: green)")
    _dispopt("'showlabelsRMS'", "'on' (default) / 'off': Show the RMS tick labels")
    _dispopt(
        "'tickRMSangle'",
        "Angle for RMS tick labels with the observation point. Default: 135 deg.",
    )
    _dispopt(
        "'rmsLabelFormat'",
        """String format for RMS contour labels, e.g. '0:.2f'.
        (Default '0', format as specified by str function.)""",
    )
    _dispopt("'styleRMS'", "Line style of the RMS grid")
    _dispopt("'widthRMS'", "Line width of the RMS grid")
    _dispopt("'titleRMS'", "'on' (default) / 'off': Show RMSD axis title")
    _disp("")

    _disp("STD axis options:")
    _dispopt("'tickSTD'", "STD values to plot gridding circles from origin")
    _dispopt("'rincSTD'", "axis tick increment for STD values")
    _dispopt("'colSTD'", "STD grid and tick labels color. (Default: black)")
    _dispopt("'showlabelsSTD'", "'on' (default) / 'off': Show the STD tick labels")
    _dispopt("'styleSTD'", "Line style of the STD grid")
    _dispopt("'widthSTD'", "Line width of the STD grid")
    _dispopt("'titleSTD'", "'on' (default) / 'off': Show STD axis title")
    _disp("")

    _disp("CORRELATION axis options:")
    _dispopt("'tickCOR'", "CORRELATON grid values")
    _dispopt("'colCOR'", "CORRELATION grid color. Default: blue")
    _dispopt(
        "'showlabelsCOR'", "'on' (default) / 'off': Show the CORRELATION tick labels"
    )
    _dispopt("'styleCOR'", "Line style of the COR grid")
    _dispopt("'widthCOR'", "Line width of the COR grid")
    _dispopt("'titleCOR'", "'on' (default) / 'off': Show CORRELATION axis title")
    _disp("")

    _disp("Observation Point options:")
    _dispopt("'colObs'", "Observation STD color. (Default: magenta)")
    _dispopt(
        "'markerObs'",
        """Marker to use for x-axis indicating observed STD.
        A choice of 'none' will suppress appearance of marker. (Default 'none')""",
    )
    _dispopt(
        "'styleObs'",
        """Line style for observation grid line. A choice of empty string ('')
        will suppress appearance of the grid line. (Default: '')""",
    )
    _dispopt("'titleOBS'", "Label for observation point (Default: '')")
    _dispopt("'widthOBS'", "Line width for observation grid line")
    _disp("")

    _disp("CONTROL options:")
    _dispopt(
        "'checkStats'",
        """'on' / 'off' (default):
        Check input statistics satisfy Taylor relationship""",
    )


def _disp(text):
    print(text)


def _dispopt(optname, optval):
    """
    Display option name and values.

    This is a support function for the DISPLAY_TARGET_DIAGRAM_OPTIONS function.
    It displays the option name OPTNAME on a line by itself followed by its
    value OPTVAL on the following line.
    """
    _disp("\t{0}".format(optname))
    _disp("\t\t{0}".format(optval))
