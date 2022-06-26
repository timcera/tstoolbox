# -*- coding: utf-8 -*-
from array import array

import numpy as np

from . import (
    get_target_diagram_axes,
    get_target_diagram_options,
    overlay_target_diagram_circles,
    plot_pattern_diagram_colorbar,
    plot_pattern_diagram_markers,
    plot_target_axes,
)


def target_diagram(*args, **kwargs):
    """
    Plot a target diagram from statistics of different series.

    target_diagram(bs,rmsds,rmsdxz,keyword=value)

    The first 3 arguments must be the inputs as described below followed by
    keywords in the format OPTION = value. An example call to the function
    would be:

    target_diagram(bs,rmsds,rmsdxz,markerdisplayed='marker')

    INPUTS:
    bs    : Bias (B) or Normalized Bias (B*). Plotted along y-axis
            as "Bias".
    rmsds : unbiased Root-Mean-Square Difference (RMSD') or normalized
            unbiased Root-Mean-Square Difference (RMSD*'). Plotted along
            x-axis as "uRMSD".
    rmsdxz : total Root-Mean-Square Difference (RMSD) or other quantities
            (if 'nonrmsdxz' == 'on'). Labeled on plot as "RMSD".

    OUTPUTS:
    None.

    LIST OF OPTIONS:
    For an exhaustive list of options to customize your diagram, call the
    function without arguments at a Python command line:
    % python
    >> from tstoolbox.skill_metrics.target_diagram import target_diagram
    >> target_diagram()

    Reference:

    Jolliff, J. K., J. C. Kindle, I. Shulman, B. Penta, M. Friedrichs,
      R. Helber, and R. Arnone (2009), Skill assessment for coupled
      biological/physical models of marine systems, J. Mar. Sys., 76(1-2),
      64-82, doi:10.1016/j.jmarsys.2008.05.014

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Nov 25, 2016
    """
    # Check for number of arguments
    nargin = len(args)
    bs, rmsds, rmsdxz = _get_target_diagram_arguments(*args)
    if nargin == 0:
        return

    # Get options
    option = get_target_diagram_options(**kwargs)

    #  Get axis values for plot
    axes = get_target_diagram_axes(rmsds, bs, option)

    # Plot axes for target diagram
    if option["overlay"] == "off":
        plot_target_axes(axes)

    # __ Overlay circles
    overlay_target_diagram_circles(option)

    # Plot data points
    lowcase = option["markerdisplayed"].lower()
    if lowcase == "marker":
        plot_pattern_diagram_markers(rmsds, bs, option)
    elif lowcase == "colorbar":
        plot_pattern_diagram_colorbar(rmsds, bs, rmsdxz, option)
    else:
        raise ValueError(f"Unrecognized option: {option['markerdisplayed']}")


def _get_target_diagram_arguments(*args):
    """
    Get arguments for target_diagram function.

    Retrieves the arguments supplied to the TARGET_DIAGRAM function as
    arguments and displays the optional arguments if none are supplied.
    Otherwise, tests the first 3 arguments are numeric quantities and
    returns their values.

    INPUTS:
    args : variable-length input argument list

    OUTPUTS:
    bs    : Bias (B) or Normalized Bias (B*). Plotted along y-axis
            as "Bias".
    rmsds : unbiased Root-Mean-Square Difference (RMSD') or normalized
            unbiased Root-Mean-Square Difference (RMSD*'). Plotted along
            x-axis as "uRMSD".
    rmsdxz : total Root-Mean-Square Difference (RMSD) or other quantities
            (if 'nonrmsdxz' == 'on'). Labeled on plot as "RMSD".
    """
    import numbers

    bs = []
    rmsds = []
    rmsdxz = []
    nargin = len(args)
    if nargin == 0:
        # Display options list
        _display_target_diagram_options()
        return bs, rmsds, rmsdxz
    if nargin != 3:
        raise ValueError("Must supply 3 arguments.")

    bs = args[0]
    rmsds = args[1]
    rmsdxz = args[2]

    # Test the above are numeric quantities
    if isinstance(bs, array):
        bs = np.array(bs)
    if isinstance(bs, numbers.Number):
        bs = np.array(bs, ndmin=1)
    if not isinstance(bs, np.ndarray):
        raise ValueError("Argument bs is not a numeric array")

    if isinstance(rmsds, array):
        rmsds = np.array(rmsds)
    if isinstance(rmsds, numbers.Number):
        rmsds = np.array(rmsds, ndmin=1)
    if not isinstance(rmsds, np.ndarray):
        raise ValueError("Argument rmsds is not a numeric array")

    if isinstance(rmsdxz, array):
        rmsdxz = np.array(rmsdxz)
    if isinstance(rmsdxz, numbers.Number):
        rmsdxz = np.array(rmsdxz, ndmin=1)
    if not isinstance(rmsdxz, np.ndarray):
        raise ValueError("Argument rmsdxz is not a numeric array")

    return bs, rmsds, rmsdxz


def _display_target_diagram_options():
    """Display available options for TARGET_DIAGRAM function."""
    _disp("General options:")
    _dispopt(
        "'overlay'",
        f"'on' / 'off' (default): Switch to overlay current statistics on target diagram. \n		Only markers will be displayed.",
    )
    _dispopt(
        "'colormap'",
        f"'on'/ 'off' (default): Switch to map color shading of markers to colormap ('on')\n		or min to max range of rmsdxz values ('off').\n		Set to same value as option['nonrmsdxz'].",
    )
    _disp("")

    _disp("Marker options:")
    _dispopt(
        "'MarkerDisplayed'",
        f"'marker' (default): Experiments are represented by individual symbols\n		'colorBar': Experiments are represented by a color described in a colorbar",
    )
    _disp("OPTIONS when 'MarkerDisplayed' == 'marker'")
    _dispopt("'markerLabel'", "Labels for markers")
    _dispopt("'markerLabelColor'", "Marker label color (Default: black)")
    _dispopt("'markerColor'", "Marker color")
    _dispopt("'markerLegend'", "'on' / 'off' (default): Use legend for markers'")
    _dispopt("'markerSize'", "Marker size (Default: 10)")
    _disp("OPTIONS when 'MarkerDisplayed' == 'colorbar'")
    _dispopt(
        "'nonrmsdxz'",
        f"'on'/ 'off' (default): Values in rmsds do not correspond to total RMS Differences.\n		(Used to make range of rmsds values appear above color bar.)",
    )
    _dispopt("'titleColorBar'", "Title of the colorbar.")
    _disp("")

    _disp("Axes options:")
    _dispopt(
        "'ticks'",
        "define tick positions (default is that used by axis function)",
    )

    _dispopt(
        "'xtickLabelPos'",
        "position of the tick labels along the x-axis (empty by default)",
    )

    _dispopt(
        "'ytickLabelPos'",
        "position of the tick labels along the y-axis (empty by default)",
    )

    _dispopt("'equalAxes'", "'on' (default) / 'off': Set axes to be equal")
    _dispopt("'limitAxis'", "Max for the Bias & uRMSD axis")
    _disp("")

    _disp("Diagram options:")
    _dispopt(
        "'alpha'",
        f"Blending of symbol face color (0.0 transparent through 1.0 opaque)\n		(Default: 1.0)",
    )
    _dispopt("'axismax'", "Maximum for the Bias & uRMSD axis")
    _dispopt(
        "'circles'",
        "define the radii of circles to draw (default of (maximum rmsds)*[.7 1], [.7 1] when normalized diagram)",
    )

    _dispopt(
        "'circleLineSpec'",
        "Circle line specification (default dashed black, '--k')",
    )

    _dispopt("'circleLineWidth'", "Circle line width")
    _dispopt("'obsUncertainty'", "Observational Uncertainty (default of 0)")
    _dispopt("'normalized'", "'on' / 'off' (default): normalized target diagram")


def _disp(text):
    print(text)


def _dispopt(optname, optval):
    """
    Display option name and values.

    This is a support function for the DISPLAY_TARGET_DIAGRAM_OPTIONS function.
    It displays the option name OPTNAME on a line by itself followed by its
    value OPTVAL on the following line.
    """
    _disp(f"\t{optname}")
    _disp(f"\t\t{optval}")
