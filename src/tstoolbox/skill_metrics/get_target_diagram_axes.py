# -*- coding: utf-8 -*-
import matplotlib.ticker as ticker
import numpy as np


def get_target_diagram_axes(x, y, option):
    """
    Get axes value for target_diagram function.

    Determines the axes information for a target diagram given the axis values
    (X,Y) and the options in the data structure OPTION returned by the
    GET_TARGET_DIAGRAM_OPTIONS function.

    INPUTS:
    x      : values for x-axis
    y      : values for y-axis
    option : dictionary containing option values. (Refer to
             GET_TARGET_DIAGRAM_OPTIONS function for more information.)

    OUTPUTS:
    axes           : dictionary containing axes information for target diagram
    axes['xtick']  : x-values at which to place tick marks
    axes['ytick']  : y-values at which to place tick marks
    axes['xlabel'] : labels for xtick values
    axes['ylabel'] : labels for ytick values
    option : dictionary containing updated option values

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Nov 25, 2016

    @author: rochfordp
    """
    # Specify max/min for axes
    foundmax = 1 if option["axismax"] != 0.0 else 0
    if foundmax == 0:
        # Axis limit not specified
        maxx = np.amax(np.absolute(x))
        maxy = np.amax(np.absolute(y))
    else:
        # Axis limit is specified
        maxx = option["axismax"]
        maxy = option["axismax"]

    # Determine default number of tick marks
    xtickvals = ticker.AutoLocator().tick_values(-1.0 * maxx, maxx)
    ytickvals = ticker.AutoLocator().tick_values(-1.0 * maxy, maxy)
    ntest = np.sum(xtickvals > 0)
    if ntest > 0:
        nxticks = np.sum(xtickvals > 0)
        nyticks = np.sum(ytickvals > 0)

        # Save nxticks and nyticks as function attributes for later
        # retrieval in function calls
        get_target_diagram_axes.nxticks = nxticks
        get_target_diagram_axes.nyticks = nyticks
    elif hasattr(get_target_diagram_axes, "nxticks"):
        nxticks = get_target_diagram_axes.nxticks
        nyticks = get_target_diagram_axes.nyticks
    else:
        raise ValueError("No saved values for nxticks & nyticks.")

    # Set default tick increment and maximum axis values
    if foundmax == 0:
        maxx = xtickvals[-1]
        maxy = ytickvals[-1]
        option["axismax"] = max(maxx, maxy)

    # Check if equal axes requested
    if option["equalaxes"] == "on":
        if maxx > maxy:
            maxy = maxx
            nyticks = nxticks
        else:
            maxx = maxy
            nxticks = nyticks

    # Convert to integer if whole number
    if isinstance(maxx, float) and maxx.is_integer():
        maxx = int(round(maxx))
    if isinstance(maxx, float) and maxy.is_integer():
        maxy = int(round(maxy))
    minx = -maxx
    miny = -maxy

    # Determine tick values
    if len(option["ticks"]) > 0:
        xtick = option["ticks"]
        ytick = option["ticks"]
    else:
        tincx = maxx / nxticks
        tincy = maxy / nyticks
        xtick = np.arange(minx, maxx + tincx, tincx)
        ytick = np.arange(miny, maxy + tincy, tincy)

    # Assign tick label positions
    if len(option["xticklabelpos"]) == 0:
        option["xticklabelpos"] = xtick
    if len(option["yticklabelpos"]) == 0:
        option["yticklabelpos"] = ytick

    # Set tick labels using provided tick label positions
    xlabel = []
    ylabel = []

    # Set x tick labels
    for item in xtick:
        index = np.where(option["xticklabelpos"] == item)
        if len(index) > 0:
            xlabel.append(str(item))
        else:
            xlabel.append("")

    # Set tick labels at 0 to blank
    index = np.where(abs(xtick) < 1.0e-7)
    index = np.asscalar(index[0])
    xlabel[index] = ""

    # Set y tick labels
    for i, item in enumerate(ytick):
        index = np.where(option["xticklabelpos"] == xtick[i])
        if len(index) > 0:
            ylabel.append(str(item))
        else:
            ylabel.append("")

    # Set tick labels at 0 to blank
    index = np.where(abs(ytick) < 1.0e-7)
    index = np.asscalar(index[0])
    ylabel[index] = ""

    return {"xtick": xtick, "ytick": ytick, "xlabel": xlabel, "ylabel": ylabel}
