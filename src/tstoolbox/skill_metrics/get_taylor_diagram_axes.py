# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def get_taylor_diagram_axes(rho, option):
    """
    Get axes value for taylor_diagram function.

    Determines the axes information for a Taylor diagram given the axis
    values (X,Y) and the options in the dictionary OPTION returned by
    the GET_TAYLOR_DIAGRAM_OPTIONS function.

    INPUTS:
    rho    : radial coordinate
    option : dictionary containing option values. (Refer to
             GET_TAYLOR_DIAGRAM_OPTIONS function for more information.)

    OUTPUTS:
    axes         : dictionary containing axes information for Taylor diagram
    axes['dx']   : observed standard deviation
    axes['next'] : directive on how to add next plot
    axes['rinc'] : increment for radial coordinate
    axes['rmax'] : maximum value for radial coordinate
    axes['rmin'] : minimum value for radial coordinate
    axes['tc']   : color for x-axis
    option       : dictionary containing updated option values

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Dec 3, 2016
    """
    axes = {"dx": rho[0]}
    cax = plt.gca()
    axes["tc"] = cax.xaxis.label.get_color()
    axes["next"] = "replace"  # needed?

    # make a radial grid
    maxrho = max(abs(rho)) if option["axismax"] == 0.0 else option["axismax"]
    # Determine default number of tick marks
    yticks = ticker.AutoLocator().tick_values(-1.0 * maxrho, maxrho)
    ticks = np.sum(yticks >= 0)

    # Check radial limits and ticks
    axes["rmin"] = 0
    if option["axismax"] == 0.0:
        axes["rmax"] = yticks[-1]
        option["axismax"] = axes["rmax"]
    else:
        axes["rmax"] = option["axismax"]
    rticks = np.amax(ticks - 1, axis=0)
    if rticks > 5:  # see if we can reduce the number
        if rticks % 2 == 0:
            rticks = rticks / 2
        elif rticks % 3 == 0:
            rticks = rticks / 3
    axes["rinc"] = (axes["rmax"] - axes["rmin"]) / rticks
    tick = np.arange(
        axes["rmin"] + axes["rinc"], axes["rmax"] + axes["rinc"], axes["rinc"]
    )

    if len(option["tickrms"]) == 0:
        option["tickrms"] = tick
        option["rincrms"] = axes["rinc"]
    if len(option["tickstd"]) == 0:
        option["tickstd"] = tick
        option["rincstd"] = axes["rinc"]

    return axes, cax
