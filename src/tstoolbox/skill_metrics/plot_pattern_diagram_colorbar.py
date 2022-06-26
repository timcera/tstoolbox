# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_pattern_diagram_colorbar(X, Y, Z, option):
    """
    Plot color markers on a pattern diagram shaded to a supplied value.

    Values are indicated via a color bar on the plot.

    Plots color markers on a target diagram according their (X,Y) locations.
    The color shading is accomplished by plotting the markers as a scatter
    plot in (X,Y) with the colors of each point specified using Z as a
    vector.

    The color range is controlled by option['nonRMSDz'].
    option['colormap'] = 'on' :
        the scatter function maps the elements in Z to colors in the
        current colormap
    option['colormap']= 'off' : the color axis is mapped to the range
        [min(Z) max(Z)]

    The color bar is titled using the content of option['titleColorBar']
    (if non-empty string).

    INPUTS:
    x : x-coordinates of markers
    y : y-coordinates of markers
    z : z-coordinates of markers (used for color shading)
    option : dictionary containing option values.
    option['colormap'] : 'on'/'off' switch to map color shading of markers
        to colormap ('on') or min to max range of Z values ('off').
    option['titleColorBar'] : title for the color bar

    OUTPUTS:
    None.

    Created on Nov 30, 2016

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    """
    # Plot color shaded data points using scatter plot
    # Keyword s defines marker size in points^2
    #         c defines the sequence of numbers to be mapped to colors
    #           using the cmap and norm
    _ = plt.scatter(X, Y, s=50, c=Z, marker="d", edgecolors="none")

    # Add color bar to plot
    if option["colormap"] == "on":
        hc = plt.colorbar(orientation="horizontal", aspect=6, fraction=0.04, pad=0.04)

    elif option["colormap"] == "off":
        if len(Z) > 1:
            plt.clim(min(Z), max(Z))
            hc = plt.colorbar(
                orientation="horizontal", aspect=6, fraction=0.04, pad=0.04
            )
            #                 hc.set_ticks([min(Z), max(Z)])
            hc.set_ticklabels("Min. RMSD", "Max. RMSD")
    else:
        raise ValueError(f"Invalid option for option.colormap: {option['colormap']}")

    # Set desired properties of color bar
    location = _getColorBarLocation(hc, option, xscale=0.75, yscale=7.5, cxscale=1.0)
    hc.ax.set_position(location)  # set new position
    hc.ax.tick_params(labelsize="medium")  # set tick labels to medium

    # Limit number of ticks on colar bar to 4
    hc.locator = ticker.MaxNLocator(nbins=5)
    hc.update_ticks()

    hc.ax.xaxis.set_ticks_position("top")
    hc.ax.xaxis.set_label_position("top")

    # Title the color bar
    if option["titlecolorbar"]:
        hc.set_label(option["titlecolorbar"])
    else:
        hc.set_label(hc, "Color Scale")


def _getColorBarLocation(hc, option, **kwargs):
    """
    Determine location for color bar.

    Determines location to place color bar for type of plot:
    target diagram and Taylor diagram. Optional scale arguments
    (xscale,cxscale) can be supplied to adjust the placement of
    the colorbar to accommodate different situations.

    INPUTS:
    hc     : handle returned by colorbar function
    option : dictionary containing option values. (Refer to
             display_target_diagram_options function for more
             information.)

    OUTPUTS:
    location : x, y, width, height for color bar

    KEYWORDS:
    xscale  : scale factor to adjust x-position of color bar
    yscale  : scale factor to adjust y-position of color bar
    cxscale : scale factor to adjust thickness of color bar
    """
    # Check for optional arguments and set defaults if required
    xscale = kwargs.get("xscale", 1.0)
    yscale = kwargs.get("yscale", 1.0)
    cxscale = kwargs.get("cxscale", 1.0)

    # Get current position of color bar
    cp = hc.ax.get_position()

    return (
        [
            cp.x0 + xscale * 0.8 * cp.width,
            yscale * cp.y0,
            cxscale * cp.width / 6,
            cp.height,
        ]
        if "checkSTATS" in option
        else [
            cp.x0 + xscale * cp.width,
            yscale * cp.y0,
            cxscale * cp.width / 6,
            cp.height,
        ]
    )
