# -*- coding: utf-8 -*-
import warnings

import matplotlib.colors as clr
import matplotlib.pyplot as plt


def plot_pattern_diagram_markers(X, Y, option):
    """
    Plot color markers on a pattern diagram.

    Plots color markers on a target diagram according their (X,Y)
    locations. The symbols and colors are chosen automatically with a
    limit of 70 symbol & color combinations.

    The color bar is titled using the content of option['titleColorBar']
    (if non-empty string).

    INPUTS:
    x : x-coordinates of markers
    y : y-coordinates of markers
    z : z-coordinates of markers (used for color shading)
    option : dictionary containing option values. (Refer to
        GET_TARGET_DIAGRAM_OPTIONS function for more information.)
    option['axismax'] : maximum for the X & Y values. Used to limit
        maximum distance from origin to display markers
    option['markerlabel'] : labels for markers

    OUTPUTS:
    None

    Created on Nov 30, 2016

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    """
    # Set face color transparency
    alpha = option["alpha"]

    # Set marker size
    markerSize = option["markersize"]

    if option["markerlegend"] == "on":
        # Check that marker labels have been provided
        if option["markerlabel"] == "":
            raise ValueError("No marker labels provided.")

        # Plot markers of different color and shapes with labels
        # displayed in a legend

        # Define markers
        kind = ["+", "o", "x", "s", "d", "^", "v", "p", "h", "*"]
        colorm = ["b", "r", "g", "c", "m", "y", "k"]
        if len(X) > 70:
            _disp("You must introduce new markers to plot more than 70 cases.")
            _disp(
                "The " "marker" " character array need to be extended inside the code."
            )

        # Define markers with specified color
        marker = []
        if len(X) <= len(kind):
            markercolor = []
            for _ in colorm:
                for symbol in kind:
                    marker.append(symbol + option["markercolor"])
                    rgba = clr.to_rgb(option["markercolor"]) + (alpha,)
                    markercolor.append(rgba)
        else:
            for color in colorm:
                for symbol in kind:
                    marker.append(symbol + color)
                    rgba = clr.to_rgb(color) + (alpha,)
                    markercolor.append(rgba)

        # Plot markers at data points
        limit = option["axismax"]
        hp = ()
        markerlabel = []
        for i, xval in enumerate(X):
            if abs(X[i]) <= limit and abs(Y[i]) <= limit:
                h = plt.plot(
                    X[i],
                    Y[i],
                    marker[i],
                    markersize=markerSize,
                    markerfacecolor=markercolor[i],
                    markeredgecolor=marker[i][1],
                    markeredgewidth=2,
                )
                hp += tuple(h)
                markerlabel.append(option["markerlabel"][i])

        # Add legend
        if markerlabel:
            markerlabel = tuple(markerlabel)
            plt.legend(
                hp, markerlabel, loc="upper right", fontsize="medium", numpoints=1
            )
        else:
            warnings.warn("No markers within axis limit ranges.")

    else:
        # Plot markers as dots of a single color with accompanying labels
        # and no legend

        # Plot markers at data points
        limit = option["axismax"]
        rgba = clr.to_rgb(option["markercolor"]) + (alpha,)
        for i, xval in enumerate(X):
            if abs(X[i]) <= limit and abs(Y[i]) <= limit:
                # Plot marker
                plt.plot(
                    X[i],
                    Y[i],
                    ".",
                    markersize=markerSize,
                    markerfacecolor=rgba,
                    markeredgecolor=option["markercolor"],
                )

                # Check if marker labels provided
                if len(option["markerlabel"]) > 0:
                    # Label marker
                    xtextpos = X[i]  # ToDo: convert to double?
                    ytextpos = Y[i]
                    plt.text(
                        xtextpos,
                        ytextpos,
                        option["markerlabel"][i],
                        color=option["markerlabelcolor"],
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        fontsize="medium",
                    )


def _disp(text):
    print(text)
