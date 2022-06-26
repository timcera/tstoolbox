# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def overlay_target_diagram_circles(option):
    """
    Overlay circle contours on a target diagram.

    Plots circle contours on a target diagram to indicate standard
    deviation ranges and observational uncertainty threshold.

    INPUTS:
    option : dictionary containing option values. (Refer to
             GET_TARGET_DIAGRAM_OPTIONS function for more information.)
    option['axismax'] : maximum for the X & Y values. Used to set
            default circles when no contours specified
    option['circles'] : radii of circles to draw to indicate isopleths
            of standard deviation
    option['circleLineSpec'] : circle line specification (default dashed
            black, '--k')
    option['normalized']     : statistics supplied are normalized with
            respect to the standard deviation of reference values
    option['obsUncertainty'] : Observational Uncertainty (default of 0)

    OUTPUTS:
    None.

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    """
    theta = np.arange(0, 2 * np.pi, 0.01)
    unit = np.ones(len(theta))
    # 1 - reference circle if normalized
    if option["normalized"] == "on":
        rho = unit
        X, Y = pol2cart(theta, rho)
        plt.plot(X, Y, "k", "LineWidth", option["circleLineWidth"])

    # Set range for target circles
    if option["normalized"] == "on":
        circles = [0.5, 1]
    elif len(option["circles"]) > 0:
        circles = np.asarray(option["circles"])
        index = np.where(circles <= option["axismax"])
        circles = [option["circles"][i] for i in index[0]]
    else:
        circles = [option["axismax"] * x for x in [0.7, 1]]

    # 2 - secondary circles
    for c in circles:
        rho = c * unit
        X, Y = pol2cart(theta, rho)
        plt.plot(X, Y, option["circlelinespec"], linewidth=option["circlelinewidth"])

    # 3 - Observational Uncertainty threshold
    if option["obsuncertainty"] > 0:
        rho = option["obsuncertainty"] * unit
        X, Y = pol2cart(theta, rho)
        plt.plot(X, Y, "--b")


def pol2cart(phi, rho):
    """
    Transform corresponding polar coordinate arrays to Cartesian coordinates.

    INPUTS:
    phi : polar angle counter-clockwise from x-axis in radians
    rho : radius

    OUTPUTS:
    x   : Cartesian x-coordinate
    y   : Cartesian y-coordinate
    """
    x = np.multiply(rho, np.cos(phi))
    y = np.multiply(rho, np.sin(phi))
    return x, y
