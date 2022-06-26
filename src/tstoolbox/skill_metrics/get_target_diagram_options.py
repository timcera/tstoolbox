# -*- coding: utf-8 -*-
def get_target_diagram_options(**kwargs):
    """
    Get optional arguments for target_diagram function.

    Retrieves the optional arguments supplied to the TARGET_DIAGRAM
    function as a variable-length keyword argument list (*KWARGS), and
    returns the values in an OPTION dictionary. Default values are
    assigned to selected optional arguments. The function will terminate
    with an error if an unrecognized optional argument is supplied.

    INPUTS:
    *kwargs : variable-length keyword argument list. The keywords by
              definition are dictionaries with keys that must correspond to
              one choices given in OUTPUTS below.

    OUTPUTS:
    option : dictionary containing option values. (Refer to
             display_target_diagram_options function for more information.)
    option['alpha']           : blending of symbol face color (0.0
                                transparent through 1.0 opaque). (Default
                                : 1.0)
    option['axismax']         : maximum for the Bias & uRMSD axis
    option['circlelinespec']  : circle line specification (default dashed
                             black, '--k')
    option['circlelinewidth'] : circle line width specification (default 0.5)
    option['circles']         : radii of circles to draw to indicate
                                isopleths of standard deviation (empty by
                                default)
    option['colormap']        : 'on'/'off' switch to map color shading of
                                 markers to colormap ('on') or min to max range
                                 of RMSDz values ('off'). Set to same value as
                                 option['nonrmsdz'].
    option['equalAxes']       : 'on'/'off' switch to set axes to be equal
    option['markerdisplayed'] : markers to use for individual experiments
    option['markerlabel']     : name of the experiment to use for marker
    option['markerlabelcolor'] : marker label color (Default 'k')
    option['markerlegend']    : 'on'/'off' switch to display marker legend
                                (Default 'off')
    option['markersize']      : marker size (Default 10)

    option['nonrmsdz']        : 'on'/'off' switch indicating values in RMSDz
                                do not correspond to total RMS Differences.
                                (Default 'off')
    option['normalized']      : statistics supplied are normalized with
                                respect to the standard deviation of reference
                                values (Default 'off')
    option['obsUncertainty']  : Observational Uncertainty (default of 0)
    option['overlay']         : 'on'/'off' switch to overlay current
                                statistics on Taylor diagram (Default 'off')
                                Only markers will be displayed.
    option['ticks']           : define tick positions (default is that used
                                by the axis function)
    option['titlecolorbar']   : title for the colorbar
    option['xticklabelpos']   : position of the tick labels along the x-axis
                                (empty by default)
    option['yticklabelpos']   : position of the tick labels along the y-axis
                                (empty by default)

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Nov 25, 2016
    """
    from . import check_on_off

    nargin = len(kwargs)

    # Set default parameters for all options
    option = {
        "alpha": 1.0,
        "axismax": 0.0,
        "circlelinespec": "k--",
        "circlelinewidth": 1.5,
        "circles": [],
        "colormap": "off",
        "equalaxes": "on",
        "markercolor": "r",
        "markerdisplayed": "marker",
        "markerlabel": "",
        "markerlabelcolor": "k",
        "markerlegend": "off",
        "markersize": 10,
        "nonrmsdz": "off",
        "normalized": "off",
        "obsuncertainty": 0.0,
        "overlay": "off",
        "ticks": [],
        "titlecolorbar": "",
        "xticklabelpos": [],
        "yticklabelpos": [],
    }

    if nargin == 0:
        # No options requested, so return with only defaults
        return option

    # Check for valid keys and values in dictionary
    for optname, optvalue in kwargs.items():
        optname = optname.lower()
        if optname not in option:
            raise ValueError(f"Unrecognized option: {optname}")
        # Replace option value with that from arguments
        option[optname] = optvalue

        # Check values for specific options
        if optname == "equalaxes":
            option["equalaxes"] = check_on_off(option["equalaxes"])
        elif optname == "markerlegend":
            option["markerlegend"] = check_on_off(option["markerlegend"])
        elif optname == "nonrmsdz":
            option["nonrmsdz"] = check_on_off(option["nonrmsdz"])
        elif optname == "normalized":
            option["normalized"] = check_on_off(option["normalized"])
        elif optname == "overlay":
            option["overlay"] = check_on_off(option["overlay"])

    option["colormap"] = option["nonrmsdz"]

    return option
