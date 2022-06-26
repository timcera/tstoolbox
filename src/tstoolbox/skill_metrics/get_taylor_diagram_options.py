# -*- coding: utf-8 -*-
import numpy as np


def get_taylor_diagram_options(*args, **kwargs):
    """
    Get optional arguments for taylor_diagram function.

    Retrieves the optional arguments supplied to the TAYLOR_DIAGRAM
    function as a variable-length input argument list (*ARGS), and
    returns the values in an OPTION dictionary. Default values are
    assigned to selected optional arguments. The function will terminate
    with an error if an unrecognized optional argument is supplied.

    INPUTS:
    *kwargs : variable-length keyword argument list. The keywords by
              definition are dictionaries with keys that must correspond to
              one choices given in OUTPUTS below.

    OUTPUTS:
    option : dictionary containing option values. (Refer to
             display_taylor_diagram_options function for more information.)
    option['alpha']           : blending of symbol face color (0.0
                                transparent through 1.0 opaque).
                                (Default: 1.0)
    option['axismax']         : maximum for the radial contours
    option['checkstats']      : Check input statistics satisfy Taylor
                                relationship (Default : 'off')
    option['colcor']   : color for correlation coefficient labels
                         (Default: blue)
    option['colobs']   : color for observation labels (Default : magenta)
    option['colrms']   : color for RMS labels (Default : medium green)
    option['colstd']   : color for STD labels (Default : black)
    option['colormap']        : 'on'/'off' switch to map color shading of
                                 markers to colormap ('on') or min to max range
                                 of RMSDz values ('off'). Set to same value as
                                 option['nonrmsdz'].
    option['markercolor']     : single color to use for all markers
                                (Default: red)
    option['markerdisplayed'] : markers to use for individual experiments
    option['markerlabel']     : name of the experiment to use for marker
    option['markerlabelcolor']: marker label color (Default : 'k')
    option['markerlegend']    : 'on'/'off' switch to display marker legend
                                (Default 'off')
    option['markerobs'  ]     : marker to use for x-axis indicating observed
                                STD. A choice of 'none' will suppress
                                appearance of marker. (Default 'none')
    option['markersize']      : marker size (Default 10)

    option['nonrmsdz']      : 'on'/'off' switch indicating values in RMSDz
                               do not correspond to total RMS Differences.
                               (Default 'off')
    option['numberpanels']  : Number of panels to display
                              = 1 for positive correlations
                              = 2 for positive and negative correlations
                             (Default value depends on correlations (cors))

    option['overlay']       : 'on'/'off' switch to overlay current
                                statistics on Taylor diagram (Default 'off')
                                Only markers will be displayed.
    option['rincrms']       : axis tick increment for RMS values
    option['rincstd']       : axis tick increment for STD values
    option['rmslabelformat'] : string format for RMS contour labels,
                               e.g. '0:.2f'.
                               (Default '0', format as specified by str
                               function)

    option['showlabelscor'] : show correlation coefficient labels
                              (Default: 'on')
    option['showlabelsrms'] : show RMS labels (Default: 'on')
    option['showlabelsstd'] : show STD labels (Default: 'on')

    option['stylecor']      : line style for correlation coefficient grid
                              lines (Default: dash-dot '-.')
    option['styleobs']      : line style for observation grid line. A choice of
                              empty string '' will suppress appearance of the
                              grid line (Default: '')
    option['stylerms']      : line style for RMS grid lines
                              (Default: dash '--')
    option['stylestd']      : line style for STD grid lines
                              (Default: dotted ':')

    option['tickcor'][panel]: tick values for correlation coefficients for
                              two types of panels
    option['tickrms']       : RMS values to plot grid circles from
                              observation point
    option['tickstd']       : STD values to plot grid circles from
                              origin
    option['tickrmsangle']  : tick RMS angle (Default: 135 degrees)
    option['titleColorBar'] : title for the colorbar
    option['titlecor']      : show correlation coefficient axis label
                              (Default: 'on')
    option['titleobs']      : label for observation point (Default: '')
    option['titlerms']      : show RMS axis label (Default: 'on')
    option['titlestd']      : show STD axis label (Default: 'on')

    option['widthcor']      : linewidth for correlation coefficient grid
                              lines (Default: .8)
    option['widthobs']      : linewidth for observation grid line (Default: .8)
    option['widthrms']      : linewidth for RMS grid lines (Default: .8)
    option['widthstd']      : linewidth for STD grid lines (Default: .8)

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Nov 25, 2016
    Revised on Apr  22, 2017
    """
    from . import check_on_off

    cors = args[0]
    nargin = len(kwargs)

    negative = cors[np.where(cors < 0.0)]
    option = {
        "alpha": 1.0,
        "axismax": 0.0,
        "checkstats": "off",
        "colcor": (0, 0, 1),
        "colobs": "m",
        "colrms": (0, 0.6, 0),
        "colstd": (0, 0, 0),
        "colormap": "off",
        "markercolor": "r",
        "markerdisplayed": "marker",
        "markerlabel": "",
        "markerlabelcolor": "k",
        "markerlegend": "off",
        "markerobs": "none",
        "markersize": 10,
        "nonrmsdz": "off",
        "numberpanels": 2 if len(negative) > 0 else 1,
        "overlay": "off",
        "rincrms": [],
        "rincstd": [],
        "rmslabelformat": "0",
        "showlabelscor": "on",
        "showlabelsrms": "on",
        "showlabelsstd": "on",
        "stylecor": "-.",
        "styleobs": "",
        "stylerms": "--",
        "stylestd": ":",
    }

    # Note that "0" must be explicitly given or a scientific number is
    # stored
    tickval1 = [1, 0.99, 0.95, 0]
    middle = np.linspace(0.9, 0.1, 9)
    tickval1[3:3] = middle
    tickval2 = tickval1[:]
    values = np.linspace(-0.1, -0.9, 9)
    tickval2.extend(values)
    tickval2.extend([-0.95, -0.99, -1])
    option["tickcor"] = (tickval1, tickval2)  # store as tuple

    option["tickrms"] = []
    option["tickstd"] = []
    option["tickrmsangle"] = -1
    option["titlecolorbar"] = ""
    option["titlecor"] = "on"
    option["titleobs"] = ""
    option["titlerms"] = "on"
    option["titlestd"] = "on"

    option["widthcor"] = 1.0
    option["widthobs"] = 1.0
    option["widthrms"] = 1.0
    option["widthstd"] = 1.0

    if nargin == 0:
        # No options requested, so return with only defaults
        return option

    # Check for valid keys and values in dictionary
    for optname, optvalue in kwargs.items():
        optname = optname.lower()
        if optname not in option:
            raise ValueError(f"Unrecognized option: {optname}")
        # Replace option value with that from arguments
        if optname == "tickcor":
            list1 = option["tickcor"][0]
            list2 = option["tickcor"][1]
            if option["numberpanels"] == 1:
                list1 = optvalue
            else:
                list2 = optvalue
            option["tickcor"] = (list1, list2)
        else:
            option[optname] = optvalue

        # Check values for specific options
        if optname == "checkstats":
            option["checkstats"] = check_on_off(option["checkstats"])
        elif optname == "markerlabel":
            if not isinstance(optvalue, list):
                raise ValueError(f"Option value is not a list: {str(optvalue)}")
            option["markerlabel"] = optvalue[1:]
        elif optname == "markerlegend":
            option["markerlegend"] = check_on_off(option["markerlegend"])
        elif optname == "overlay":
            option["overlay"] = check_on_off(option["overlay"])
        elif optname == "rmslabelformat":
            # Check for valid string format
            labelformat = f"{{{optvalue}}}"
            try:
                labelformat.format(99.0)
            except ValueError:
                raise ValueError(
                    f"""
        *
        *   Invalid string format for rmslabelformat: {optvalue}
        *
        """
                )
        elif optname == "showlabelscor":
            option["showlabelscor"] = check_on_off(option["showlabelscor"])
        elif optname == "showlabelsrms":
            option["showlabelsrms"] = check_on_off(option["showlabelsrms"])
        elif optname == "showlabelsstd":
            option["showlabelsstd"] = check_on_off(option["showlabelsstd"])
        elif optname == "tickrms":
            option["tickrms"] = np.sort(optvalue)
            option["rincrms"] = (max(option["tickrms"]) - min(option["tickrms"])) / len(
                option["tickrms"]
            )
        elif optname == "tickstd":
            option["tickstd"] = np.sort(optvalue)
            option["rincstd"] = (max(option["tickstd"]) - min(option["tickstd"])) / len(
                option["tickstd"]
            )
        elif optname == "titlecor":
            option["titlecor"] = check_on_off(option["titlecor"])
        elif optname == "titlerms":
            option["titlerms"] = check_on_off(option["titlerms"])
        elif optname == "titlestd":
            option["titlestd"] = check_on_off(option["titlestd"])

    option["colormap"] = option["nonrmsdz"]

    return option
