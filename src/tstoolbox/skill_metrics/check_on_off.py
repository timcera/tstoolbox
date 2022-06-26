# -*- coding: utf-8 -*-
def check_on_off(value):
    """
    Check whether variable contains a value of 'on', 'off', True, or False.

    Returns an error if neither for the first two, and sets True to 'on',
    and False to 'off'. The 'on' and 'off' can be provided in any combination
    of upper and lower case letters.

    INPUTS:
    value : string or boolean to check

    OUTPUTS:
    None.

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    """
    if isinstance(value, str):
        lowcase = value.lower()
        if lowcase == "off":
            return lowcase
        if lowcase == "on":
            return lowcase
        raise ValueError(f"Invalid value: {str(value)}")
    if isinstance(value, bool):
        value = "on" if value else "off"
    else:
        raise ValueError(f"Invalid value: {str(value)}")

    return value
