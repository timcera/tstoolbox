# -*- coding: utf-8 -*-

"""
Capture function
----------------------------------

"""

import sys

try:
    from cStringIO import StringIO
except:
    from io import StringIO


def capture(func, *args, **kwds):
    sys.stdout = StringIO()  # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    try:
        out = bytes(out, "utf8")
    except:
        pass
    return out
