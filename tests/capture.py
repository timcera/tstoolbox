"""
Capture function
----------------------------------

"""

# Standard library imports
import sys

try:
    # Third party imports
    from cStringIO import StringIO
except ImportError:
    # Standard library imports
    from io import StringIO


def capture(func, *args, **kwds):
    """Captures stdout from tests to test command line stdout."""
    sys.stdout = StringIO()  # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    try:
        out = bytes(out, "utf8")
    except TypeError:
        pass
    return out
