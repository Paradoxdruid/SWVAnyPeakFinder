#!/usr/bin/env python3

"""GUI_script wrapper for SWV_AnyPeakFinder."""

import sys
from . import SWV_AnyPeakFinder


def main(args=None):
    """The main routine."""

    app = SWV_AnyPeakFinder.PeakFinderApp()  # noqa


if __name__ == "__main__":
    sys.exit(main())
