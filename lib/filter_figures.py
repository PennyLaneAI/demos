#!/usr/bin/env python

"""
Pandoc filter to convert figure directives to images.
"""

from pandocfilters import toJSONFilter, RawInline
from filter_helpers import make_html_figure


def filter_images(key, value, format, _):
    if key == 'Image':
        _, _, [url, _] = value
        return RawInline("html", make_html_figure(url))

if __name__ == '__main__':
    toJSONFilter(filter_images)
