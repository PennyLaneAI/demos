#!/usr/bin/env python

"""
Pandoc filter to convert all unrecognized directives to epigraphs (BlockQuotes).
Also convert reference links to links to the online demo's references section.
Code, link URLs, etc. are not affected.
"""

import os

from pandocfilters import BlockQuote, Link, toJSONFilter

DEMOS_URL = "https://pennylane.ai/demos/"


def filter_directives(key, value, format, _):
    if key == "Div":
        [[_, classes, _], body] = value
        if "related" in classes or "meta" in classes:
            return []
        elif "rst-class" in classes:
            metadata = body[0]
            rst_class_type = metadata.get("c")[0].get("c")
            if rst_class_type == "sphx-glr-script-out":
                if len(body) == 1:
                    # This is the new format for sphx-glr-script-out.
                    # Just discard this block.
                    return []
                return body[1]
            else:
                return
        else:
            return BlockQuote(body)

    if key == "Link":
        [roles, _, [link, _]] = value
        if len(link) > 8:
            if link[:8] == "##NOTE##":
                # This is a reference link.
                # Replace it with a link to the online demo.
                link = DEMOS_URL + os.getenv("CURRENT_DEMO") + "#references"
                return Link(roles, [{"t": "Str", "c": " [Refs]"}], [link, ""])

    # Remove the references section from the demo.
    if key == "Header":
        [level, keys, _] = value
        if "references" in keys:
            return []


if __name__ == "__main__":
    toJSONFilter(filter_directives)
