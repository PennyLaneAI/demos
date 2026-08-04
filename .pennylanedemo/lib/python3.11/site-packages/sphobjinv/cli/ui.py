r"""``sphobjinv`` *CLI UI functions*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    19 Nov 2020

**Copyright**
    \(c) 2016-2026 Brian Skinn and community contributors

**Source Repository**
    https://github.com/bskinn/sphobjinv

**Documentation**
    https://sphobjinv.readthedocs.io/en/stable

**License**
    Code: `MIT License`_

    Docs & Docstrings: |CC BY 4.0|_

    See |license_txt|_ for full license terms.

**Members**

"""

import sys

from sphobjinv.cli.parser import PrsConst


def print_stderr(thing, params, *, end="\n"):
    r"""Print `thing` to stderr if not in quiet mode.

    Quiet mode is indicated by the value at the QUIET key
    within `params`.

    Quiet mode is not implemented for the ":doc:`suggest </cli/suggest>`"
    CLI mode.

    Parameters
    ----------
    thing

        *any* -- Object to be printed

    params

        |dict| -- Parameters/values mapping from the active subparser

    end

        |str| -- String to append to printed content (default: ``\n``\ )

    """
    if (
        PrsConst.SUBPARSER_NAME not in params  # textconv
        or params[PrsConst.SUBPARSER_NAME][:2] == "su"  # suggest is never quiet
        or not params[PrsConst.QUIET]  # non-quiet convert
    ):
        print(thing, file=sys.stderr, end=end)


def err_format(exc):
    r"""Pretty-format an exception.

    Parameters
    ----------
    exc

        :class:`Exception` -- Exception instance to pretty-format

    Returns
    -------
    pretty_exc

        |str| -- Exception type and message formatted as
        |cour|\ '{type}: {message}'\ |/cour|

    """
    return f"{type(exc).__name__}: {str(exc)}"


def yesno_prompt(prompt):
    r"""Query user at `stdin` for yes/no confirmation.

    Uses :func:`input`, so will hang if used programmatically
    unless `stdin` is suitably mocked.

    The value returned from :func:`input` must satisfy either
    |cour|\ resp.lower() == 'n'\ |/cour| or
    |cour|\ resp.lower() == 'y'\ |/cour|,
    or else the query will be repeated *ad infinitum*.
    This function does **NOT** augment `prompt`
    to indicate the constraints on the accepted values.

    Parameters
    ----------
    prompt

        |str| -- Prompt to display to user that
        requests a 'Y' or 'N' response

    Returns
    -------
    resp

        |str| -- User response

    """
    resp = ""
    while not (resp.lower() == "n" or resp.lower() == "y"):
        resp = input(prompt)
    return resp
