r"""``sphobjinv`` *CLI path resolution module*.

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

import os

from sphobjinv.cli.parser import PrsConst


def resolve_inpath(in_path):
    """Resolve the input file, handling invalid values.

    Currently, only checks for existence and not-directory.

    Parameters
    ----------
    in_path

        |str| -- Path to desired input file

    Returns
    -------
    abs_path

        |str| -- Absolute path to indicated file

    Raises
    ------
    :exc:`FileNotFoundError`

        If a file is not found at the given path

    """
    # Path MUST be to a file, that exists
    if not os.path.isfile(in_path):
        raise FileNotFoundError("Indicated path is not a valid file")

    # Return the path as absolute
    return os.path.abspath(in_path)


def resolve_outpath(out_path, in_path, params):
    r"""Resolve the output location, handling mode-specific defaults.

    If the output path or basename are not specified, they are
    taken as the same as the input file. If the extension is
    unspecified, it is taken as the appropriate mode-specific value
    from DEF_OUT_EXT.

    If URL is passed, the input directory
    is taken to be :func:`os.getcwd` and the input basename
    is taken as DEF_BASENAME.

    Parameters
    ----------
    out_path

        |str| or |None| -- Output location provided by the user,
        or |None| if omitted

    in_path

        |str| -- For a local input file, its absolute path.
        For a URL, the (possibly truncated) URL text.

    params

        |dict| -- Parameters/values mapping from the active subparser

    Returns
    -------
    out_path

        |str| -- Absolute path to the target output file

    """
    mode = params[PrsConst.MODE]

    if params[PrsConst.URL] or in_path is None:
        in_fld = os.getcwd()
        in_fname = PrsConst.DEF_BASENAME
    else:
        in_fld, in_fname = os.path.split(in_path)

    if out_path:
        # Must check if the path entered is a folder
        if os.path.isdir(out_path):
            # Set just the folder and leave the name blank
            out_fld = out_path
            out_fname = None
        else:
            # Split appropriately
            out_fld, out_fname = os.path.split(out_path)

        # Output to same folder if unspecified
        if not out_fld:
            out_fld = in_fld

        # Use same base filename if not specified
        if not out_fname:
            out_fname = os.path.splitext(in_fname)[0] + PrsConst.DEF_OUT_EXT[mode]

        # Composite the full output path
        out_path = os.path.join(out_fld, out_fname)
    else:
        # No output location specified; use defaults
        out_fname = os.path.splitext(in_fname)[0] + PrsConst.DEF_OUT_EXT[mode]
        out_path = os.path.join(in_fld, out_fname)

    return out_path
