r"""``sphobjinv`` *module for CLI convert functionality*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    20 Oct 2022

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

from sphobjinv.cli.parser import PrsConst
from sphobjinv.cli.write import write_file, write_stdout


def do_convert(inv, in_path, params):
    r"""Carry out the conversion operation, including writing output.

    If OVERWRITE is passed and the output file
    (the default location, or as passed to OUTFILE)
    exists, it will be overwritten without a prompt. Otherwise,
    the user will be queried if it is desired to overwrite
    the existing file.

    If QUIET is passed, nothing will be
    printed to |cour|\ stdout\ |/cour|
    (potentially useful for scripting),
    and any existing output file will be overwritten
    without prompting.

    Parameters
    ----------
    inv

        |Inventory| -- Inventory object to be output in the format
        indicated by MODE.

    in_path

        |str| -- For a local input file, its absolute path.
        For a URL, the (possibly truncated) URL text.

    params

        |dict| -- Parameters/values mapping from the active subparser

    """
    if params[PrsConst.OUTFILE] == "-" or (
        params[PrsConst.INFILE] == "-" and params[PrsConst.OUTFILE] is None
    ):
        write_stdout(inv, params)
    else:
        write_file(inv, in_path, params)
