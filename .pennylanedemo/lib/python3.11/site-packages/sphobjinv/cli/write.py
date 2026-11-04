r"""*Module for* ``sphobjinv`` *CLI* |Inventory| *writing*.

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

import json
import os
import sys

from sphobjinv.cli.parser import PrsConst
from sphobjinv.cli.paths import resolve_outpath
from sphobjinv.cli.ui import err_format, print_stderr, yesno_prompt
from sphobjinv.fileops import writebytes, writejson
from sphobjinv.zlib import compress


def write_plaintext(inv, path, *, expand=False, contract=False):
    """Write an |Inventory| to plaintext.

    Newlines are inserted in an OS-aware manner,
    based on the value of :data:`os.linesep`.

    Calling with both `expand` and `contract` as |True| is invalid.

    Parameters
    ----------
    inv

        |Inventory| -- Objects inventory to be written as plaintext

    path

        |str| -- Path to output file

    expand

        |bool| *(optional)* -- Generate output with any
        :data:`~sphobjinv.data.SuperDataObj.uri` or
        :data:`~sphobjinv.data.SuperDataObj.dispname`
        abbreviations expanded

    contract

        |bool| *(optional)* -- Generate output with abbreviated
        :data:`~sphobjinv.data.SuperDataObj.uri` and
        :data:`~sphobjinv.data.SuperDataObj.dispname` values

    Raises
    ------
    ValueError

        If both `expand` and `contract` are |True|

    """
    b_str = inv.data_file(expand=expand, contract=contract)
    writebytes(path, b_str.replace(b"\n", os.linesep.encode("utf-8")))


def write_zlib(inv, path, *, expand=False, contract=False):
    """Write an |Inventory| to zlib-compressed format.

       Calling with both `expand` and `contract` as |True| is invalid.

    Parameters
    ----------
    inv

        |Inventory| -- Objects inventory to be written zlib-compressed

    path

        |str| -- Path to output file

    expand

        |bool| *(optional)* -- Generate output with any
        :data:`~sphobjinv.data.SuperDataObj.uri` or
        :data:`~sphobjinv.data.SuperDataObj.dispname`
        abbreviations expanded

    contract

        |bool| *(optional)* -- Generate output with abbreviated
        :data:`~sphobjinv.data.SuperDataObj.uri` and
        :data:`~sphobjinv.data.SuperDataObj.dispname` values

    Raises
    ------
    ValueError

        If both `expand` and `contract` are |True|

    """
    b_str = inv.data_file(expand=expand, contract=contract)
    bz_str = compress(b_str)
    writebytes(path, bz_str)


def write_json(inv, path, params):
    """Write an |Inventory| to JSON.

    Writes output via
    :func:`fileops.writejson() <sphobjinv.fileops.writejson>`.

    Calling with both `expand` and `contract` as |True| is invalid.

    Parameters
    ----------
    inv

        |Inventory| -- Objects inventory to be written zlib-compressed

    path

        |str| -- Path to output file

    params

        dict -- `argparse` parameters

    Raises
    ------
    ValueError

        If both `params["expand"]` and `params["contract"]` are |True|

    """
    json_dict = inv.json_dict(
        expand=params[PrsConst.EXPAND], contract=params[PrsConst.CONTRACT]
    )

    if params.get(PrsConst.FOUND_URL, False):
        json_dict.update({"metadata": {PrsConst.URL: params[PrsConst.FOUND_URL]}})

    writejson(path, json_dict)


def write_stdout(inv, params):
    r"""Write the inventory contents to stdout.

    Parameters
    ----------
    inv

        |Inventory| -- Objects inventory to be written to stdout

    params

        dict -- `argparse` parameters

    Raises
    ------
    ValueError

        If both `params["expand"]` and `params["contract"]` are |True|

    """
    if params[PrsConst.MODE] == PrsConst.PLAIN:
        print(
            inv.data_file(
                expand=params[PrsConst.EXPAND], contract=params[PrsConst.CONTRACT]
            ).decode()
        )
    elif params[PrsConst.MODE] == PrsConst.JSON:
        json_dict = inv.json_dict(
            expand=params[PrsConst.EXPAND], contract=params[PrsConst.CONTRACT]
        )

        if params.get(PrsConst.FOUND_URL, False):
            json_dict.update({"metadata": {PrsConst.URL: params[PrsConst.FOUND_URL]}})

        print(json.dumps(json_dict))
    else:
        print_stderr("Error: Only plaintext and JSON can be emitted to stdout.", params)
        sys.exit(1)


def write_file(inv, in_path, params):
    r"""Write the inventory contents to a file on disk.

    Parameters
    ----------
    inv

        |Inventory| -- Objects inventory to be written to stdout

    in_path

        |str| -- For a local input file, its absolute path.
        For a URL, the (possibly truncated) URL text.

    params

        dict -- `argparse` parameters

    Raises
    ------
    ValueError

        If both `params["expand"]` and `params["contract"]` are |True|

    """
    mode = params[PrsConst.MODE]

    # Work up the output location
    try:
        out_path = resolve_outpath(params[PrsConst.OUTFILE], in_path, params)
    except Exception as e:  # pragma: no cover
        # This may not actually be reachable except in exceptional situations
        print_stderr("\nError while constructing output file path:", params)
        print_stderr(err_format(e), params)
        sys.exit(1)

    # If exists, must handle overwrite
    if os.path.isfile(out_path) and not params[PrsConst.OVERWRITE]:
        if params[PrsConst.INFILE] == "-":
            # If reading from stdin, just alert and don't overwrite
            print_stderr("\nFile exists. To overwrite, supply '-o'. Exiting...", params)
            sys.exit(0)
        # This could be written w/o nesting via elif, but would be harder to read.
        else:
            if not params[PrsConst.QUIET]:
                # If not a stdin read, confirm overwrite; or, just clobber if QUIET
                resp = yesno_prompt("File exists. Overwrite (Y/N)? ")
                if resp.lower() == "n":
                    print_stderr("\nExiting...", params)
                    sys.exit(0)

    # Write the output file
    try:
        if mode == PrsConst.ZLIB:
            write_zlib(
                inv,
                out_path,
                expand=params[PrsConst.EXPAND],
                contract=params[PrsConst.CONTRACT],
            )
        if mode == PrsConst.PLAIN:
            write_plaintext(
                inv,
                out_path,
                expand=params[PrsConst.EXPAND],
                contract=params[PrsConst.CONTRACT],
            )
        if mode == PrsConst.JSON:
            write_json(inv, out_path, params)
    except Exception as e:
        print_stderr("\nError during write of output file:", params)
        print_stderr(err_format(e), params)
        sys.exit(1)

    # Report success, if not QUIET
    print_stderr(
        "Conversion completed.\n"
        f"'{in_path if in_path else 'stdin'}' converted to '{out_path}' ({mode}).",
        params,
    )
