r"""``sphobjinv`` *CLI core execution module*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    15 Nov 2020

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

from sphobjinv.cli.convert import do_convert
from sphobjinv.cli.load import inv_local, inv_stdin, inv_url
from sphobjinv.cli.parser import PrsConst, getparser, getparser_textconv
from sphobjinv.cli.suggest import do_suggest
from sphobjinv.cli.ui import print_stderr


def main():
    r"""Handle command line invocation.

    Parses command line arguments,
    handling the no-arguments and
    VERSION cases.

    Creates the |Inventory| from the indicated source
    and method.

    Invokes :func:`~sphobjinv.cli.convert.do_convert` or
    :func:`~sphobjinv.cli.suggest.do_suggest`
    per the subparser name stored in SUBPARSER_NAME.

    """
    # If no args passed, stick in '-h'
    if len(sys.argv) == 1:
        sys.argv.append("-h")

    # Parse commandline arguments, discarding any unknown ones
    # I forget why I set it up to discard these, it might be
    # more confusing than it's worth to swallow them this way....
    prs = getparser()
    ns, _ = prs.parse_known_args()
    params = vars(ns)

    # Print version &c. and exit if indicated
    if params[PrsConst.VERSION]:
        print(PrsConst.VER_TXT)
        sys.exit(0)

    # At this point, need to trap for a null subparser
    if not params[PrsConst.SUBPARSER_NAME]:
        prs.error("No subparser selected")

    # Regardless of mode, insert extra blank line
    # for cosmetics
    print_stderr(" ", params)

    # Generate the input Inventory based on --url or stdio or file.
    # These inventory-load functions should call
    # sys.exit(n) internally in error-exit situations
    if params[PrsConst.URL]:
        if params[PrsConst.INFILE] == "-":
            prs.error("argument -u/--url not allowed with '-' as infile")
        inv, in_path = inv_url(params)
    elif params[PrsConst.INFILE] == "-":
        inv = inv_stdin(params)
        in_path = None
    else:
        inv, in_path = inv_local(params)

    # Perform action based upon mode
    if params[PrsConst.SUBPARSER_NAME][:2] == PrsConst.CONVERT[:2]:
        do_convert(inv, in_path, params)
    elif params[PrsConst.SUBPARSER_NAME][:2] == PrsConst.SUGGEST[:2]:
        do_suggest(inv, params)

    # Cosmetic final blank line
    print_stderr(" ", params)

    # Clean exit
    sys.exit(0)


def main_textconv():
    """Entrypoint for textconv operation."""
    # If no args passed, stick in '-h'
    if len(sys.argv) == 1:
        sys.argv.append("-h")

    prs = getparser_textconv()
    params = vars(prs.parse_args())

    # No version arg handling, using 'version' action in this parser

    inv, in_path = inv_local(params)

    params[PrsConst.CONTRACT] = False
    params[PrsConst.EXPAND] = False
    params[PrsConst.MODE] = PrsConst.PLAIN
    params[PrsConst.OUTFILE] = "-"

    do_convert(inv, in_path, params)

    sys.exit(0)
