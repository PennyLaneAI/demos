r"""``sphobjinv`` *CLI parser definition module*.

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

import argparse as ap

from sphobjinv.version import __version__


class PrsConst:
    """Container for CLI parser constants."""

    # ### Version arg and helpers
    #: Optional argument name for use with the base
    #: argument parser, to show version &c. info, and exit
    VERSION = "version"

    #: Version &c. output blurb
    VER_TXT = (
        f"\nsphobjinv v{__version__}\n\n"
        "Copyright (c) 2016-2026 Brian Skinn and community contributors\n"
        "License: The MIT License\n\n"
        "Bug reports & feature requests:"
        " https://github.com/bskinn/sphobjinv\n"
        "Documentation:"
        " https://sphobjinv.readthedocs.io\n"
    )

    #: Short version text for textconv entrypoint
    VER_TXT_SHORT = f"sphobjinv v{__version__}"

    # ### Subparser selectors and argparse param for storing subparser name
    #: Subparser name for inventory file conversions; stored in
    #: :data:`SUBPARSER_NAME` when selected
    CONVERT = "convert"

    #: Subparser name for inventory object suggestions; stored in
    #: :data:`SUBPARSER_NAME` when selected
    SUGGEST = "suggest"

    #: Param for storing subparser name
    #: (:data:`CONVERT` or :data:`SUGGEST`)
    SUBPARSER_NAME = "sprs_name"

    # ### Common URL argument for both subparsers
    #: Optional argument name for use with both :data:`CONVERT` and
    #: :data:`SUGGEST` subparsers, indicating that
    #: :data:`INFILE` is to be treated as a URL
    #: rather than a local file path
    URL = "url"

    # ### Conversion subparser: 'mode' param and choices
    #: Positional argument name for use with :data:`CONVERT` subparser,
    #: indicating output file format
    #: (:data:`ZLIB`, :data:`PLAIN` or :data:`JSON`)
    MODE = "mode"

    #: Argument value for :data:`CONVERT` :data:`MODE`,
    #: to output a :mod:`zlib`-compressed inventory
    ZLIB = "zlib"

    #: Argument value for :data:`CONVERT` :data:`MODE`,
    #: to output a plaintext inventory
    PLAIN = "plain"

    #: Argument value for :data:`CONVERT` :data:`MODE`,
    #: to output an inventory as JSON
    JSON = "json"

    # ### Source/destination params
    #: Required positional argument name for use with both :data:`CONVERT` and
    #: :data:`SUGGEST` subparsers, holding the path
    #: (or URL, if :data:`URL` is specified)
    #: to the input file
    INFILE = "infile"

    #: Optional positional argument name
    #: for use with the :data:`CONVERT` subparser,
    #: holding the path to the output file
    #: (:data:`DEF_BASENAME` and the appropriate item from :data:`DEF_OUT_EXT`
    #: are used if this argument is not provided)
    OUTFILE = "outfile"

    # ### Convert subparser optional params
    #: Optional argument name for use with the :data:`CONVERT` subparser,
    #: indicating to suppress console output
    QUIET = "quiet"

    #: Optional argument name for use with the :data:`CONVERT` subparser,
    #: indicating to expand URI and display name
    #: abbreviations in the generated output file
    EXPAND = "expand"

    #: Optional argument name for use with the :data:`CONVERT` subparser,
    #: indicating to contract URIs and display names
    #: to abbreviated forms in the generated output file
    CONTRACT = "contract"

    #: Optional argument name for use with the :data:`CONVERT` subparser,
    #: indicating to overwrite any existing output
    #: file without prompting
    OVERWRITE = "overwrite"

    # ### Suggest subparser params
    #: Positional argument name for use with the :data:`SUGGEST` subparser,
    #: holding the search term for |fuzzywuzzy|_ text matching
    SEARCH = "search"

    #: Optional argument name for use with the :data:`SUGGEST` subparser,
    #: taking the minimum desired |fuzzywuzzy|_ match quality
    #: as one required argument
    THRESH = "thresh"

    #: Optional argument name for use with the :data:`SUGGEST` subparser,
    #: indicating to print the location index of each returned object
    #: within :data:`INFILE` along with the object domain/role/name
    #: (may be specified with :data:`SCORE`)
    INDEX = "index"

    #: Optional argument name for use with the :data:`SUGGEST` subparser,
    #: indicating to print the |fuzzywuzzy|_ score of each returned object
    #: within :data:`INFILE` along with the object domain/role/name
    #: (may be specified with :data:`INDEX`)
    SCORE = "score"

    #: Optional argument name for use with the :data:`SUGGEST` subparser,
    #: indicating to print all returned objects, regardless of the
    #: number returned, without asking for confirmation
    ALL = "all"

    #: Optional argument name for use with the :data:`SUGGEST` subparser,
    #: indicating to paginate the suggest subcommand results
    PAGINATE = "paginate"

    # ### Helper strings
    #: Help text for the :data:`CONVERT` subparser
    HELP_CO_PARSER = (
        "Convert intersphinx inventory to zlib-compressed, plaintext, or JSON formats."
    )

    #: Help text for the :data:`SUGGEST` subparser
    HELP_SU_PARSER = "Fuzzy-search intersphinx inventory for desired object(s)."

    #: Help text for default extensions for the various conversion types
    HELP_CONV_EXTS = "'.inv/.txt/.json'"

    # ### Defaults for an unspecified OUTFILE
    #: Default base name for an unspecified :data:`OUTFILE`
    DEF_BASENAME = "objects"

    #: Default extensions for an unspecified :data:`OUTFILE`
    DEF_OUT_EXT = {ZLIB: ".inv", PLAIN: ".txt", JSON: ".json"}

    # ### Useful constants
    #: Number of returned objects from a :data:`SUGGEST` subparser invocation
    #: above which user will be prompted for confirmation to print the results
    #: (unless :data:`ALL` is specified)
    SUGGEST_CONFIRM_LENGTH = 30

    #: Default match threshold for :option:`sphobjinv suggest --thresh`
    DEF_THRESH = 75

    #: Dict key for URL at which an inventory was actually found
    FOUND_URL = "found_url"


def getparser():
    """Generate argument parser.

    Returns
    -------
    prs

        :class:`~argparse.ArgumentParser` -- Parser for commandline usage
        of |soi|

    """
    prs = ap.ArgumentParser(
        description="Format conversion for "
        "and introspection of "
        "intersphinx "
        "'objects.inv' files."
    )
    prs.add_argument(
        "-" + PrsConst.VERSION[0],
        "--" + PrsConst.VERSION,
        help="Print package version & other info",
        action="store_true",
    )

    sprs = prs.add_subparsers(
        title="Subcommands",
        dest=PrsConst.SUBPARSER_NAME,
        metavar=f"{{{PrsConst.CONVERT},{PrsConst.SUGGEST}}}",
        help="Execution mode. Type "
        "'sphobjinv [mode] -h' "
        "for more information "
        "on available options. "
        "Mode names can be abbreviated "
        "to their first two letters.",
    )

    # Enforce subparser as optional. No effect for 3.4 to 3.7;
    # briefly required a/o 3.7.0b4 due to change in default behavior, per:
    # https://bugs.python.org/issue33109. 3.6 behavior restored for
    # 3.7 release.
    #
    # We retain for explicitness
    sprs.required = False

    spr_convert = sprs.add_parser(
        PrsConst.CONVERT,
        aliases=[PrsConst.CONVERT[:2]],
        help=PrsConst.HELP_CO_PARSER,
        description=PrsConst.HELP_CO_PARSER,
    )
    spr_suggest = sprs.add_parser(
        PrsConst.SUGGEST,
        aliases=[PrsConst.SUGGEST[:2]],
        help=PrsConst.HELP_SU_PARSER,
        description=PrsConst.HELP_SU_PARSER,
    )

    # ### Args for conversion subparser
    spr_convert.add_argument(
        PrsConst.MODE,
        help="Conversion output format",
        choices=(PrsConst.ZLIB, PrsConst.PLAIN, PrsConst.JSON),
    )

    spr_convert.add_argument(
        PrsConst.INFILE,
        help=(
            "Path to file to be converted. Passing '-' indicates to read from stdin "
            "(plaintext/JSON only)."
        ),
    )

    spr_convert.add_argument(
        PrsConst.OUTFILE,
        help=(
            "Path to desired output file. "
            "Defaults to same directory and main "
            "file name as input file but with extension "
            + PrsConst.HELP_CONV_EXTS
            + ", as appropriate for the output format. "
            "A path to a directory is accepted here, "
            "in which case the default output file name will be used. "
            "Passing '-' indicates to write to stdout. If "
            + PrsConst.INFILE
            + " is passed as '-', "
            + PrsConst.OUTFILE
            + " can be omitted and both stdin and stdout will be used."
        ),
        nargs="?",
        default=None,
    )

    # Mutually exclusive group for --expand/--contract
    gp_expcont = spr_convert.add_argument_group(title="URI/display name conversions")
    meg_expcont = gp_expcont.add_mutually_exclusive_group()
    meg_expcont.add_argument(
        "-" + PrsConst.EXPAND[0],
        "--" + PrsConst.EXPAND,
        help="Expand all URI and display name abbreviations",
        action="store_true",
    )

    meg_expcont.add_argument(
        "-" + PrsConst.CONTRACT[0],
        "--" + PrsConst.CONTRACT,
        help="Contract all URI and display name abbreviations",
        action="store_true",
    )

    # Clobber argument
    spr_convert.add_argument(
        "-" + PrsConst.OVERWRITE[0],
        "--" + PrsConst.OVERWRITE,
        help="Overwrite output files without prompting",
        action="store_true",
    )

    # stdout suppressor option (e.g., for scripting)
    spr_convert.add_argument(
        "-" + PrsConst.QUIET[0],
        "--" + PrsConst.QUIET,
        help="Suppress printing of status messages and "
        "overwrite output files without prompting",
        action="store_true",
    )

    # Flag to treat infile as a URL
    spr_convert.add_argument(
        "-" + PrsConst.URL[0],
        "--" + PrsConst.URL,
        help=(
            "Treat 'infile' as a URL for download. "
            "Cannot be used with an infile of '-'."
        ),
        action="store_true",
    )

    # ### Args for suggest subparser
    spr_suggest.add_argument(
        PrsConst.INFILE,
        help=(
            "Path to inventory file to be searched. "
            "Passing '-' indicates to read from stdin (plaintext/JSON only)."
        ),
    )
    spr_suggest.add_argument(PrsConst.SEARCH, help="Search term for object suggestions")
    spr_suggest.add_argument(
        "-" + PrsConst.ALL[0],
        "--" + PrsConst.ALL,
        help="Display all results "
        "regardless of the number returned "
        "without prompting for confirmation.",
        action="store_true",
    )
    spr_suggest.add_argument(
        "-" + PrsConst.PAGINATE[0],
        "--" + PrsConst.PAGINATE,
        help="Paginate long search results",
        action="store_true",
    )
    spr_suggest.add_argument(
        "-" + PrsConst.INDEX[0],
        "--" + PrsConst.INDEX,
        help="Include Inventory.objects list indices with the search results",
        action="store_true",
    )
    spr_suggest.add_argument(
        "-" + PrsConst.SCORE[0],
        "--" + PrsConst.SCORE,
        help="Include fuzzywuzzy scores with the search results",
        action="store_true",
    )
    spr_suggest.add_argument(
        "-" + PrsConst.THRESH[0],
        "--" + PrsConst.THRESH,
        help="Match quality threshold, integer 0-100, "
        "default 75. Default is suitable when "
        "'search' is exactly a known object name. "
        "A value of 30-50 gives better results "
        "for approximate matches.",
        default=PrsConst.DEF_THRESH,
        type=int,
        choices=range(101),
        metavar="{0-100}",
    )
    spr_suggest.add_argument(
        "-" + PrsConst.URL[0],
        "--" + PrsConst.URL,
        help=(
            "Treat 'infile' as a URL for download. "
            f"Cannot be used with --{PrsConst.URL}."
        ),
        action="store_true",
    )

    return prs


def getparser_textconv():
    """Generate argument parser for textconv entrypoint.

    Returns
    -------
    prs

        :class:`~argparse.ArgumentParser` -- Parser for textconv commandline
        usage of |soi|

    """
    prs = ap.ArgumentParser(
        description=(
            "Emit the plaintext of the local Sphinx inventory at 'infile' to stdout."
        )
    )
    prs.add_argument(
        "-" + PrsConst.VERSION[0],
        "--" + PrsConst.VERSION,
        help="Print package version & other info",
        action="version",
        version=PrsConst.VER_TXT_SHORT,
    )

    prs.add_argument(
        PrsConst.INFILE,
        help=("Path to file to be converted"),
    )

    return prs
