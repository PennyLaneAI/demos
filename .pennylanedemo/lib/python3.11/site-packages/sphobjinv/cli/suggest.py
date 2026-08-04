r"""``sphobjinv`` *module for CLI suggest functionality*.

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

import itertools as itt
import shutil
import sys
import urllib.parse as urlparse

from sphobjinv.cli.parser import PrsConst
from sphobjinv.cli.ui import print_stderr, yesno_prompt


def do_suggest(inv, params):
    r"""Perform the suggest call and output the results.

    Results are printed one per line.

    If neither INDEX nor SCORE is specified,
    the results are output without a header.
    If either or both are specified,
    the results are output in a lightweight tabular format.

    If the number of results exceeds
    SUGGEST_CONFIRM_LENGTH,
    the user will be queried whether to display
    all of the returned results
    unless ALL is specified.

    No QUIET option is available here, since
    a silent mode for suggestion output is nonsensical.

    Parameters
    ----------
    inv

        |Inventory| -- Inventory object to be output in the format
        indicated by MODE.

    params

        |dict| -- Parameters/values mapping from the active subparser

    """
    with_index = params[PrsConst.INDEX]
    with_score = params[PrsConst.SCORE]
    results = inv.suggest(
        params[PrsConst.SEARCH],
        thresh=params[PrsConst.THRESH],
        with_index=with_index,
        with_score=with_score,
    )

    print_divider(params)
    print_stderr_inferred_mapping(params)
    print_divider(params)

    print_stderr(f"Project: {inv.project}", params)
    print_stderr(f"Version: {inv.version}\n", params)

    print_stderr(f"{inv.count} objects in inventory.\n", params)
    print_divider(params)

    print_stderr_result_count(params, results)

    if not results:
        print_stderr("\nExiting...\n", params)
        sys.exit(0)

    # The query here for printing the full list only occurs in some
    # circumstances; see the function docstring.
    confirm_print_if_long_list(params, results)

    print_results_table(with_index, with_score, results, params)


def print_divider(params):
    """Print a visual divider to break up sections of the CLI output."""
    length = shutil.get_terminal_size().columns * 3 // 5
    print_stderr("-" * length + "\n", params)


def print_stderr_result_count(params, results):
    """Report the count of found objects from the suggest search."""
    if len(results) == 0:
        print_stderr(
            (
                "No results found with score at/above current threshold of "
                f"{params[PrsConst.THRESH]}."
            ),
            params,
        )
    else:
        print_stderr(
            (
                f"{len(results)} result"
                f"{'' if len(results) == 1 else 's'}"
                " found at/above current threshold of "
                f"{params[PrsConst.THRESH]}.\n"
            ),
            params,
        )


def confirm_print_if_long_list(params, results):
    """Check if the results list is too long and query user if to proceed.

    Skip the check if ``--all`` has been passed.

    Also skip the check if receiving data from ``stdin``, as a stream
    interaction here fouls the data flow...I forget exactly how.

    """
    if (
        len(results) > PrsConst.SUGGEST_CONFIRM_LENGTH
        and not params[PrsConst.ALL]
        and params[PrsConst.INFILE] != "-"
    ):
        resp = yesno_prompt(f"Display all {len(results)} results (Y/N)?")
        if resp.lower() == "n":
            print_stderr("\nExiting...", params)
            sys.exit(0)

        # Extra cosmetic newline
        print_stderr("", params)


def print_results_table(with_index, with_score, results, params):
    """Prepare and print the table of suggest search results."""
    # Field widths in output
    score_width = 7
    index_width = 7

    # This is necessary because the results are returned as a
    # list[str] if neither index nor score is included; but are
    # returned as a list[tuple[...]] if one or both of index/score
    # is included.
    if with_index or with_score:
        rst_width = max(len(res[0]) for res in results)
    else:
        rst_width = max(len(res) for res in results)

    rst_width += 2

    # For now, in each case the formatting for each row is dynamically
    # stored in `fmt`, and then `fmt` is used to actually format each row.

    # TODO: Consider replacing this with a *real* table formatting tool
    if with_index:
        if with_score:
            gen = generate_score_index_lines(
                results, score_width, index_width, rst_width
            )
        else:
            gen = generate_index_lines(results, index_width, rst_width)
    else:
        if with_score:
            gen = generate_score_lines(results, score_width, rst_width)
        else:
            gen = generate_names_only_lines(results)

    if not params[PrsConst.PAGINATE]:
        print("\n".join(gen))
    else:
        # To make sure the initial output is not scrolled off the screen
        # when --all is specified.
        if params[PrsConst.ALL]:
            input("Press Enter to continue...")

        while True:
            # Adjust the number of lines per page if the user changes their
            # terminal window size mid-execution
            n_lines = shutil.get_terminal_size().lines - 2
            out_text = "\n".join(itt.islice(gen, n_lines))

            if out_text:
                print(out_text)
                # Don't paginate after the last, partial screenful of output
                if out_text.count("\n") == n_lines - 1:
                    input("Press Enter to continue...")
            else:
                # join() above will supply an empty string once gen is exhausted
                break


def generate_score_index_lines(results, score_width, index_width, rst_width):
    """Yield lines to print the table with scores & indices."""
    fmt = (
        f"{{name: <{rst_width}}}  {{score: ^{score_width}}}  {{index: ^{index_width}}}"
    )
    yield fmt.format(name="  Name", score="Score", index="Index")
    yield fmt.format(
        name=("-" * rst_width), score=("-" * score_width), index=("-" * index_width)
    )
    yield from (fmt.format(name=res[0], score=res[1], index=res[2]) for res in results)


def generate_index_lines(results, index_width, rst_width):
    """Yield lines to print the table with indices."""
    fmt = f"{{name: <{rst_width}}}  {{index: ^{index_width}}}"
    yield fmt.format(name="  Name", index="Index")
    yield fmt.format(name=("-" * rst_width), index=("-" * index_width))
    yield from (fmt.format(name=res[0], index=res[1]) for res in results)


def generate_score_lines(results, score_width, rst_width):
    """Yield lines to print the table with scores."""
    fmt = f"{{name: <{rst_width}}}  {{score: ^{score_width}}}"
    yield fmt.format(name="  Name", score="Score")
    yield fmt.format(name=("-" * rst_width), score=("-" * score_width))
    yield from (fmt.format(name=res[0], score=res[1]) for res in results)


def generate_names_only_lines(results):
    """Yield lines to print containing just the object search results."""
    yield from (str(res) for res in results)


def print_stderr_inferred_mapping(params):
    """Print as good of an `intersphinx_mapping` entry as can be determined."""
    if not params[PrsConst.URL]:
        print_stderr(
            "Cannot infer intersphinx_mapping from a local objects.inv.\n", params
        )
        return

    input_url = params[PrsConst.INFILE]
    inv_url = params[PrsConst.FOUND_URL]

    reduced_inv_url = extract_objectsinv_url_base(inv_url)

    if input_url == inv_url:
        # User provided an exact URL to an inventory
        # The tail of input_url thus should not match anything in the inventory,
        # so we can't say anything about what the URL base of the docset is.
        if inv_url == reduced_inv_url:
            # The inventory URL does not end with the standard /objects.inv,
            # so we don't even have *that* point of reference to work with.
            print_stderr(
                (
                    "Cannot infer intersphinx_mapping for this docset using "
                    "the provided input URL.\n"
                ),
                params,
            )
        else:
            # The inventory URL *does* end with standard /objects.inv,
            # so there's a reasonable chance we know the mapping.
            print_stderr(
                (
                    "The intersphinx_mapping for this docset is PROBABLY:\n\n"
                    f"  ({reduced_inv_url}, None)\n"
                ),
                params,
            )
    else:
        # User provided a URL from the docset that reduced to a base, atop which
        # an inventory at .../objects.inv was found.
        if inv_url == reduced_inv_url:
            # This should never happen, because the only way we should be in
            # this outer else is if inv_url *DOES* end in /objects.inv
            print_stderr(  # pragma: no cover
                (
                    "ERROR: Inconsistent internal state "
                    "during intersphinx_mapping inference.\n"
                ),
                params,
            )
        else:
            # Here we're *very* confident that we've go the mapping.
            print_stderr(
                (
                    "The intersphinx_mapping for this docset is LIKELY:\n\n"
                    f"  ({reduced_inv_url}, None)\n"
                ),
                params,
            )


def extract_objectsinv_url_base(objectsinv_url):
    """Infer a base URL for the provided ``objects.inv`` inventory URL.

    If this function is a no-op, then the resulting base is NOT RELIABLE,
    because the URL did not end with ``/objects.inv``.

    If this function *does* make a change, then the resulting base is
    RELATIVELY RELIABLE, since the only change that should occur is
    stripping of a ``/objects.inv`` suffix, which strongly implies but
    does not guarantee that the URL came from a Sphinx docset in the
    standard multi-page HTML layout.

    Parameters
    ----------
    objectsinv_url

        |str| -- URL from which to attempt docset base inference

    Returns
    -------
    trimmed

        |str| -- URL after attempt to trim a trailing ``/objects.inv``

    """
    trimmed = _strip_url_to_netloc_path(objectsinv_url, with_scheme=True)
    base = trimmed.rpartition("/objects.inv")[0]

    if base:
        return f"{base}/"
    else:
        return objectsinv_url


def _strip_url_to_netloc_path(url, *, with_scheme=False):
    """Reduce a URL to only netloc and path, optionally with scheme."""
    parts = urlparse.urlsplit(url)
    trimmed = parts._replace(
        query="",
        fragment="",
    )

    if not with_scheme:
        trimmed = trimmed._replace(scheme="")

    return urlparse.urlunsplit(trimmed)
