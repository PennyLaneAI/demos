r"""*Module for* ``sphobjinv`` *CLI* |Inventory| *loading*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    17 Nov 2020

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
import sys
from json import JSONDecodeError
from urllib.error import HTTPError, URLError

from jsonschema.exceptions import ValidationError

from sphobjinv import Inventory, VersionError, readjson, urlwalk
from sphobjinv.cli.parser import PrsConst
from sphobjinv.cli.paths import resolve_inpath
from sphobjinv.cli.ui import err_format, print_stderr


def import_infile(in_path):
    """Attempt import of indicated file.

    Convenience function wrapping attempts to load an
    |Inventory| from a local path.

    Parameters
    ----------
    in_path

        |str| -- Path to input file

    Returns
    -------
    inv

        |Inventory| or |None| -- If instantiation with the file at
        `in_path` succeeds, the resulting |Inventory| instance;
        otherwise, |None|

    """
    # Try general import, for zlib or plaintext files
    try:
        inv = Inventory(in_path)
    except AttributeError:
        pass  # Punt to JSON attempt
    else:
        return inv

    # Maybe it's JSON
    try:
        inv = Inventory(readjson(in_path))
    except JSONDecodeError:
        return None
    else:
        return inv


def inv_local(params):
    """Create |Inventory| from local source.

    Uses ``resolve_inpath`` to sanity-check and/or convert
    INFILE.

    Calls :func:`sys.exit` internally in error-exit situations.

    Parameters
    ----------
    params

        |dict| -- Parameters/values mapping from the active subparser

    Returns
    -------
    inv

        |Inventory| -- Object representation of the inventory
        at INFILE

    in_path

        |str| -- Input file path as resolved/checked by
        ``resolve_inpath``

    """
    # Resolve input file path
    try:
        in_path = resolve_inpath(params[PrsConst.INFILE])
    except Exception as e:
        print_stderr("\nError while parsing input file path:", params)
        print_stderr(err_format(e), params)
        sys.exit(1)

    # Attempt import
    inv = import_infile(in_path)
    if inv is None:
        print_stderr("\nError: Unrecognized file format", params)
        sys.exit(1)

    return inv, in_path


def inv_url(params):
    """Create |Inventory| from file downloaded from URL.

    Initially, treats INFILE as a download URL to be passed to
    the `url` initialization argument
    of :class:`~sphobjinv.inventory.Inventory`.

    If an inventory is not found at that exact URL, progressively
    searches the directory tree of the URL for |objects.inv|.

    Injects the URL at which an inventory was found into `params`
    under the FOUND_URL key.

    Calls :func:`sys.exit` internally in error-exit situations.

    Parameters
    ----------
    params

        |dict| -- Parameters/values mapping from the active subparser

    Returns
    -------
    inv

        |Inventory| -- Object representation of the inventory
        at INFILE

    ret_path

        |str| -- URL from INFILE used to construct `inv`.
        If URL is longer than 45 characters, the central portion is elided.

    """
    in_file = params[PrsConst.INFILE]

    def attempt_inv_load(url, params):
        """Attempt the Inventory load and report outcome."""
        inv = None

        try:
            inv = Inventory(url=url)
        except HTTPError as e:
            print_stderr(f"  ... HTTP error: {e.code} {e.reason}.", params)
        except URLError:  # pragma: no cover
            print_stderr("  ... error attempting to retrieve URL.", params)
        except VersionError:  # pragma: no cover
            print_stderr("  ... no recognized inventory.", params)
        except ValueError:
            print_stderr(
                (
                    "  ... file found but inventory could not be loaded. "
                    "(Did you forget https:// ?)"
                ),
                params,
            )
        else:
            print_stderr("  ... inventory found.", params)

        return inv

    # Disallow --url mode on local files
    if in_file.startswith("file:/"):
        print_stderr("\nError: URL mode on local file is invalid", params)
        sys.exit(1)

    print_stderr(f"Attempting {in_file} ...", params)
    inv = attempt_inv_load(in_file, params)

    if inv:
        url = in_file
    else:
        for url in urlwalk(in_file):
            print_stderr(f'Attempting "{url}" ...', params)
            inv = attempt_inv_load(url, params)
            if inv:
                break

    # Cosmetic line break
    print_stderr(" ", params)

    # Success or no?
    if not inv:
        print_stderr("No inventory found!", params)
        sys.exit(1)

    params.update({PrsConst.FOUND_URL: url})
    if len(url) > 45:
        ret_path = url[:20] + "[...]" + url[-20:]
    else:  # pragma: no cover
        ret_path = url

    return inv, ret_path


def inv_stdin(params):
    """Create |Inventory| from contents of stdin.

    Due to stdin's encoding and formatting assumptions, only
    text-based inventory formats can be sanely parsed.

    Thus, only plaintext and JSON inventory formats can be
    used as inputs here.

    Parameters
    ----------
    params

        |dict| -- Parameters/values mapping from the active subparser

    Returns
    -------
    inv

        |Inventory| -- Object representation of the inventory
        provided at stdin

    """
    data = sys.stdin.read()

    try:
        return Inventory(dict_json=json.loads(data))
    except (JSONDecodeError, ValidationError):
        pass

    try:
        return Inventory(plaintext=data)
    except (AttributeError, UnicodeEncodeError, TypeError):
        pass

    print_stderr("Invalid plaintext or JSON inventory format.", params)
    sys.exit(1)
