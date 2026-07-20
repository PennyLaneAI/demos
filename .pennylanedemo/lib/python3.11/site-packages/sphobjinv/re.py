r"""*Helper regexes for* ``sphobjinv``.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    5 Nov 2017

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

import re

from sphobjinv.data import DataFields as DF  # noqa: N817
from sphobjinv.enum import HeaderFields as HF  # noqa: N817

#: Compiled |re| |bytes|  pattern for comment lines in decompressed
#: inventory files
pb_comments = re.compile(b"^#.*$", re.M)

#: Compiled |re| |bytes| pattern for project line
pb_project = re.compile(
    rf"""
    ^                            # Start of line
    [#][ ]Project:[ ]            # Preamble
    (?P<{HF.Project.value}>.*?)  # Lazy rest of line is project name
    \r?$                         # Ignore possible CR at EOL
    """.encode(encoding="utf-8"),
    re.M | re.X,
)

#: Compiled |re| |bytes| pattern for version line
pb_version = re.compile(
    rf"""
    ^                            # Start of line
    [#][ ]Version:[ ]            # Preamble
    (?P<{HF.Version.value}>.*?)  # Lazy rest of line is version
    \r?$                         # Ignore possible CR at EOL
    """.encode(encoding="utf-8"),
    re.M | re.X,
)

#: Regex pattern string used to compile
#: :data:`~sphobjinv.re.p_data` and
#: :data:`~sphobjinv.re.pb_data`
ptn_data = rf"""
    ^                               # Start of line
    (?P<{DF.Name.value}>.+?)        # --> Name
    \s+                             # Dividing space
    (?P<{DF.Domain.value}>[^\s:]+)  # --> Domain
    :                               # Dividing colon
    (?P<{DF.Role.value}>[^\s]+)     # --> Role
    \s+                             # Dividing space
    (?P<{DF.Priority.value}>-?\d+)  # --> Priority
    \s+?                            # Dividing space
    (?P<{DF.URI.value}>\S*)         # --> URI
    \s+                             # Dividing space
    (?P<{DF.DispName.value}>.+?)    # --> Display name, lazy b/c possible CR
    \r?$                            # Ignore possible CR at EOL
    """

#: Compiled |re| |bytes| regex pattern for data lines in |bytes| decompressed
#: inventory files
pb_data = re.compile(ptn_data.encode(encoding="utf-8"), re.M | re.X)

#: Compiled |re| |str| regex pattern for data lines in |str| decompressed
#: inventory files
p_data = re.compile(ptn_data, re.M | re.X)
