r"""``sphobjinv`` *package definition module*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    17 May 2016

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

from sphobjinv.data import DataFields, DataObjBytes, DataObjStr
from sphobjinv.enum import HeaderFields, SourceTypes
from sphobjinv.error import SphobjinvError, VersionError
from sphobjinv.fileops import readbytes, readjson, urlwalk, writebytes, writejson
from sphobjinv.inventory import Inventory
from sphobjinv.re import p_data, pb_comments, pb_data, pb_project, pb_version
from sphobjinv.schema import json_schema
from sphobjinv.version import __version__
from sphobjinv.zlib import compress, decompress
