r"""*Custom errors for* ``sphobjinv``.

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


class SphobjinvError(Exception):
    """Custom ``sphobjinv`` error superclass."""


class VersionError(SphobjinvError):
    """Raised when attempting an operation on an unsupported version.

    The current version of ``sphobjinv`` only supports 'version 2'
    |objects.inv| files (see :doc:`here </syntax>`).

    """

    # TODO: Add SOI prefix to this class name as part of the exceptions refactor
