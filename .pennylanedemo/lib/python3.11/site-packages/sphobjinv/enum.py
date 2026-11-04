r"""*Helper enums for* ``sphobjinv``.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    4 May 2019

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

from enum import Enum


class SourceTypes(Enum):
    """|Enum| for the import mode used in instantiating an |Inventory|.

    Since |Enum| keys iterate in definition order, the
    definition order here defines the order in which |Inventory|
    objects attempt to parse a source object passed to
    :class:`Inventory.__init__() <sphobjinv.inventory.Inventory>`
    either as a positional argument
    or via the generic `source` keyword argument.

    This order **DIFFERS** from the documentation order, which is
    alphabetical.

    """

    #: No source; |Inventory| was instantiated with
    #: :data:`~sphobjinv.inventory.Inventory.project`
    #: and :data:`~sphobjinv.inventory.Inventory.version`
    #: as empty strings and
    #: :data:`~sphobjinv.inventory.Inventory.objects` as an empty |list|.
    Manual = "manual"

    #: Instantiation from a plaintext |objects.inv| |bytes|.
    BytesPlaintext = "bytes_plain"

    #: Instantiation from a zlib-compressed
    #: |objects.inv| |bytes|.
    BytesZlib = "bytes_zlib"

    #: Instantiation from a plaintext |objects.inv| file on disk.
    FnamePlaintext = "fname_plain"

    #: Instantiation from a zlib-compressed |objects.inv| file on disk.
    FnameZlib = "fname_zlib"

    #: Instantiation from a |dict| validated against
    #: :data:`schema.json_schema <sphobjinv.schema.json_schema>`.
    DictJSON = "dict_json"

    #: Instantiation from a zlib-compressed |objects.inv| file
    #: downloaded from a URL.
    URL = "url"


class HeaderFields(Enum):
    """|Enum| for various inventory-level data items.

    A subset of these |Enum| values is used in various Regex,
    JSON, and string formatting contexts within
    class:`~sphobjinv.inventory.Inventory`
    and :data:`schema.json_schema <sphobjinv.schema.json_schema>`.

    """

    #: Project name associated with an inventory
    Project = "project"

    #: Project version associated with an inventory
    Version = "version"

    #: Number of objects contained in the inventory
    Count = "count"

    #: The |str| value of this |Enum| member is accepted as a root-level
    #: key in a |dict| to be imported into an
    #: :class:`~sphobjinv.inventory.Inventory`.
    #: The corresponding value in the |dict| may contain any arbitrary data.
    #: Its possible presence is accounted for in
    #: :data:`schema.json_schema <sphobjinv.schema.json_schema>`.
    #:
    #: The data associated with this key are **ignored**
    #: during import into an
    #: :class:`~sphobjinv.inventory.Inventory`.
    Metadata = "metadata"
