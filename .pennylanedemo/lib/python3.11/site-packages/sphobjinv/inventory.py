r"""``sphobjinv`` *data class for full inventories*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    7 Dec 2017

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
import ssl
import urllib.request as urlrq
from zlib import error as zlib_error

import attr
import certifi
import jsonschema
from jsonschema.exceptions import ValidationError

from sphobjinv.data import DataObjStr, _utf8_encode
from sphobjinv.enum import HeaderFields, SourceTypes
from sphobjinv.fileops import readbytes
from sphobjinv.re import pb_data, pb_project, pb_version
from sphobjinv.schema import json_schema
from sphobjinv.version import __version__ as soi_version
from sphobjinv.zlib import decompress


@attr.s(slots=True, eq=True, order=False)
class Inventory:
    r"""Entire contents of an |objects.inv| inventory.

    All information is stored internally as |str|,
    even if imported from a |bytes| source.

    All arguments except `count_error` are used to specify the source
    from which the |Inventory| contents are to be populated.
    **At most ONE** of these source arguments may be other than |None|.

    The `count_error` argument is only relevant to the `dict_json` source type.

    Equality comparisons between |Inventory| instances
    will return |True| if
    :attr:`~sphobjinv.inventory.Inventory.project`,
    :attr:`~sphobjinv.inventory.Inventory.version`, and
    **all** contents of :attr:`~sphobjinv.inventory.Inventory.objects`
    are identical, even if the instances were created from different
    sources:

    .. doctest:: inventory-equality

        >>> inv1 = soi.Inventory(
        ...     url="https://sphobjinv.readthedocs.io/en/latest/objects.inv"
        ... )
        >>> inv2 = soi.Inventory(inv1.data_file())
        >>> inv1 is inv2
        False
        >>> inv1 == inv2
        True

    .. versionchanged:: 2.1
        Previously, an |Inventory| instance would compare equal only
        to itself.

    `source`

        The |Inventory| will attempt to parse the indicated
        source object as each of the below types in sequence,
        **except** for `url`.

        This argument is included mainly as a convenience
        feature for use in interactive sessions, as
        invocations of the following form implicitly
        populate `source`, as the first positional argument::

            >>> inv = Inventory(src_obj)

        In most cases, for clarity it is recommended
        that programmatic instantiation of |Inventory| objects
        utilize the below format-specific arguments.

    `plaintext`

        Object is to be parsed as the UTF-8 |bytes|
        plaintext contents of an |objects.inv| inventory.

    `zlib`

        Object is to be parsed as the UTF-8 |bytes|
        zlib-compressed contents of an
        |objects.inv| inventory.

    `fname_plain`

        Object is the |str| or |Path| path to a file containing
        the plaintext contents of an |objects.inv| inventory.

        .. versionchanged:: 2.1

            Previously, this argument could only be a |str|.

    `fname_zlib`

        Object is the |str| or |Path| path to a file containing
        the zlib-compressed contents of an
        |objects.inv| inventory.

        .. versionchanged:: 2.1

            Previously, this argument could only be a |str|.

    `dict_json`

        Object is a |dict| containing the contents
        of an |objects.inv| inventory, conforming to
        the JSON schema of
        :data:`schema.json_schema <sphobjinv.schema.json_schema>`.

        If `count_error` is passed as |True|,
        then a :exc:`ValueError` is raised if
        the number of objects found in the |dict|
        does not match the value associated with
        its `count` key.
        If `count_error` is passed as |False|,
        an object count mismatch is ignored.

    `url`

        Object is a |str| URL to a zlib-compressed
        |objects.inv| file.
        Any URL type supported by
        :mod:`urllib.request` SHOULD work; only
        |cour|\ http:\ |/cour| and
        |cour|\ file:\ |/cour|
        have been directly tested, however.

        No authentication is supported at this time.

    **Members**

    """

    # General source for try-most-types import
    # Needs to be first so it absorbs a positional arg
    _source = attr.ib(repr=False, default=None, eq=False)

    # Stringlike types (both accept str & bytes)
    _plaintext = attr.ib(repr=False, default=None, eq=False)
    _zlib = attr.ib(repr=False, default=None, eq=False)

    # Filename types (must be str or Path)
    _fname_plain = attr.ib(repr=False, default=None, eq=False)
    _fname_zlib = attr.ib(repr=False, default=None, eq=False)

    # dict types
    _dict_json = attr.ib(repr=False, default=None, eq=False)

    # URL for remote retrieval of objects.inv/.txt
    _url = attr.ib(repr=False, default=None, eq=False)

    # Flag for whether to raise error on object count mismatch
    _count_error = attr.ib(
        repr=False, default=True, validator=attr.validators.instance_of(bool), eq=False
    )

    # Actual regular attributes
    #: |str| project display name for the inventory
    #: (see :ref:`here <syntax-mouseover-example>`).
    project = attr.ib(init=False, default=None)

    #: |str| project display version for the inventory
    #: (see :ref:`here <syntax-mouseover-example>`).
    version = attr.ib(init=False, default=None)

    #: |list| of |DataObjStr| representing the
    #: data objects of the inventory.
    #: Can be edited directly to change the inventory contents.
    #: Undefined/random behavior/errors will result if the type
    #: of the elements is anything other than |DataObjStr|.
    objects = attr.ib(init=False, default=attr.Factory(list), repr=False)

    #: :class:`~sphobjinv.enum.SourceTypes` |Enum| value indicating the type of
    #: source from which the instance was generated.
    source_type = attr.ib(init=False, default=None, eq=False)

    # Helper strings for inventory datafile output
    #: Preamble line for v2 |objects.inv| header
    header_preamble = "# Sphinx inventory version 2"

    #: Project line |str.format| template for |objects.inv| header
    header_project = "# Project: {project}"

    #: Version line |str.format| template for |objects.inv| header
    header_version = "# Version: {version}"

    #: zlib compression line for v2 |objects.inv| header
    header_zlib = "# The remainder of this file is compressed using zlib."

    # Private class member for SSL context, since context creation is slow(?)
    _sslcontext = ssl.create_default_context(cafile=certifi.where())

    @property
    def count(self):
        """Count of objects currently in inventory."""
        return len(self.objects)

    def json_dict(self, expand=False, contract=False):
        """Generate a flat |dict| representation of the inventory.

        The returned |dict| matches the
        schema of :data:`sphobjinv.schema.json_schema`.

        Calling with both `expand` and `contract` as |True| is invalid.

        Parameters
        ----------
        expand

            |bool| *(optional)* -- Return |dict| with any
            :data:`~sphobjinv.data.SuperDataObj.uri` or
            :data:`~sphobjinv.data.SuperDataObj.dispname`
            abbreviations expanded

        contract

            |bool| *(optional)* -- Return |dict| with abbreviated
            :data:`~sphobjinv.data.SuperDataObj.uri` and
            :data:`~sphobjinv.data.SuperDataObj.dispname` values

        Returns
        -------
        d

            |dict| -- Inventory data; keys and values are all |str|

        Raises
        ------
        ValueError

            If both `expand` and `contract` are |True|

        """
        d = {
            HeaderFields.Project.value: self.project,
            HeaderFields.Version.value: self.version,
            HeaderFields.Count.value: self.count,
        }

        for i, o in enumerate(self.objects):
            d.update({str(i): o.json_dict(expand=expand, contract=contract)})

        return d

    @property
    def objects_rst(self):
        r"""|list| of objects formatted in a |str| reST-like representation.

        The format of each |str| in the |list| is given by
        :class:`data.SuperDataObj.rst_fmt
        <sphobjinv.data.SuperDataObj.rst_fmt>`.

        Returns
        -------
        obj_l

            |list| of |str| -- Inventory object data in reST-like format

        """
        return [_.as_rst for _ in self.objects]

    def __str__(self):  # pragma: no cover
        """Return concise, readable description of contents."""
        ret_str = "<{0} ({1}): {2} {3}, {4} objects>"

        proj = self.project if self.project else "<no project>"
        ver = "v" + self.version if self.version else "<no version>"

        return ret_str.format(
            type(self).__name__, self.source_type.value, proj, ver, self.count
        )

    def __attrs_post_init__(self):
        """Construct the inventory from the indicated source."""
        # List of sources
        src_list = (
            self._source,
            self._plaintext,
            self._zlib,
            self._fname_plain,
            self._fname_zlib,
            self._dict_json,
            self._url,
        )
        src_count = sum(1 for _ in src_list if _ is not None)

        # Complain if multiple sources provided
        if src_count > 1:
            raise RuntimeError("At most one data source can be specified.")

        # Leave uninitialized ("manual" init) if no source provided
        if src_count == 0:
            self.source_type = SourceTypes.Manual
            return

        # If general ._source was provided, run the generalized import
        if self._source is not None:
            self._general_import()
            return

        # For all of these below, '()' is passed as 'exc' argument since
        # desire _try_import not to handle any exception types

        # Plaintext str or bytes
        # Special case, since preconverting input.
        if self._plaintext is not None:
            self._try_import(
                self._import_plaintext_bytes, _utf8_encode(self._plaintext), ()
            )
            self.source_type = SourceTypes.BytesPlaintext
            return

        # Remainder are iterable
        for src, fxn, st in zip(
            (
                self._zlib,
                self._fname_plain,
                self._fname_zlib,
                self._dict_json,
                self._url,
            ),
            (
                self._import_zlib_bytes,
                self._import_plaintext_fname,
                self._import_zlib_fname,
                self._import_json_dict,
                self._import_url,
            ),
            (
                SourceTypes.BytesZlib,
                SourceTypes.FnamePlaintext,
                SourceTypes.FnameZlib,
                SourceTypes.DictJSON,
                SourceTypes.URL,
            ),
        ):
            if src is not None:
                self._try_import(fxn, src, ())
                self.source_type = st
                return

    def data_file(self, *, expand=False, contract=False):
        """Generate a plaintext |objects.inv| as UTF-8 |bytes|.

        |bytes| is used here as the output type
        since the most common use cases are anticipated to be
        either (1) dumping to file via :func:`sphobjinv.fileops.writebytes`
        or (2) compressing via :func:`sphobjinv.zlib.compress`,
        both of which take |bytes| input.

        Calling with both `expand` and `contract` as |True| is invalid.

        Parameters
        ----------
        expand

            |bool| *(optional)* -- Generate |bytes| with any
            :data:`~sphobjinv.data.SuperDataObj.uri` or
            :data:`~sphobjinv.data.SuperDataObj.dispname`
            abbreviations expanded

        contract

            |bool| *(optional)* -- Generate |bytes| with abbreviated
            :data:`~sphobjinv.data.SuperDataObj.uri` and
            :data:`~sphobjinv.data.SuperDataObj.dispname` values

        Returns
        -------
        b

            |bytes| -- Inventory in plaintext |objects.inv| format

        Raises
        ------
        ValueError

            If both `expand` and `contract` are |True|

        """
        # Rely on SuperDataObj to proof expand/contract args
        # Extra empty string at the end puts a newline at the end
        # of the generated string, consistent with files
        # generated by Sphinx.
        return "\n".join(
            (
                self.header_preamble,
                self.header_project.format(project=self.project),
                self.header_version.format(version=self.version),
                self.header_zlib,
                *(
                    obj.data_line(expand=expand, contract=contract)
                    for obj in self.objects
                ),
                "",
            )
        ).encode("utf-8")

    def suggest(self, name, *, thresh=50, with_index=False, with_score=False):
        r"""Suggest objects in the inventory to match a name.

        :meth:`~Inventory.suggest` makes use of
        the edit-distance scoring library |fuzzywuzzy|_
        to identify potential matches to the given `name`
        within the inventory.
        The search is performed over the |list| of |str|
        generated by :meth:`~objects_rst`.

        `thresh` defines the minimum |fuzzywuzzy|_ match quality
        (an integer ranging from 0 to 100)
        required for a given object to be included
        in the results list.
        Can be any float value,
        but best results are generally obtained
        with values between 50 and 80,
        depending on the number of objects in the inventory,
        the confidence of the user in the match
        between `name` and the object(s) of interest,
        and the desired fidelity of the search results to `name`.

        This functionality is provided by the
        :doc:`'suggest' subparser </cli/suggest>`
        of the command-line interface.

        Parameters
        ----------
        name

            |str| -- Object name for |fuzzywuzzy|_ pattern matching

        thresh

            |float| -- |fuzzywuzzy|_ match quality threshold

        with_index

            |bool| -- Include with each matched name
            its index within :attr:`Inventory.objects`

        with_score

            |bool| -- Include with each matched name
            its |fuzzywuzzy|_ match quality score

        Returns
        -------
        res_l

            |list|

            If both `with_index` and `with_score`
            are |False|, members are the |str|
            :meth:`SuperDataObj.as_rst()
            <sphobjinv.data.SuperDataObj.as_rst>`
            representations of matching objects.

            If either is |True|, members are |tuple|\ s
            of the indicated match results:

            `with_index == True`: |cour|\ (as_rst, index)\ |/cour|

            `with_score == True`: |cour|\ (as_rst, score)\ |/cour|

            `with_index == with_score == True`:
            |cour|\ (as_rst, score, index)\ |/cour|

        """
        from sphobjinv._vendored.fuzzywuzzy import process as fwp

        # Must propagate list index to include in output
        # Search vals are rst prepended with list index
        srch_list = [f"{i} {o}" for i, o in enumerate(self.objects_rst)]

        # Composite each string result extracted by fuzzywuzzy
        # and its match score into a single string. The match
        # and score are returned together in a tuple.
        results = [
            f"{match} {score}"
            for match, score in fwp.extract(name, srch_list, limit=None)
            if score >= thresh
        ]

        # Define regex for splitting the three components, and
        # use it to convert composite result string to tuple:
        # result --> (rst, score, index)
        p_idx = re.compile(r"^(\d+)\s+(.+?)\s+(\d+)$")
        results = [
            (m.group(2), int(m.group(3)), int(m.group(1)))
            for m in map(p_idx.match, results)
        ]

        # Return based on flags
        if with_score:
            if with_index:
                return results
            else:
                return [tup[:2] for tup in results]
        else:
            if with_index:
                return [tup[::2] for tup in results]
            else:
                return [tup[0] for tup in results]

    def _general_import(self):
        """Attempt sequence of all imports."""
        # Lookups for method names and expected import-failure errors
        importers = {
            SourceTypes.BytesPlaintext: self._import_plaintext_bytes,
            SourceTypes.BytesZlib: self._import_zlib_bytes,
            SourceTypes.FnamePlaintext: self._import_plaintext_fname,
            SourceTypes.FnameZlib: self._import_zlib_fname,
            SourceTypes.DictJSON: self._import_json_dict,
        }
        import_errors = {
            SourceTypes.BytesPlaintext: TypeError,
            SourceTypes.BytesZlib: (zlib_error, TypeError),
            SourceTypes.FnamePlaintext: (OSError, TypeError, UnicodeDecodeError),
            SourceTypes.FnameZlib: (OSError, TypeError, zlib_error),
            SourceTypes.DictJSON: (ValidationError),
        }

        # Attempt series of import approaches
        # Enum keys are ordered, so iteration is too.
        for st in SourceTypes:
            if st not in importers:
                # No action for source types w/o a handler function defined.
                continue

            if self._try_import(importers[st], self._source, import_errors[st]):
                self.source_type = st
                return

        # Nothing worked, complain.
        raise TypeError("Invalid Inventory source type")

    def _try_import(self, import_fxn, src, exc):
        """Attempt the indicated import method on the indicated source.

        Returns True on success.

        """
        try:
            p, v, o = import_fxn(src)
        except exc:
            return False

        self.project = p
        self.version = v
        self.objects = o

        return True

    def _import_plaintext_bytes(self, b_str):
        """Import an inventory from plaintext UTF-8 bytes."""
        b_res = pb_project.search(b_str).group(HeaderFields.Project.value)
        project = b_res.decode("utf-8")

        b_res = pb_version.search(b_str).group(HeaderFields.Version.value)
        version = b_res.decode("utf-8")

        def gen_dataobjs():
            """Generate a data object for each line in the inventory."""
            for mch in pb_data.finditer(b_str):
                yield DataObjStr(**mch.groupdict())

        objects = []
        objects.extend(gen_dataobjs())

        if len(objects) == 0:
            raise TypeError("No objects found in plaintext")

        return project, version, objects

    def _import_zlib_bytes(self, b_str):
        """Import a zlib-compressed inventory."""
        b_plain = decompress(b_str)
        p, v, o = self._import_plaintext_bytes(b_plain)

        return p, v, o

    def _import_plaintext_fname(self, fn):
        """Import a plaintext inventory file."""
        b_plain = readbytes(fn)

        return self._import_plaintext_bytes(b_plain)

    def _import_zlib_fname(self, fn):
        """Import a zlib-compressed inventory file."""
        b_zlib = readbytes(fn)

        return self._import_zlib_bytes(b_zlib)

    def _import_url(self, url):
        """Import a file from a remote URL."""
        # Caller's responsibility to ensure URL points
        # someplace safe/sane!
        req = urlrq.Request(url, headers={"User-Agent": "sphobjinv URL/" + soi_version})
        resp = urlrq.urlopen(req, context=self._sslcontext)  # noqa: S310
        b_str = resp.read()

        # Plaintext URL D/L is unreliable; zlib only
        return self._import_zlib_bytes(b_str)

    def _import_json_dict(self, d):
        """Import flat-dict composited data."""
        # Validate the dict against the schema. Schema
        # WILL allow an inventory with no objects here
        val = jsonschema.Draft4Validator(json_schema)
        val.validate(d)

        # Pull header items first
        project = d[HeaderFields.Project.value]
        version = d[HeaderFields.Version.value]
        count = d[HeaderFields.Count.value]

        # No objects is not allowed
        if count < 1:
            raise ValueError("Import of zero-length inventory")

        # Going to destructively process d, so shallow-copy it first
        d = d.copy()

        # Expecting the dict to be indexed by string integers
        objects = []
        for i in range(count):
            try:
                objects.append(DataObjStr(**d.pop(str(i))))
            except KeyError as e:
                if self._count_error:
                    err_str = (
                        f"Too few objects found in dict (halt at {i}, expect {count})"
                    )
                    raise ValueError(err_str) from e

        # Complain if remaining objects are anything other than the
        # valid inventory-level header keys
        hf_values = {e.value for e in HeaderFields}
        check_value = self._count_error and set(d.keys()).difference(hf_values)
        if check_value:
            # A truthy value here will be the contents
            # of the above set difference
            err_str = f"Too many objects in dict ({check_value})"
            raise ValueError(err_str)

        # Should be good to return
        return project, version, objects
