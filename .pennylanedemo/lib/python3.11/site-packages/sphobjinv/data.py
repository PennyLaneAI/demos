r"""``sphobjinv`` *data classes for individual objects*.

``sphobjinv`` is a toolkit for manipulation and inspection of
Sphinx |objects.inv| files.

**Author**
    Brian Skinn (brian.skinn@gmail.com)

**File Created**
    7 Nov 2017

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

from abc import ABCMeta, abstractmethod
from enum import Enum

import attr


class DataFields(Enum):
    """|Enum| for the fields of |objects.inv| data objects."""

    #: Object name, as recognized internally by Sphinx
    Name = "name"

    #: Sphinx domain housing the object
    Domain = "domain"

    #: Full name of Sphinx role to be used when referencing the object
    Role = "role"

    #: Object search priority
    Priority = "priority"

    #: URI to the location of the object's documentation,
    #: relative to the documentation root
    URI = "uri"

    #: Default display name for the object
    #: in rendered documentation
    #: when referenced as
    #: |cour|\ \:domain\:role\:\`name\`\ |/cour|
    DispName = "dispname"


def _utf8_decode(b):
    """Decode (if needed) to str.

    Helper for type conversions among DataObjStr and DataObjBytes.

    """
    if isinstance(b, bytes):
        return b.decode(encoding="utf-8")
    elif isinstance(b, str):
        return b
    else:
        raise TypeError("Argument must be 'bytes' or 'str'")


def _utf8_encode(s):
    """Encode (if needed) to bytes.

    Helper for type conversions among DataObjStr and DataObjBytes.

    """
    if isinstance(s, str):
        return s.encode(encoding="utf-8")
    elif isinstance(s, bytes):
        return s
    else:
        raise TypeError("Argument must be 'bytes' or 'str'")


class SuperDataObj(metaclass=ABCMeta):
    """Abstract base superclass defining common methods &c. for data objects.

    Intended only to be subclassed
    by :class:`DataObjBytes` and :class:`DataObjStr`,
    to allow definition of common methods, properties, etc.
    all in one place.

    Where marked with |dag|,
    :class:`DataObjBytes` instances will return |bytes| values, whereas
    :class:`DataObjStr` instances will return |str| values.

    """

    #: Helper |str| for generating plaintext |objects.inv|
    #: data lines. The field names MUST match the |str| values
    #: of the :class:`~DataFields` members.
    data_line_fmt = "{name} {domain}:{role} {priority} {uri} {dispname}"

    #: |str.format| template for generating reST-like representations
    #: of object data for :data:`as_rst` (used with
    #: :meth:`Inventory.suggest() <sphobjinv.inventory.Inventory.suggest>`).
    rst_fmt = ":{domain}:{role}:`{name}`"

    def __str__(self):  # pragma: no cover
        """Return pretty string representation."""
        fmt_str = "<{0}:: :{1}:{2}:`{3}`>"

        return fmt_str.format(type(self).__name__, self.domain, self.role, self.name)

    @property
    @abstractmethod
    def name(self):
        r"""Object name, as recognized internally by Sphinx\ |dag|."""

    @property
    @abstractmethod
    def domain(self):
        r"""Sphinx domain containing the object\ |dag|."""

    @property
    @abstractmethod
    def role(self):
        r"""Sphinx role to be used when referencing the object\ |dag|."""

    @property
    @abstractmethod
    def priority(self):
        r"""Object search priority, as handled internally by Sphinx\ |dag|."""

    @property
    @abstractmethod
    def uri(self):
        r"""Object URI relative to documentation root\ |dag|.

        Possibly abbreviated; see :ref:`here <syntax_shorthand>`.

        """

    @property
    @abstractmethod
    def dispname(self):
        r"""Object default name in rendered documentation\ |dag|.

        Possibly abbreviated; see :ref:`here <syntax_shorthand>`.

        """

    @property
    @abstractmethod
    def uri_abbrev(self):
        r"""Abbreviation character(s) for URI tail\ |dag|.

        ``'$'`` or ``b'$'``
        for :doc:`version 2 </syntax>` |objects.inv| files.

        """

    @property
    @abstractmethod
    def dispname_abbrev(self):
        r"""Abbreviation character(s) for display name\ |dag|.

        ``'-'`` or ``b'-'``
        for :doc:`version 2 </syntax>` |objects.inv| files.

        """

    @property
    @abstractmethod
    def as_str(self, s):
        """:class:`DataObjStr` version of instance."""

    @property
    @abstractmethod
    def as_bytes(self, s):
        """:class:`DataObjBytes` version of instance."""

    @abstractmethod
    def _data_line_postprocess(self, s):
        """Post-process the data_line chars output."""

    @property
    def uri_contracted(self):
        """Object-relative URI, contracted with `uri_abbrev`."""
        if self.uri.endswith(self.name):
            return self.uri[: -len(self.name)] + self.uri_abbrev
        else:
            return self.uri

    @property
    def uri_expanded(self):
        """Object-relative URI, with `uri_abbrev` expanded."""
        if self.uri.endswith(self.uri_abbrev):
            return self.uri[: -len(self.uri_abbrev)] + self.name
        else:
            return self.uri

    @property
    def dispname_contracted(self):
        """Object display name, contracted with `dispname_abbrev`."""
        if self.dispname == self.name:
            return self.dispname_abbrev
        else:
            return self.dispname

    @property
    def dispname_expanded(self):
        """Object display name, with `dispname_abbrev` expanded."""
        if self.dispname == self.dispname_abbrev:
            return self.name
        else:
            return self.dispname

    @property
    def as_rst(self):
        r"""|str| reST reference-like object representation.

        Typically will NOT function as a proper reST reference
        in Sphinx source (e.g., a `role` of
        |cour|\ function\ |/cour| must be referenced using
        |cour|\ \:func\:\ |/cour| for the
        |cour|\ py\ |/cour| domain).
        """
        return self.rst_fmt.format(**self.as_str.json_dict())

    def json_dict(self, *, expand=False, contract=False):
        r"""Return the object data formatted as a flat |dict|.

        The returned |dict| is constructed such that it matches the
        relevant subschema of :data:`sphobjinv.schema.json_schema`, to
        facilitate implementation of
        :meth:`Inventory.json_dict()
        <sphobjinv.inventory.Inventory.json_dict>`.

        The |dict|\ s returned by :class:`~sphobjinv.data.DataObjBytes` and
        :class:`~sphobjinv.data.DataObjStr` both have |str|
        keys, but they have |bytes| and |str| values, respectively.
        The |dict| keys are identical to the |str| values of the
        :data:`~sphobjinv.data.DataFields` |Enum| members.

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
            :data:`~sphobjinv.data.SuperDataObj.dispname`

        Returns
        -------
        d

            |dict| -- Object data

        Raises
        ------
        ValueError

            If both `expand` and `contract` are |True|

        """
        if expand and contract:
            raise ValueError("'expand' and 'contract' cannot both be true.")

        d = {a: getattr(self, a) for a in (e.value for e in DataFields)}

        if expand:
            d.update(
                {
                    DataFields.URI.value: self.uri_expanded,
                    DataFields.DispName.value: self.dispname_expanded,
                }
            )

        if contract:
            d.update(
                {
                    DataFields.URI.value: self.uri_contracted,
                    DataFields.DispName.value: self.dispname_contracted,
                }
            )

        return d

    def data_line(self, *, expand=False, contract=False):
        """Compose plaintext |objects.inv| data line from instance contents.

        The format of the resulting data line is given by
        :data:`~sphobjinv.data.SuperDataObj.data_line_fmt`.
        :class:`~sphobjinv.data.DataObjBytes` and
        :class:`~sphobjinv.data.DataObjStr` instances generate data lines
        as |bytes| and |str|, respectively.

        Calling with both `expand` and `contract` as |True| is invalid.

        Parameters
        ----------
        expand

            |bool| *(optional)* -- Return data line with any
            :data:`~sphobjinv.data.SuperDataObj.uri` or
            :data:`~sphobjinv.data.SuperDataObj.dispname`
            abbreviations expanded

        contract

            |bool| *(optional)* -- Return data line with abbreviated
            :data:`~sphobjinv.data.SuperDataObj.uri` and
            :data:`~sphobjinv.data.SuperDataObj.dispname`

        Returns
        -------
        dl

            |bytes| (for :class:`~sphobjinv.data.DataObjBytes`)
            or |str| (for :class:`~sphobjinv.data.DataObjStr`)
            -- Object data line

        Raises
        ------
        ValueError

            If both `expand` and `contract` are |True|

        """
        # Rely on .json_dict to check for invalid expand == contract == True
        fmt_d = self.as_str.json_dict(expand=expand, contract=contract)

        retval = self.data_line_fmt.format(**fmt_d)

        return self._data_line_postprocess(retval)

    def evolve(self, **kwargs):
        r"""Create a new instance with changes applied.

        This helper method provides a concise means for creating new
        instances with only a subset of changed data fields.

        The names of any `kwargs` MUST be keys of the |dict|\ s
        generated by :meth:`~sphobjinv.data.SuperDataObj.json_dict`.

        Parameters
        ----------
        kwargs

            |str| or |bytes| -- Revised value(s) to use in the new
            instance for the passed keyword argument(s).

        Returns
        -------
        dobj

            :class:`~sphobjinv.data.DataObjBytes` or
            :class:`~sphobjinv.data.DataObjStr`
            -- New instance with updated data

        """
        d = self.json_dict()
        d.update(kwargs)
        return self.__class__(**d)


@attr.s(slots=True)
class DataObjStr(SuperDataObj):
    """:class:`SuperDataObj` subclass generating |str| object data.

    Two :class:`DataObjStr` instances will test equal if all of
    :attr:`~sphobjinv.data.SuperDataObj.name`,
    :attr:`~sphobjinv.data.SuperDataObj.domain`,
    :attr:`~sphobjinv.data.SuperDataObj.role`,
    :attr:`~sphobjinv.data.SuperDataObj.priority`,
    :attr:`~sphobjinv.data.SuperDataObj.uri`,
    and :attr:`~sphobjinv.data.SuperDataObj.dispname`
    are equal between them.

    .. doctest:: dataobjstr

        >>> obj = soi.DataObjStr(
        ...     name="foo",
        ...     domain="py",
        ...     role="method",
        ...     priority="1",
        ...     uri="$",
        ...     dispname="-",
        ... )
        >>> obj == obj
        True
        >>> obj == obj.evolve(name="quux")
        False

    .. versionchanged:: 2.1
        Previously, attempts to compare instances resulted in a
        :exc:`RecursionError`.

    """

    uri_abbrev = "$"
    dispname_abbrev = "-"

    name = attr.ib(converter=_utf8_decode)
    domain = attr.ib(converter=_utf8_decode)
    role = attr.ib(converter=_utf8_decode)
    priority = attr.ib(converter=_utf8_decode)
    uri = attr.ib(converter=_utf8_decode)
    dispname = attr.ib(converter=_utf8_decode)

    as_bytes = attr.ib(repr=False, eq=False)

    @as_bytes.default
    def _as_bytes_default(self):
        """Create DataObjBytes equivalent."""
        return DataObjBytes(
            name=self.name,
            domain=self.domain,
            role=self.role,
            priority=self.priority,
            uri=self.uri,
            dispname=self.dispname,
            as_str=self,
        )

    as_str = attr.ib(repr=False, eq=False)

    @as_str.default
    def _as_str_default(self):
        """Return this instance."""
        return self

    def _data_line_postprocess(self, s):
        """Perform no postprocessing."""
        return s


@attr.s(slots=True)
class DataObjBytes(SuperDataObj):
    """:class:`SuperDataObj` subclass generating |bytes| object data.

    Two :class:`DataObjBytes` instances will test equal if all of
    :attr:`~sphobjinv.data.SuperDataObj.name`,
    :attr:`~sphobjinv.data.SuperDataObj.domain`,
    :attr:`~sphobjinv.data.SuperDataObj.role`,
    :attr:`~sphobjinv.data.SuperDataObj.priority`,
    :attr:`~sphobjinv.data.SuperDataObj.uri`,
    and :attr:`~sphobjinv.data.SuperDataObj.dispname`
    are equal between them.

    .. doctest:: dataobjbytes

        >>> obj = soi.DataObjBytes(
        ...     name=b"foo",
        ...     domain=b"py",
        ...     role=b"method",
        ...     priority=b"1",
        ...     uri=b"$",
        ...     dispname=b"-",
        ... )
        >>> obj == obj
        True
        >>> obj == obj.evolve(name=b"quux")
        False

    .. versionchanged:: 2.1
        Previously, attempts to compare instances resulted in a
        :exc:`RecursionError`.

    """

    uri_abbrev = b"$"
    dispname_abbrev = b"-"

    name = attr.ib(converter=_utf8_encode)
    domain = attr.ib(converter=_utf8_encode)
    role = attr.ib(converter=_utf8_encode)
    priority = attr.ib(converter=_utf8_encode)
    uri = attr.ib(converter=_utf8_encode)
    dispname = attr.ib(converter=_utf8_encode)

    as_str = attr.ib(repr=False, eq=False)

    @as_str.default
    def _as_str_default(self):
        """Create DataObjStr equivalent."""
        return DataObjStr(
            name=self.name,
            domain=self.domain,
            role=self.role,
            priority=self.priority,
            uri=self.uri,
            dispname=self.dispname,
            as_bytes=self,
        )

    as_bytes = attr.ib(repr=False, eq=False)

    @as_bytes.default
    def _as_bytes_default(self):
        """Return this instance."""
        return self

    def _data_line_postprocess(self, s):
        """Encode to bytes before data_line return."""
        return s.encode("utf-8")
