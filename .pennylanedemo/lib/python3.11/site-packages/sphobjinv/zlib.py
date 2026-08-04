r"""*zlib (de)compression helpers for* ``sphobjinv``.

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

import io
import os
import zlib

BUFSIZE = 16 * 1024  # 16k chunks


def decompress(bstr):
    """Decompress a version 2 |isphx| |objects.inv| bytestring.

    The `#`-prefixed comment lines are left unchanged, whereas the
    :mod:`zlib`-compressed data lines are decompressed to plaintext.

    Parameters
    ----------
    bstr

        |bytes| -- Binary string containing a compressed |objects.inv|
        file.

    Returns
    -------
    out_b

        |bytes| -- Decompressed binary string containing the plaintext
        |objects.inv| content.

    """
    from sphobjinv.error import VersionError

    def decompress_chunks(bstrm):
        """Handle chunk-wise zlib decompression.

        Internal function pulled from intersphinx.py@v1.4.1:
        https://github.com/sphinx-doc/sphinx/blob/1.4.1/sphinx/
        ext/intersphinx.py#L79-L124.

        BUFSIZE taken as the default value from intersphinx signature
        Modified slightly to take the stream as a parameter,
        rather than assuming one from the parent namespace.

        """
        decompressor = zlib.decompressobj()
        for chunk in iter(lambda: bstrm.read(BUFSIZE), b""):
            yield decompressor.decompress(chunk)
        yield decompressor.flush()

    # Make stream and output string
    strm = io.BytesIO(bstr)

    # Check to be sure it's v2
    out_b = strm.readline()
    if not out_b.endswith(b"2\n"):  # pragma: no cover
        raise VersionError("Only v2 objects.inv files currently supported")

    # Pull name, version, and description lines
    for _ in range(3):
        out_b += strm.readline()

    # Decompress chunks and append
    for chunk in decompress_chunks(strm):
        out_b += chunk

    # Replace newlines with the OS-local newlines, and return
    return out_b.replace(b"\n", os.linesep.encode("utf-8"))


def compress(bstr):
    """Compress a version 2 |isphx| |objects.inv| bytestring.

    The `#`-prefixed comment lines are left unchanged, whereas the
    plaintext data lines are compressed with :mod:`zlib`.

    Parameters
    ----------
    bstr

        |bytes| -- Binary string containing the decompressed contents of an
        |objects.inv| file.

    Returns
    -------
    out_b

        |bytes| -- Binary string containing the compressed |objects.inv|
        content.

    """
    from sphobjinv.re import pb_comments, pb_data

    # Preconvert any DOS newlines to Unix
    s = bstr.replace(b"\r\n", b"\n")

    # Pull all of the lines
    m_comments = pb_comments.findall(s)
    m_data = pb_data.finditer(s)

    # Assemble the binary header comments and data
    # Comments and data blocks must end in newlines
    hb = b"\n".join(m_comments) + b"\n"
    db = b"\n".join(_.group(0) for _ in m_data) + b"\n"

    # Compress the data block
    # Compression level nine is to match that specified in
    #  sphinx html builder:
    # https://github.com/sphinx-doc/sphinx/blob/1.4.1/sphinx/
    #    builders/html.py#L843
    dbc = zlib.compress(db, 9)

    # Return the composited bytestring
    return hb + dbc
