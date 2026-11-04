"""
Extended attributes extend the basic attributes of files and directories
in the file system.  They are stored as name:data pairs associated with
file system objects (files, directories, symlinks, etc).

The xattr type wraps a path or file descriptor with a dict-like interface
that exposes these extended attributes.
"""

__version__ = '1.3.0'

import errno

from .lib import (XATTR_NOFOLLOW, XATTR_CREATE, XATTR_REPLACE,
    XATTR_NOSECURITY, XATTR_MAXNAMELEN, XATTR_FINDERINFO_NAME,
    XATTR_RESOURCEFORK_NAME, XATTR_COMPAT_USER_PREFIX,
    _getxattr, _fgetxattr, _setxattr, _fsetxattr,
    _removexattr, _fremovexattr, _listxattr, _flistxattr)


__all__ = [
    "XATTR_NOFOLLOW", "XATTR_CREATE", "XATTR_REPLACE", "XATTR_NOSECURITY",
    "XATTR_MAXNAMELEN", "XATTR_FINDERINFO_NAME", "XATTR_RESOURCEFORK_NAME",
    "xattr", "listxattr", "getxattr", "setxattr", "removexattr"
]


_SENTINEL_MISSING = object()
ERRNO_MISSING = set()
try:
    ERRNO_MISSING.add(errno.ENOATTR)
except AttributeError:
    ERRNO_MISSING.add(errno.ENODATA)


class xattr(object):
    """
    A wrapper for paths or file descriptors to access
    their extended attributes with a dict-like interface
    """

    def __init__(self, obj, options=0):
        """
        obj should be a path, a file descriptor, or an
        object that implements fileno() and returns a file
        descriptor.

        options should be 0 or XATTR_NOFOLLOW.  If set, it will
        be OR'ed with the options passed to getxattr, setxattr, etc.
        """
        self.obj = obj
        self.options = options
        fileno = getattr(obj, 'fileno', None)
        if fileno is not None:
            self.value = fileno()
        else:
            self.value = obj

    def __repr__(self):
        if isinstance(self.value, int):
            flavor = "fd"
        else:
            flavor = "file"
        return "<%s %s=%r>" % (type(self).__name__, flavor, self.value)

    def _call(self, name_func, fd_func, *args):
        if isinstance(self.value, int):
            return fd_func(self.value, *args)
        else:
            return name_func(self.value, *args)

    def get(self, name, options=0, *, default=_SENTINEL_MISSING):
        """
        Retrieve the extended attribute ``name`` as a ``str``.
        Raises ``IOError`` on failure unless default is provided.

        See x-man-page://2/getxattr for options and possible errors.
        """
        try:
            return self._call(_getxattr, _fgetxattr, name, 0, 0, options | self.options)
        except OSError as e:
            if default is not _SENTINEL_MISSING and e.errno in ERRNO_MISSING:
                return default
            raise

    def set(self, name, value, options=0):
        """
        Set the extended attribute ``name`` to ``value``
        Raises ``IOError`` on failure.

        See x-man-page://2/setxattr for options and possible errors.
        """
        return self._call(_setxattr, _fsetxattr, name, value, 0, options | self.options)

    def remove(self, name, options=0):
        """
        Remove the extended attribute ``name``
        Raises ``IOError`` on failure.

        See x-man-page://2/removexattr for options and possible errors.
        """
        return self._call(_removexattr, _fremovexattr, name, options | self.options)

    def list(self, options=0):
        """
        Retrieves the extended attributes currently set as a list
        of strings.  Raises ``IOError`` on failure.

        See x-man-page://2/listxattr for options and possible errors.
        """
        res = self._call(_listxattr, _flistxattr, options | self.options).split(b'\x00')
        res.pop()
        return [XATTR_COMPAT_USER_PREFIX + s.decode('utf-8') for s in res]

    # dict-like methods

    def __len__(self):
        return len(self.list())

    def __delitem__(self, item):
        try:
            self.remove(item)
        except IOError:
            raise KeyError(item)

    def __setitem__(self, item, value):
        self.set(item, value)

    def __getitem__(self, item):
        try:
            return self.get(item)
        except IOError:
            raise KeyError(item)

    def iterkeys(self):
        return iter(self.list())

    __iter__ = iterkeys

    def has_key(self, item):
        try:
            self.get(item)
        except IOError:
            return False
        else:
            return True

    __contains__ = has_key

    def clear(self):
        for k in self.keys():
            del self[k]

    def update(self, seq):
        if not hasattr(seq, 'items'):
            seq = dict(seq)
        for k, v in seq.items():
            self[k] = v

    def copy(self):
        return dict(self.iteritems())

    def setdefault(self, k, d=''):
        try:
            d = self.get(k)
        except IOError:
            self[k] = d
        return d

    def keys(self):
        return self.list()

    def itervalues(self):
        for k, v in self.iteritems():
            yield v

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        for k in self.list():
            yield k, self.get(k)

    def items(self):
        return list(self.iteritems())


def listxattr(f, symlink=False):
    return tuple(xattr(f).list(options=symlink and XATTR_NOFOLLOW or 0))


def getxattr(f, attr, symlink=False):
    return xattr(f).get(attr, options=symlink and XATTR_NOFOLLOW or 0)


def setxattr(f, attr, value, options=0, symlink=False):
    if symlink:
        options |= XATTR_NOFOLLOW
    return xattr(f).set(attr, value, options=options)


def removexattr(f, attr, symlink=False):
    options = symlink and XATTR_NOFOLLOW or 0
    return xattr(f).remove(attr, options=options)
