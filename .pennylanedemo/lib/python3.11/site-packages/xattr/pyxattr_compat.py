"""
pyxattr and xattr have differing API, for example xattr assumes
that (like on OSX) attribute keys are valid UTF-8, while pyxattr
just passes through the raw bytestring.

This module provides compatibility for the pyxattr API.
"""

import sys

from .lib import (XATTR_NOFOLLOW, XATTR_CREATE, XATTR_REPLACE,
    XATTR_NOSECURITY, XATTR_MAXNAMELEN, XATTR_FINDERINFO_NAME,
    XATTR_RESOURCEFORK_NAME, _getxattr, _fgetxattr, _setxattr, _fsetxattr,
    _removexattr, _fremovexattr, _listxattr, _flistxattr)

__all__ = [
    "NS_SECURITY", "NS_USER", "NS_SYSTEM", "NS_TRUSTED",
    "getxattr", "get", "get_all", "setxattr", "set",
    "removexattr", "remove", "listxattr", "list"
]

NS_SECURITY = b"security"
NS_USER = b"user"
NS_SYSTEM = b"system"
NS_TRUSTED = b"trusted"

_NO_NS = object()

_fsencoding = sys.getfilesystemencoding()

def _call(item, name_func, fd_func, *args):
    if isinstance(item, int):
        return fd_func(item, *args)
    elif hasattr(item, 'fileno'):
        return fd_func(item.fileno(), *args)
    elif isinstance(item, bytes):
        return name_func(item, *args)
    elif isinstance(item, str):
        item = item.encode(_fsencoding)
        return name_func(item, *args)
    else:
        raise TypeError("argument must be string, int or file object")

def _add_ns(item, ns):
    if ns is None:
        raise TypeError("namespace must not be None")
    if ns == _NO_NS:
        return item
    return b'.'.join((ns, item))

def getxattr(item, attribute, nofollow=False):
    options = nofollow and XATTR_NOFOLLOW or 0
    return _call(item, _getxattr, _fgetxattr, attribute, 0, 0, options)

def get(item, name, nofollow=False, namespace=_NO_NS):
    name = _add_ns(name, namespace)
    return getxattr(item, name, nofollow=nofollow)

def get_all(item, nofollow=False, namespace=_NO_NS):
    if namespace is not None and namespace != _NO_NS:
        namespace += b'.'
    l = listxattr(item, nofollow=nofollow)
    result = []
    for name in l:
        try:
            if namespace is not None and namespace != _NO_NS:
                if not name.startswith(namespace):
                    continue
                result.append((name[len(namespace):],
                               getxattr(item, name, nofollow=nofollow)))
            else:
                result.append((name, getxattr(item, name, nofollow=nofollow)))
        except IOError:
            pass
    return result

def setxattr(item, name, value, flags=0, nofollow=False):
    options = nofollow and XATTR_NOFOLLOW or 0
    options |= flags
    return _call(item, _setxattr, _fsetxattr, name, value, 0, options)

def set(item, name, value, nofollow=False, flags=0, namespace=_NO_NS):
    name = _add_ns(name, namespace)
    return setxattr(item, name, value, flags=flags, nofollow=nofollow)

def removexattr(item, name, nofollow=False):
    options = nofollow and XATTR_NOFOLLOW or 0
    return _call(item, _removexattr, _fremovexattr, name, options)

def remove(item, name, nofollow=False, namespace=_NO_NS):
    name = _add_ns(name, namespace)
    return removexattr(item, name, nofollow=nofollow)

def listxattr(item, nofollow=False):
    options = nofollow and XATTR_NOFOLLOW or 0
    res = _call(item, _listxattr, _flistxattr, options).split(b'\x00')
    res.pop()
    return res

def list(item, nofollow=False, namespace=_NO_NS):
    if not namespace or namespace == _NO_NS:
        return listxattr(item, nofollow=nofollow)
    namespace += b'.'
    l = listxattr(item, nofollow=nofollow)
    result = []
    for name in l:
        if not name.startswith(namespace):
            continue
        result.append(name[len(namespace):])
    return result
