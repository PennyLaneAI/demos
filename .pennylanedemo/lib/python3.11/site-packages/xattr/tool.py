##
# Copyright (c) 2007 Apple Inc.
#
# This is the MIT license. This software may also be distributed under the
# same terms as Python (the PSF license).
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
##

from __future__ import print_function

import sys
import os
import getopt
import zlib

import xattr


class NullsInString(Exception):
    """Nulls in string."""


def usage(e=None):
    if e:
        print(e)
        print("")

    name = os.path.basename(sys.argv[0])
    print("usage: %s [-slz] file [file ...]" % (name,))
    print("       %s -p [-slz] attr_name file [file ...]" % (name,))
    print("       %s -w [-sz] attr_name attr_value file [file ...]" % (name,))
    print("       %s -d [-s] attr_name file [file ...]" % (name,))
    print("       %s -c [-s] file [file ...]" % (name,))
    print("")
    print("The first form lists the names of all xattrs on the given file(s).")
    print("The second form (-p) prints the value of the xattr attr_name.")
    print("The third form (-w) sets the value of the xattr attr_name to attr_value.")
    print("The fourth form (-d) deletes the xattr attr_name.")
    print("The fifth form (-c) deletes (clears) all xattrs.")
    print("")
    print("options:")
    print("  -h: print this help")
    print("  -s: act on symbolic links themselves rather than their targets")
    print("  -l: print long format (attr_name: attr_value)")
    print("  -z: compress or decompress (if compressed) attribute value in zip format")

    if e:
        return 64
    else:
        return 0


_FILTER = u''.join([(len(ascii(chr(x))) == 3) and chr(x) or u'.' for x in range(256)])


def _dump(src, length=16):
    result = []
    for i in range(0, len(src), length):
        s = src[i:i+length]
        hexa = ' '.join(["%02X" % ord(x) for x in s])
        printable = s.translate(_FILTER)
        result.append("%04X   %-*s   %s\n" % (i, length*3, hexa, printable))
    return ''.join(result)


def main(argv=sys.argv):
    try:
        (optargs, args) = getopt.getopt(argv[1:], "hlpwdzsc", ["help"])
    except getopt.GetoptError as e:
        return usage(e)

    attr_name = None
    long_format = False
    read = False
    write = False
    delete = False
    clear = False
    nofollow = False
    compress = lambda x: x
    decompress = compress
    errors = []

    for opt, arg in optargs:
        if opt in ("-h", "--help"):
            return usage()
        elif opt == "-l":
            long_format = True
        elif opt == "-s":
            nofollow = True
        elif opt == "-p":
            read = True
            if write or delete or clear:
                return usage("-p not allowed with -w, -d or -c")
        elif opt == "-w":
            write = True
            if read or delete or clear:
                return usage("-w not allowed with -p, -d or -c")
        elif opt == "-d":
            delete = True
            if read or write or clear:
                return usage("-d not allowed with -p, -w or -c")
        elif opt == "-c":
            clear = True
            if read or write or delete:
                return usage("-c not allowed with -p, -w or -d")
        elif opt == "-z":
            compress = zlib.compress
            decompress = zlib.decompress

    if write or delete or clear:
        if long_format:
            return usage("-l not allowed with -w, -d or -c")

    if read or write or delete:
        if not args:
            return usage("No attr_name")
        attr_name = args.pop(0)

    if write:
        if not args:
            return usage("No attr_value")
        attr_value = args.pop(0).encode('utf-8')

    if len(args) == 0:
        return usage("No file")

    if len(args) > 1:
        multiple_files = True
    else:
        multiple_files = False

    options = 0
    if nofollow:
        options |= xattr.XATTR_NOFOLLOW

    for filename in args:
        def onError(e):
            errors.append(e)
            if not os.path.exists(filename):
                sys.stderr.write("No such file: %s\n" % (filename,))
            else:
                sys.stderr.write(str(e) + "\n")

        try:
            attrs = xattr.xattr(filename, options=options)
        except (IOError, OSError) as e:
            onError(e)
            continue

        if write:
            try:
                attrs[attr_name] = compress(attr_value)
            except (IOError, OSError) as e:
                onError(e)
                continue

        elif delete:
            try:
                del attrs[attr_name]
            except (IOError, OSError) as e:
                onError(e)
                continue
            except KeyError:
                onError("No such xattr: %s" % (attr_name,))
                continue

        elif clear:
            try:
                attrs.clear()
            except (IOError, OSError) as e:
                onError(e)
                continue

        else:
            try:
                if read:
                    attr_names = (attr_name,)
                else:
                    attr_names = attrs.keys()
            except (IOError, OSError) as e:
                onError(e)
                continue

            if multiple_files:
                file_prefix = "%s: " % (filename,)
            else:
                file_prefix = ""

            for attr_name in attr_names:
                should_dump = False
                try:
                    try:
                        attr_value = decompress(attrs[attr_name])
                    except zlib.error:
                        attr_value = attrs[attr_name]
                    try:
                        if b'\0' in attr_value:
                            # force dumping
                            raise NullsInString
                        attr_value = attr_value.decode('utf-8')
                    except (UnicodeDecodeError, NullsInString):
                        attr_value = attr_value.decode('latin-1')
                        should_dump = True
                except KeyError:
                    onError("%sNo such xattr: %s" % (file_prefix, attr_name))
                    continue

                if long_format:
                    if should_dump:
                        print("".join((file_prefix, "%s:" % (attr_name,))))
                        print(_dump(attr_value))
                    else:
                        print("".join((file_prefix, "%s: " % (attr_name,), attr_value)))
                else:
                    if read:
                        if should_dump:
                            if file_prefix:
                                print(file_prefix)
                            print(_dump(attr_value))
                        else:
                            print("".join((file_prefix, attr_value)))
                    else:
                        print("".join((file_prefix, attr_name)))

    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
