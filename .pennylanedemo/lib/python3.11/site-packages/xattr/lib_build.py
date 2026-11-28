import sys
import os
from cffi import FFI

PATH = os.path.dirname(__file__)

with open(os.path.join(PATH, 'lib_build.h')) as hf:
    c_header = hf.read()
with open(os.path.join(PATH, 'lib_build.c')) as cf:
    c_source = cf.read()

ffi = FFI()
ffi.cdef(c_header)
ffi.set_source('_lib', c_source)

if __name__ == '__main__':
    ffi.compile()
