import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/roi_pooling.c']
headers = ['src/roi_pooling.h']
defines = []
with_cuda = False

if torch.cuda.is_available() and False:
    print('Including CUDA code.')
    sources += ['src/my_lib_cuda.c']
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    '_ext.roi_pooling',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
