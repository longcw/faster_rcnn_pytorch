import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/roi_pooling.c']
headers = ['src/roi_pooling.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_pooling_cuda.c']
    headers += ['src/roi_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

curdir = os.path.abspath(os.curdir)
extra_objects = ['src/cuda/roi_pooling.cu.o']
extra_objects = [os.path.join(curdir, fname) for fname in extra_objects]

include_dirs = ['src/cuda']
include_dirs = [os.path.join(curdir, fname) for fname in include_dirs]

ffi = create_extension(
    '_ext.roi_pooling',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    # include_dirs=include_dirs
)

if __name__ == '__main__':
    ffi.build()
