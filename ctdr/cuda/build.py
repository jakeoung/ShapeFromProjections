from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='diff_fp',
    ext_modules=[ CUDAExtension('cuda_diff_fp', [
        'diff_fp.cpp',
        'diff_fp_for.cu',
        'diff_fp_back.cu'
    ]) ],
    cmdclass={'build_ext': BuildExtension}
)
