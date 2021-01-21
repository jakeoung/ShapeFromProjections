from setuptools import setup, find_packages

INSTALL_REQUIREMENTS = ['numpy', 'trimesh', 'torch', 'matplotlib', 'h5py']

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules=[ CUDAExtension('ctdr.cuda.diff_fp', [
        'ctdr/cuda/diff_fp.cpp',
        'ctdr/cuda/diff_fp_for.cu',
        'ctdr/cuda/diff_fp_back.cu'
    ]) ]


if __name__ == '__main__':
#   print(find_packages(exclude=( 'test')))
    
    setup(
        description='CTDR"',
        version='0.9.0',
        name='ctdrm', # not module name
        packages=find_packages(include=['ctdr','ctdr.*']),
        zip_safe=True,
        install_requires=INSTALL_REQUIREMENTS,
        ext_modules=ext_modules,
        cmdclass = {'build_ext': BuildExtension}   
    )
