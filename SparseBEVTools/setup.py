import os
import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    #find ./ -name *.so | xargs -I f rm f
    # python setup.py build_ext --inplace
    # TORCH_CUDA_ARCH_LIST="6.1 7.5"
    setup(
        name='SparseBEVTools',
        version='0.1',
        description='',
        author='',
        author_email='',
        license='Apache License 2.0',
        cmdclass={'build_ext': BuildExtension},
        packages=find_packages(),
        include_package_data=True,
        package_data={'SparseBEVTools': ['*/*.so']},
        ext_modules=[
            make_cuda_ext(
                name='SparseBEVToolsCUDA',
                module='SparseBEVTools.ops',
                sources=[
                    'src/sparse2bev.cpp',
                    'src/sparse2bev_cuda.cu',
                ]
            )
        ],
    )

