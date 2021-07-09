from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize

# pip install pip -U
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# python setup.py build_ext --inplace
# python3 setup.py bdist_egg

ext_module = Extension(
    "autoflow/samples/ruletree/*",
    ["autoflow/samples/ruletree/*.pyx"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)

setup(
    name='autoflow',
    version='0.0.1',
    author='DidaZxc',
    author_email='didazxc@gmail.com',
    description='Only for dida group',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    ext_modules=cythonize(ext_module, annotate=True),

    install_requires=[
        'numpy>=1.18.1',
        'pyspark>=2.4.1',
        'pyarrow>=0.8.0',
        'pandas>=1.0.5',
        'torch>=1.4.0',
        'click>=7.1.2',
        'matplotlib>=3.1.3',
        'pyahocorasick>=1.4.0',
        'findspark>=1.4.2',
        'Cython>=0.29.21 ',
        'pycryptodome>=3.9.8',
        'mmlspark>=0.0.11111111',
        'aiohttp>=3.7.3'
    ],

    entry_points={
        'console_scripts': [
            'autoflow = autoflow.manage:cli'
        ]
    }
)
