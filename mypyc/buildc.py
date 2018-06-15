"""Build C extension module from C source."""

import glob
import os
import shutil
import subprocess
import tempfile
from typing import List


# TODO: Make compiler arguments platform-specific.
setup_format = """\
from distutils.core import setup, Extension

module = Extension('{package_name}',
                   sources=['{cpath}'],
                   extra_compile_args=['-Wno-unused-function', '-Wno-unused-label', '-Werror',
                                       '-Wno-unreachable-code'])

setup(name='{package_name}',
      version='1.0',
      description='{package_name}',
      include_dirs=['{include_dir}'],
      ext_modules=[module])
"""


def include_dir() -> str:
    return os.path.join(os.path.dirname(__file__), '..', 'lib-rt')


class BuildError(Exception):
    def __init__(self, output: bytes) -> None:
        super().__init__('Build failed')
        self.output = output


def build_c_extension(cpath: str, module_name: str, preserve_setup: bool = False) -> str:
    tempdir = tempfile.mkdtemp()
    if preserve_setup:
        tempdir = '.'
    else:
        tempdir = tempfile.mkdtemp()
    try:
        setup_path = os.path.join(tempdir, 'setup.py')
        basename = os.path.basename(cpath)
        package_name = os.path.splitext(basename)[0]
        with open(setup_path, 'w') as f:
            f.write(setup_format.format(
                cpath=cpath,
                package_name=package_name,
                include_dir=include_dir()))
        try:
            subprocess.check_output(['python', setup_path, 'build'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            raise BuildError(err.output)
        so_path = glob.glob('build/*/%s*.so' % module_name)
        assert len(so_path) == 1
        return so_path[0]
    finally:
        if not preserve_setup:
            shutil.rmtree(tempdir)


# TODO: Make compiler arguments platform-specific.
setup_format_shim = """\
from distutils.core import setup, Extension

module = Extension('{package_name}',
                   sources=['{cpath}'],
                   extra_compile_args=['-Wno-unused-function', '-Wno-unused-label', '-Werror',
                                       '-Wno-unreachable-code'],
                   libraries=['{sharedlib}'],
                   library_dirs=['{libdir}'])

setup(name='{package_name}',
      version='1.0',
      description='{package_name}',
      include_dirs=['{include_dir}'],
      ext_modules=[module])
"""

shim_format = """\
#include <Python.h>

PyObject *CPyInit_{modname}(void);

PyMODINIT_FUNC
PyInit_{modname}(void)
{{
    return CPyInit_{modname}();
}}
"""


def build_c_extension_shim(module_name: str, shared_lib: str) -> str:
    tempdir = tempfile.mkdtemp()
    cpath = os.path.join(tempdir, '%s.c' % module_name)
    with open(cpath, 'w') as f:
        f.write(shim_format.format(modname=module_name))
    try:
        setup_path = os.path.join(tempdir, 'setup.py')
        basename = os.path.basename(cpath)
        package_name = os.path.splitext(basename)[0]
        with open(setup_path, 'w') as f:
            f.write(setup_format_shim.format(
                package_name=package_name,
                cpath=cpath,
                sharedlib='stuff',
                libdir='.',
                include_dir=include_dir()))
        try:
            subprocess.check_output(['python', setup_path, 'build'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            raise BuildError(err.output)
        so_path = glob.glob('build/*/%s*.so' % module_name)
        assert len(so_path) == 1
        return so_path[0]
    finally:
        shutil.rmtree(tempdir)


def build_shared_lib_for_modules(cpath: str) -> str:
    """Build the shared lib for a set of modules."""
    basic_flags = ['-arch', 'x86_64', '-shared', '-I', include_dir()]
    warning_flags = ['-Wno-unused-label', '-Wno-unused-function', '-Wno-unreachable-code']
    py_flags = get_python_flags()
    base_name = 'stuff'
    out_file = 'lib%s.so' % base_name
    cmd = ['clang'] + basic_flags + ['-o', out_file, cpath] + py_flags + warning_flags
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as err:
        raise BuildError(err.output)
    return out_file


def get_python_flags() -> List[str]:
    out = subprocess.check_output(['python-config', '--cflags', '--ldflags']).decode()
    return out.strip().split()
