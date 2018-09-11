import glob
import sys
import os.path
import subprocess
import hashlib
import time

from typing import List, Tuple, Any, Optional, Union, Dict, NoReturn, cast

base_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(base_path, 'external/mypy'))

from mypy.main import process_options
from mypy.errors import CompileError
from mypy.options import Options
from mypy.build import BuildSource
from mypyc.namegen import exported_name

from mypyc import emitmodule


# We can work with either setuptools or distutils, and pick setuptools
# if it has been imported.
assert 'setuptools' in sys.modules or 'distutils' in sys.modules, (
    "'setuptools' or 'distutils' must be imported before mypyc.build")
USE_SETUPTOOLS = 'setuptools' in sys.modules

if USE_SETUPTOOLS:
    from setuptools import setup, Extension  # type: ignore
    from setuptools.command.build_ext import build_ext  # type: ignore
else:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext  # type: ignore

from distutils import sysconfig


def setup_mypycify_vars() -> None:
    # Rewrite a bunch of state variables in pretty dubious ways.
    # There has to be a better approach to this.

    # The vars can contain ints but we only work with str ones
    vars = cast(Dict[str, str], sysconfig.get_config_vars())
    if sys.platform == 'darwin':
        # On OS X, force the creation of dynamic libraries instead of bundles so that
        # we can link against multi-module shared libraries.
        # From https://stackoverflow.com/a/32765319
        vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
        # Also disable building 32-bit binaries, since we generate too much code
        # for a 32-bit Mach-O object. There has to be a better way to do this.
        vars['LDFLAGS'] = vars['LDFLAGS'].replace('-arch i386', '')
        vars['CFLAGS'] = vars['CFLAGS'].replace('-arch i386', '')


class MypycifyExtension(Extension):
    def __init__(self, *args: Any,
                 is_mypyc_shared: bool = False,
                 mypyc_shared_target: Optional['MypycifyExtension'] = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.is_mypyc_shared = is_mypyc_shared
        self.mypyc_shared_target = mypyc_shared_target


def fail(message: str) -> NoReturn:
    # TODO: Is there something else we should do to fail?
    sys.exit(message)


def get_sources(paths: List[str],
                mypy_options: Optional[List[str]]) -> Tuple[List[BuildSource], Options]:
    # It is kind of silly to do this but oh well
    mypy_options = mypy_options or []
    mypy_options.append('--')
    mypy_options.extend(paths)

    sources, options = process_options(mypy_options)
    if options.python_version[0] == 2:
        fail('Python 2 not supported')
    if not options.strict_optional:
        fail('Disabling strict optional checking not supported')
    options.show_traceback = True
    # Needed to get types for all AST nodes
    options.export_types = True
    # TODO: Support incremental checking
    options.incremental = False

    return sources, options


shim_template = """\
#include <Python.h>

PyObject *CPyInit_{full_modname}(void);

PyMODINIT_FUNC
PyInit_{modname}(void)
{{
    return CPyInit_{full_modname}();
}}
"""


def generate_c_extension_shim(full_module_name: str, module_name: str, dirname: str) -> str:
    cname = '%s.c' % full_module_name.replace('.', '___')  # XXX
    cpath = os.path.join(dirname, cname)

    with open(cpath, 'w') as f:
        f.write(shim_template.format(modname=module_name,
                                     full_modname=exported_name(full_module_name)))

    return cpath


def shared_lib_name(modules: List[str]) -> str:
    h = hashlib.sha1()
    h.update(','.join(modules).encode())
    return 'mypyc_%s' % h.hexdigest()[:20]


def include_dir() -> str:
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'lib-rt')


def generate_c(sources: List[BuildSource], options: Options,
               use_shared_lib: bool) -> Tuple[str, str]:
    module_names = [source.module for source in sources]
    # We generate a shared lib if there are multiple modules or if any
    # of the modules are in package. (Because I didn't want to fuss
    # around with making the single module code handle packages.)

    # Do the actual work now
    t0 = time.time()
    try:
        result = emitmodule.parse_and_typecheck(sources, options)
    except CompileError as e:
        for line in e.messages:
            print(line)
        fail('Typechecking failure')

    t1 = time.time()
    print("Parsed and typechecked in {:.3f}s".format(t1 - t0))

    ops = []  # type: List[str]
    ctext = emitmodule.compile_modules_to_c(result, module_names, use_shared_lib, ops=ops)

    t2 = time.time()
    print("Compiled to C in {:.3f}s".format(t2 - t1))

    return ctext, '\n'.join(ops)


def build_using_shared_lib(sources: List[BuildSource],
                           lib_name: str,
                           cfile: str,
                           build_dir: str,
                           extra_compile_args: List[str],
                           ) -> List[MypycifyExtension]:
    shared_lib = MypycifyExtension(
        'lib' + lib_name,
        is_mypyc_shared=True,
        sources=[cfile],
        include_dirs=[include_dir()],
        extra_compile_args=extra_compile_args,
    )
    extensions = [shared_lib]

    for source in sources:
        module_name = source.module.split('.')[-1]
        stub_file = generate_c_extension_shim(source.module, module_name, build_dir)

        # We include the __init__ in the "module name" we stick in the Extension,
        # since this seems to be needed for it to end up in the right place.
        full_module_name = source.module
        assert source.path
        if os.path.split(source.path)[1] == '__init__.py':
            full_module_name += '.__init__'
        extensions.append(MypycifyExtension(
            full_module_name,
            mypyc_shared_target=shared_lib,
            sources=[stub_file],
            extra_compile_args=extra_compile_args,
        ))

    return extensions


def build_single_module(sources: List[BuildSource],
                        cfile: str,
                        extra_compile_args: List[str],
                        ) -> List[MypycifyExtension]:
    return [MypycifyExtension(
        sources[0].module,
        sources=[cfile],
        include_dirs=[include_dir()],
        extra_compile_args=extra_compile_args,
    )]


def mypycify(paths_arg: Union[str, List[str]],
             mypy_options: Optional[List[str]] = None,
             opt_level: str = '',
             skip_cgen: bool = False) -> Dict[str, Any]:
    setup_mypycify_vars()

    if not isinstance(paths_arg, list):
        paths_arg = [paths_arg]

    paths = []
    for path in paths_arg:
        paths.extend(glob.glob(path))

    build_dir = 'build'  # can this be overridden??
    try:
        os.mkdir(build_dir)
    except FileExistsError:
        pass

    sources, options = get_sources(paths, mypy_options)
    use_shared_lib = len(sources) > 1 or any('.' in x.module for x in sources)
    cfile = os.path.join(build_dir, '__native.c')

    # We let the test harness make us skip doing the full compilation
    # so that it can do a corner-cutting version without full stubs.
    # TODO: Be able to do this based on file mtimes?
    if not skip_cgen:
        ctext, ops_text = generate_c(sources, options, use_shared_lib)
        # TODO: unique names?
        with open(os.path.join(build_dir, 'ops.txt'), 'w') as f:
            f.write(ops_text)
        with open(cfile, 'w') as f:
            f.write(ctext)

    cflags = [
        '-O{}'.format(opt_level),
        '-Werror', '-Wno-unused-function', '-Wno-unused-label',
        '-Wno-unreachable-code', '-Wno-unused-variable', '-Wno-trigraphs',
        '-Wno-unused-command-line-argument'
    ]
    if sys.platform == 'linux' and os.getenv('CC', None) != 'clang':
        # This flag is needed for gcc but does not exist on clang.
        cflags += ['-Wno-unused-but-set-variable']

    if use_shared_lib:
        lib_name = shared_lib_name([source.module for source in sources])
        extensions = build_using_shared_lib(sources, lib_name, cfile, build_dir, cflags)
    else:
        extensions = build_single_module(sources, cfile, cflags)

    return {
        'ext_modules': extensions,
        'cmdclass': {'build_ext': MypycifyBuildExt},
    }


class MypycifyBuildExt(build_ext):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # parallel is busted because the shared library needs to go first.
        # We could override build_extensions to arrange for that to happen
        # and to parallelize the rest but it wouldn't help much.
        self.parallel = 0

    def _get_rt_lib_path(self, ext: MypycifyExtension) -> str:
        module_parts = ext.name.split('.')
        # if is_package:
        #     module_parts.append('__init__')
        if len(module_parts) > 1:
            relative_lib_path = os.path.join(*(['..'] * (len(module_parts) - 1)))
        else:
            relative_lib_path = '.'
        return relative_lib_path

    def build_extension(self, ext: MypycifyExtension) -> None:
        # First we need to figure out what the real library names are
        # so that we can set them up properly.
        if isinstance(ext, MypycifyExtension) and ext.mypyc_shared_target:
            relative_lib_path = self._get_rt_lib_path(ext)
            shared_dir, shared_file = os.path.split(
                self.get_ext_fullpath(ext.mypyc_shared_target.name))
            shared_name = os.path.splitext(shared_file)[0][3:]
            ext.libraries.append(shared_name)
            ext.library_dirs.append(shared_dir)
            if sys.platform == 'linux':
                ext.runtime_library_dirs.append('$ORIGIN/{}'.format(
                    relative_lib_path))

        # Run the actual C build
        super().build_extension(ext)

        # On OS X, we need to patch up these paths post-hoc, tragically
        if sys.platform == 'darwin':
            out_path = self.get_ext_fullpath(ext.name)
            if isinstance(ext, MypycifyExtension) and ext.is_mypyc_shared:
                subprocess.check_call(['install_name_tool', '-id',
                                       os.path.basename(out_path),
                                       out_path])
            if isinstance(ext, MypycifyExtension) and ext.mypyc_shared_target:
                new_path = os.path.join('@loader_path', relative_lib_path,
                                        shared_file)
                subprocess.check_call(['install_name_tool', '-change',
                                       shared_file, new_path, out_path])
