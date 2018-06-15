"""Test cases for building an C extension and running it."""

import os.path
import subprocess
from typing import List

from mypy import build
from mypy.test.data import parse_test_cases, DataDrivenTestCase
from mypy.test.config import test_temp_dir
from mypy.errors import CompileError
from mypy.options import Options

from mypyc import genops
from mypyc import emitmodule
from mypyc import buildc
from mypyc.test.testutil import (
    ICODE_GEN_BUILTINS, use_custom_builtins, MypycDataSuite, assert_test_output,
)

import pytest  # type: ignore  # no pytest in typeshed

files = [
    'run.test',
    'run-classes.test',
    'run-multimodule.test',
    'run-bench.test',
]


class TestRun(MypycDataSuite):
    """Test cases that build a C extension and run code."""
    files = files
    base_path = test_temp_dir
    optional_out = True

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        bench = testcase.config.getoption('--bench', False) and 'Benchmark' in testcase.name

        with use_custom_builtins(os.path.join(self.data_prefix, ICODE_GEN_BUILTINS), testcase):
            text = '\n'.join(testcase.input)

            options = Options()
            options.use_builtins_fixtures = True
            options.show_traceback = True
            options.strict_optional = True
            options.python_version = (3, 6)

            os.mkdir('tmp/py')
            source_path = 'tmp/py/native.py'
            with open(source_path, 'w') as f:
                f.write(text)
            with open('tmp/interpreted.py', 'w') as f:
                f.write(text)

            source = build.BuildSource(source_path, 'native', text)
            sources = [source]
            module_names = ['native']

            # Hard code another module name to compile in the same compilation unit.
            to_delete = []
            for fn, text in testcase.files:
                if os.path.basename(fn) == 'other.py':
                    module_names.append('other')
                    sources.append(build.BuildSource(fn, 'other', text))
                    to_delete.append(fn)

            try:
                ctext = emitmodule.compile_modules_to_c(
                    sources=sources,
                    module_names=module_names,
                    options=options,
                    alt_lib_path=test_temp_dir)
            except CompileError as e:
                for line in e.messages:
                    print(line)
                assert False, 'Compile error'

            # If compiling more than one native module, compile a shared
            # library that contains all the modules. Also generate shims that
            # just call into the shared lib.
            use_shared_lib = len(module_names) > 1

            if use_shared_lib:
                common_path = os.path.join(test_temp_dir, 'stuff.c')
                with open(common_path, 'w') as f:
                    f.write(ctext)
                try:
                    shared_lib = buildc.build_shared_lib_for_modules(common_path)
                except buildc.BuildError as err:
                    heading('Generated C')
                    with open(common_path) as f:
                        print_with_line_nums(f.read().rstrip())
                    heading('End C')
                    heading('Build output')
                    print(err.output.decode('utf8').rstrip('\n'))
                    heading('End output')
                    raise

            for mod in module_names:
                cpath = os.path.join(test_temp_dir, '%s.c' % mod)
                with open(cpath, 'w') as f:
                    f.write(ctext)

                try:
                    if use_shared_lib:
                        native_lib_path = buildc.build_c_extension_shim(mod, shared_lib)
                    else:
                        native_lib_path = buildc.build_c_extension(cpath, mod, preserve_setup=True)
                except buildc.BuildError as err:
                    heading('Generated C')
                    with open(cpath) as f:
                        print(f.read().rstrip())
                    heading('End C')
                    heading('Build output')
                    print(err.output.decode('utf8').rstrip('\n'))
                    heading('End output')
                    raise

            # # TODO: is the location of the shared lib good?
            # shared_lib = buildc.build_shared_lib_for_modules(cpath)

            for p in to_delete:
                os.remove(p)

            driver_path = os.path.join(test_temp_dir, 'driver.py')
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.dirname(native_lib_path)
            env['MYPYC_RUN_BENCH'] = '1' if bench else '0'
            proc = subprocess.Popen(['python', driver_path], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, env=env)
            output, _ = proc.communicate()
            output = output.decode('utf8')
            outlines = output.splitlines()

            heading('Generated C')
            with open(cpath) as f:
                print(f.read().rstrip())
            heading('End C')
            if proc.returncode != 0:
                print()
                print('*** Exit status: %d' % proc.returncode)

            # Verify output.
            if bench:
                print('Test output:')
                print(output)
            else:
                assert_test_output(testcase, outlines, 'Invalid output')

            assert proc.returncode == 0


def heading(text: str) -> None:
    print('=' * 20 + ' ' + text + ' ' + '=' * 20)


def print_with_line_nums(s: str) -> None:
    lines = s.splitlines()
    for i, line in enumerate(lines):
        print('%-4d %s' % (i, line))
