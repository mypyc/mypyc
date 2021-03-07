# mypyc: Compile type-annotated Python to fast C extensions

Mypyc compiles Python modules to C extensions. It uses standard Python
[type hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) to
generate fast code. Mypyc uses [mypy](http://www.mypy-lang.org) to
perform type checking and type inference.

Mypyc can compile anything from one module or your entire codebase.

## Documentation

Documentation is available at
[ReadTheDocs](https://mypyc.readthedocs.io/en/latest/index.html).

## Benchmarks

We track the performance of mypyc using
[several benchmarks](https://github.com/mypyc/mypyc-benchmarks). Results are
updated daily to make it easy to track progress.

## Questions or issues?

The mypyc [issue tracker](https://github.com/mypyc/mypyc/issues) lives in this
repository. You can also ask questions in our
[Gitter chat](https://gitter.im/mypyc-dev/community).

## Differences from Cython

* Write clean code without non-standard syntax, such as ``cpdef``, or
  extra decorators, with good performance.

* First-class support for type system features such as tuple types,
  union types and generics.

* Variable type annotations are not needed for good performance, due to
  powerful type inference provided by mypy.

* Full integration with mypy for robust and seamless static type
  checking.

* Mypyc performs strict enforcement of type annotations at runtime,
  for better runtime type safety.

## Development roadmap

These are our near-term focus areas for improving mypyc:

* Improved compatibility with Python
* Much faster compilation (parallel and incremental compilation, and more)
* Usability

... and better performance (always!).

## Development status

We are actively looking for early adopters! Mypyc is currently alpha
software. It's only recommended for production use cases with careful
testing, and if you are willing to contribute fixes or to work around
issues you will encounter.

## Help wanted

New contributors are very welcome! Any help in development, testing,
documentation and benchmarking tasks is highly appreciated.

Useful links for contributors:

* The code lives in the
  [mypyc subdirectory](https://github.com/python/mypy/tree/master/mypyc) of the
  [mypy](https://github.com/python/mypy) repository.

* We have
  [developer documentation](https://github.com/python/mypy/blob/master/mypyc/doc/dev-intro.md).

* Use the [issue tracker](https://github.com/mypyc/mypyc/issues) to find things
  to work on.

* You can ask questions in our [Gitter chat](https://gitter.im/mypyc-dev/community).

## Changelog

Follow our updates on the mypy blog: https://mypy-lang.blogspot.com/

## License

Mypyc and mypy are licensed under the terms of the MIT License, with portions under
the Python Software Foundation license (see
the file [LICENSE](https://github.com/python/mypy/blob/master/LICENSE)
in the mypy repository).
