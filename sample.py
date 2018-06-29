def f(flag: bool) -> str:
    if flag:
        def inner() -> str:
            return 'f.inner: first definition'
    else:
        def inner() -> str:
            return 'f.inner: second definition'
    return inner()

print(f(True))

