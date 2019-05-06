#include <stdbool.h>
#include <Python.h>
#include <frameobject.h>
#include <assert.h>
#include "CPy.h"

#ifdef __cplusplus
extern "C" {
#endif
#if 0
} // make emacs happy
#endif

// TODO: Currently only the things that *need* to be defined a single time
// instead of copied into every module live here. This is silly, and most
// of the code in CPy.h and pythonsupport.h should move here.

struct ExcDummyStruct _CPy_ExcDummyStruct = { PyObject_HEAD_INIT(NULL) };
PyObject *_CPy_ExcDummy = (PyObject *)&_CPy_ExcDummyStruct;

// Because its dynamic linker is more restricted than linux/OS X,
// Windows doesn't allow initializing globals with values from
// other dynamic libraries. This means we need to initialize
// things at load time.
void CPy_Init(void) {
    _CPy_ExcDummyStruct.ob_base.ob_type = &PyBaseObject_Type;
}

bool _CPy_IsSafeMetaClass(PyTypeObject *metaclass) {
    // mypyc classes can't work with metaclasses in
    // general. Through some various nasty hacks we *do*
    // manage to work with TypingMeta and its friends.
    if (metaclass == &PyType_Type)
        return true;
    PyObject *module = PyObject_GetAttrString((PyObject *)metaclass, "__module__");
    if (!module) {
        PyErr_Clear();
        return false;
    }

    bool matches = false;
    if (PyUnicode_CompareWithASCIIString(module, "typing") == 0 &&
            (strcmp(metaclass->tp_name, "TypingMeta") == 0
             || strcmp(metaclass->tp_name, "GenericMeta") == 0)) {
        matches = true;
    } else if (PyUnicode_CompareWithASCIIString(module, "abc") == 0 &&
               strcmp(metaclass->tp_name, "ABCMeta") == 0) {
        matches = true;
    }
    Py_DECREF(module);
    return matches;
}

PyObject *CPyType_FromTemplate(PyTypeObject *template_,
                               PyObject *orig_bases,
                               PyObject *modname) {
    PyHeapTypeObject *t = NULL;
    PyTypeObject *dummy_class = NULL;
    PyObject *name = NULL;
    PyObject *bases = NULL;
    PyObject *slots;

    // If the type of the class (the metaclass) is NULL, we default it
    // to being type.  (This allows us to avoid needing to initialize
    // it explicitly on windows.)
    if (!Py_TYPE(template_)) {
        Py_TYPE(template_) = &PyType_Type;
    }
    PyTypeObject *metaclass = Py_TYPE(template_);

    if (orig_bases) {
        bases = update_bases(orig_bases);
        // update_bases doesn't increment the refcount if nothing changes,
        // so we do it to make sure we have distinct "references" to both
        if (bases == orig_bases)
            Py_INCREF(bases);

        // Find the appropriate metaclass from our base classes. We
        // care about this because Generic uses a metaclass prior to
        // Python 3.7.
        metaclass = _PyType_CalculateMetaclass(metaclass, bases);
        if (!metaclass)
            goto error;

        if (!_CPy_IsSafeMetaClass(metaclass)) {
            PyErr_SetString(PyExc_TypeError, "mypyc classes can't have a metaclass");
            goto error;
        }
    }

    name = PyUnicode_FromString(template_->tp_name);
    if (!name)
        goto error;

    // If there is a metaclass other than type, we would like to call
    // its __new__ function. Unfortunately there doesn't seem to be a
    // good way to mix a C extension class and creating it via a
    // metaclass. We need to do it anyways, though, in order to
    // support subclassing Generic[T] prior to Python 3.7.
    //
    // We solve this with a kind of atrocious hack: create a parallel
    // class using the metaclass, determine the bases of the real
    // class by pulling them out of the parallel class, creating the
    // real class, and then merging its dict back into the original
    // class. There are lots of cases where this won't really work,
    // but for the case of GenericMeta setting a bunch of properties
    // on the class we should be fine.
    if (metaclass != &PyType_Type) {
        assert(bases && "non-type metaclasses require non-NULL bases");

        PyObject *ns = PyDict_New();
        if (!ns)
            goto error;

        dummy_class = (PyTypeObject *)PyObject_CallFunctionObjArgs(
            (PyObject *)metaclass, name, bases, ns, NULL);
        Py_DECREF(ns);
        if (!dummy_class)
            goto error;

        Py_DECREF(bases);
        bases = dummy_class->tp_bases;
        Py_INCREF(bases);
    }

    // Allocate the type and then copy the main stuff in.
    t = (PyHeapTypeObject*)PyType_GenericAlloc(&PyType_Type, 0);
    if (!t)
        goto error;
    memcpy((char *)t + sizeof(PyVarObject),
           (char *)template_ + sizeof(PyVarObject),
           sizeof(PyTypeObject) - sizeof(PyVarObject));

    if (bases != orig_bases) {
        if (PyObject_SetAttrString((PyObject *)t, "__orig_bases__", orig_bases) < 0)
            goto error;
    }

    // Having tp_base set is I think required for stuff to get
    // inherited in PyType_Ready, which we needed for subclassing
    // BaseException. XXX: Taking the first element is wrong I think though.
    if (bases) {
        t->ht_type.tp_base = (PyTypeObject *)PyTuple_GET_ITEM(bases, 0);
        Py_INCREF((PyObject *)t->ht_type.tp_base);
    }

    t->ht_name = name;
    Py_INCREF(name);
    t->ht_qualname = name;
    t->ht_type.tp_bases = bases;
    // references stolen so NULL these out
    bases = name = NULL;

    if (PyType_Ready((PyTypeObject *)t) < 0)
        goto error;

    assert(t->ht_type.tp_base != NULL);

    // XXX: This is a terrible hack to work around a cpython check on
    // the mro. It was needed for mypy.stats. I need to investigate
    // what is actually going on here.
    Py_INCREF(metaclass);
    Py_TYPE(t) = metaclass;

    if (dummy_class) {
        if (PyDict_Merge(t->ht_type.tp_dict, dummy_class->tp_dict, 0) != 0)
            goto error;
        // This is the *really* tasteless bit. GenericMeta's __new__
        // in certain versions of typing sets _gorg to point back to
        // the class. We need to override it to keep it from pointing
        // to the proxy.
        if (PyDict_SetItemString(t->ht_type.tp_dict, "_gorg", (PyObject *)t) < 0)
            goto error;
    }

    // Reject anything that would give us a nontrivial __slots__,
    // because the layout will conflict
    slots = PyObject_GetAttrString((PyObject *)t, "__slots__");
    if (slots) {
        // don't fail on an empty __slots__
        int is_true = PyObject_IsTrue(slots);
        Py_DECREF(slots);
        if (is_true > 0)
            PyErr_SetString(PyExc_TypeError, "mypyc classes can't have __slots__");
        if (is_true != 0)
            goto error;
    } else {
        PyErr_Clear();
    }

    if (PyObject_SetAttrString((PyObject *)t, "__module__", modname) < 0)
        goto error;

    if (init_subclass((PyTypeObject *)t, NULL))
        goto error;

    Py_XDECREF(dummy_class);

    return (PyObject *)t;

error:
    Py_XDECREF(t);
    Py_XDECREF(bases);
    Py_XDECREF(dummy_class);
    Py_XDECREF(name);
    return NULL;
}

CPyTagged CPyTagged_Multiply(CPyTagged left, CPyTagged right) {
    // TODO: Consider using some clang/gcc extension
    if (CPyTagged_CheckShort(left) && CPyTagged_CheckShort(right)) {
        if (!CPyTagged_IsMultiplyOverflow(left, right)) {
            return left * CPyTagged_ShortAsSsize_t(right);
        }
    }
    PyObject *left_obj = CPyTagged_AsObject(left);
    PyObject *right_obj = CPyTagged_AsObject(right);
    PyObject *result = PyNumber_Multiply(left_obj, right_obj);
    if (result == NULL) {
        CPyError_OutOfMemory();
    }
    Py_DECREF(left_obj);
    Py_DECREF(right_obj);
    return CPyTagged_StealFromObject(result);
}

CPyTagged CPyTagged_FloorDivide(CPyTagged left, CPyTagged right) {
    if (CPyTagged_CheckShort(left) && CPyTagged_CheckShort(right)
        && !CPyTagged_MaybeFloorDivideOverflow(left, right)) {
        if (right == 0)
            abort();
        Py_ssize_t result = ((Py_ssize_t)left / CPyTagged_ShortAsSsize_t(right)) & ~1;
        if (((Py_ssize_t)left < 0) != (((Py_ssize_t)right) < 0)) {
            if (result / 2 * right != left) {
                // Round down
                result -= 2;
            }
        }
        return result;
    }
    PyObject *left_obj = CPyTagged_AsObject(left);
    PyObject *right_obj = CPyTagged_AsObject(right);
    PyObject *result = PyNumber_FloorDivide(left_obj, right_obj);
    if (result == NULL) {
        CPyError_OutOfMemory();
    }
    Py_DECREF(left_obj);
    Py_DECREF(right_obj);
    return CPyTagged_StealFromObject(result);
}

CPyTagged CPyTagged_Remainder(CPyTagged left, CPyTagged right) {
    if (CPyTagged_CheckShort(left) && CPyTagged_CheckShort(right)
        && !CPyTagged_MaybeRemainderOverflow(left, right)) {
        Py_ssize_t result = (Py_ssize_t)left % (Py_ssize_t)right;
        if (((Py_ssize_t)right < 0) != ((Py_ssize_t)left < 0) && result != 0) {
            result += right;
        }
        return result;
    }
    PyObject *left_obj = CPyTagged_AsObject(left);
    PyObject *right_obj = CPyTagged_AsObject(right);
    PyObject *result = PyNumber_Remainder(left_obj, right_obj);
    if (result == NULL) {
        CPyError_OutOfMemory();
    }
    Py_DECREF(left_obj);
    Py_DECREF(right_obj);
    return CPyTagged_StealFromObject(result);
}

PyObject *CPyList_GetItemShort(PyObject *list, CPyTagged index) {
    Py_ssize_t n = CPyTagged_ShortAsSsize_t(index);
    Py_ssize_t size = PyList_GET_SIZE(list);
    if (n >= 0) {
        if (n >= size) {
            PyErr_SetString(PyExc_IndexError, "list index out of range");
            return NULL;
        }
    } else {
        n += size;
        if (n < 0) {
            PyErr_SetString(PyExc_IndexError, "list index out of range");
            return NULL;
        }
    }
    PyObject *result = PyList_GET_ITEM(list, n);
    Py_INCREF(result);
    return result;
}

PyObject *CPyList_GetItem(PyObject *list, CPyTagged index) {
    if (CPyTagged_CheckShort(index)) {
        Py_ssize_t n = CPyTagged_ShortAsSsize_t(index);
        Py_ssize_t size = PyList_GET_SIZE(list);
        if (n >= 0) {
            if (n >= size) {
                PyErr_SetString(PyExc_IndexError, "list index out of range");
                return NULL;
            }
        } else {
            n += size;
            if (n < 0) {
                PyErr_SetString(PyExc_IndexError, "list index out of range");
                return NULL;
            }
        }
        PyObject *result = PyList_GET_ITEM(list, n);
        Py_INCREF(result);
        return result;
    } else {
        PyErr_SetString(PyExc_IndexError, "list index out of range");
        return NULL;
    }
}

bool CPyList_SetItem(PyObject *list, CPyTagged index, PyObject *value) {
    if (CPyTagged_CheckShort(index)) {
        Py_ssize_t n = CPyTagged_ShortAsSsize_t(index);
        Py_ssize_t size = PyList_GET_SIZE(list);
        if (n >= 0) {
            if (n >= size) {
                PyErr_SetString(PyExc_IndexError, "list assignment index out of range");
                return false;
            }
        } else {
            n += size;
            if (n < 0) {
                PyErr_SetString(PyExc_IndexError, "list assignment index out of range");
                return false;
            }
        }
        // PyList_SET_ITEM doesn't decref the old element, so we do
        Py_DECREF(PyList_GET_ITEM(list, n));
        // N.B: Steals reference
        PyList_SET_ITEM(list, n, value);
        return true;
    } else {
        PyErr_SetString(PyExc_IndexError, "list assignment index out of range");
        return false;
    }
}

PyObject *CPyList_Pop(PyObject *obj, CPyTagged index) {
    if (CPyTagged_CheckShort(index)) {
        Py_ssize_t n = CPyTagged_ShortAsSsize_t(index);
        return list_pop_impl((PyListObject *)obj, n);
    } else {
        PyErr_SetString(PyExc_IndexError, "pop index out of range");
        return NULL;
    }
}

bool CPySet_Remove(PyObject *set, PyObject *key) {
    int success = PySet_Discard(set, key);
    if (success == 1) {
        return true;
    }
    if (success == 0) {
        _PyErr_SetKeyError(key);
    }
    return false;
}

PyObject *CPySequenceTuple_GetItem(PyObject *tuple, CPyTagged index) {
    if (CPyTagged_CheckShort(index)) {
        Py_ssize_t n = CPyTagged_ShortAsSsize_t(index);
        Py_ssize_t size = PyTuple_GET_SIZE(tuple);
        if (n >= 0) {
            if (n >= size) {
                PyErr_SetString(PyExc_IndexError, "tuple index out of range");
                return NULL;
            }
        } else {
            n += size;
            if (n < 0) {
                PyErr_SetString(PyExc_IndexError, "tuple index out of range");
                return NULL;
            }
        }
        PyObject *result = PyTuple_GET_ITEM(tuple, n);
        Py_INCREF(result);
        return result;
    } else {
        PyErr_SetString(PyExc_IndexError, "tuple index out of range");
        return NULL;
    }
}

PyObject *CPyDict_GetItem(PyObject *dict, PyObject *key) {
    if (PyDict_CheckExact(dict)) {
        PyObject *res = PyDict_GetItemWithError(dict, key);
        if (!res) {
            if (!PyErr_Occurred()) {
                PyErr_SetObject(PyExc_KeyError, key);
            }
        } else {
            Py_INCREF(res);
        }
        return res;
    } else {
        return PyObject_GetItem(dict, key);
    }
}

PyObject *CPyDict_Get(PyObject *dict, PyObject *key, PyObject *fallback) {
    // We are dodgily assuming that get on a subclass doesn't have
    // different behavior.
    PyObject *res = PyDict_GetItemWithError(dict, key);
    if (!res) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        res = fallback;
    }
    Py_INCREF(res);
    return res;
}

PyObject *CPyDict_FromAny(PyObject *obj) {
    if (PyDict_Check(obj)) {
        return PyDict_Copy(obj);
    } else {
        int res;
        PyObject *dict = PyDict_New();
        if (!dict) {
            return NULL;
        }
        _Py_IDENTIFIER(keys);
        if (_PyObject_HasAttrId(obj, &PyId_keys)) {
            res = PyDict_Update(dict, obj);
        } else {
            res = PyDict_MergeFromSeq2(dict, obj, 1);
        }
        if (res < 0) {
            Py_DECREF(dict);
            return NULL;
        }
        return dict;
    }
}

PyCodeObject *CPy_CreateCodeObject(const char *filename, const char *funcname, int line) {
    PyObject *filename_obj = PyUnicode_FromString(filename);
    PyObject *funcname_obj = PyUnicode_FromString(funcname);
    PyObject *empty_bytes = PyBytes_FromStringAndSize("", 0);
    PyObject *empty_tuple = PyTuple_New(0);
    PyCodeObject *code_obj = NULL;
    if (filename_obj == NULL || funcname_obj == NULL || empty_bytes == NULL
        || empty_tuple == NULL) {
        goto Error;
    }
    code_obj = PyCode_New(0, 0, 0, 0, 0,
                          empty_bytes,
                          empty_tuple,
                          empty_tuple,
                          empty_tuple,
                          empty_tuple,
                          empty_tuple,
                          filename_obj,
                          funcname_obj,
                          line,
                          empty_bytes);
  Error:
    Py_XDECREF(empty_bytes);
    Py_XDECREF(empty_tuple);
    Py_XDECREF(filename_obj);
    Py_XDECREF(funcname_obj);
    return code_obj;
}

void CPy_CatchError(PyObject **p_type, PyObject **p_value, PyObject **p_traceback) { 
    // We need to return the existing sys.exc_info() information, so
    // that it can be restored when we finish handling the error we
    // are catching now. Grab that triple and convert NULL values to
    // the ExcDummy object in order to simplify refcount handling in
    // generated code.
    PyErr_GetExcInfo(p_type, p_value, p_traceback);
    _CPy_ToDummy(p_type);
    _CPy_ToDummy(p_value);
    _CPy_ToDummy(p_traceback);

    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "CPy_CatchError called with no error!");
    }

    // Retrieve the error info and normalize it so that it looks like
    // what python code needs it to be.
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    // Could we avoid always normalizing?
    PyErr_NormalizeException(&type, &value, &traceback);
    if (traceback != NULL) {
        PyException_SetTraceback(value, traceback);
    }
    // Indicate that we are now handling this exception by stashing it
    // in sys.exc_info().  mypyc routines that need access to the
    // exception will read it out of there.
    PyErr_SetExcInfo(type, value, traceback);
    // Clear the error indicator, since the exception isn't
    // propagating anymore.
    PyErr_Clear();
}

// A somewhat hairy implementation of specifically most of the error handling
// in `yield from` error handling. The point here is to reduce code size.
//
// This implements most of the bodies of the `except` blocks in the
// pseudocode in PEP 380.
//
// Returns true (1) if a StopIteration was received and we should return.
// Returns false (0) if a value should be yielded.
// In both cases the value is stored in outp.
// Signals an error (2) if the an exception should be propagated.
int CPy_YieldFromErrorHandle(PyObject *iter, PyObject **outp) {
    _Py_IDENTIFIER(close);
    _Py_IDENTIFIER(throw);
    PyObject *exc_type = CPy_ExcState()->exc_type;
    PyObject *type, *value, *traceback;
    PyObject *_m;
    PyObject *res;
    *outp = NULL;

    if (PyErr_GivenExceptionMatches(exc_type, PyExc_GeneratorExit)) {
        _m = _PyObject_GetAttrId(iter, &PyId_close);
        if (_m) {
            res = PyObject_CallFunctionObjArgs(_m, NULL);
            Py_DECREF(_m);
            if (!res)
                return 2;
            Py_DECREF(res);
        } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        } else {
            return 2;
        }
    } else {
        _m = _PyObject_GetAttrId(iter, &PyId_throw);
        if (_m) {
            CPy_GetExcInfo(&type, &value, &traceback);
            res = PyObject_CallFunctionObjArgs(_m, type, value, traceback, NULL);
            Py_DECREF(type);
            Py_DECREF(value);
            Py_DECREF(traceback);
            Py_DECREF(_m);
            if (res) {
                *outp = res;
                return 0;
            } else {
                res = CPy_FetchStopIterationValue();
                if (res) {
                    *outp = res;
                    return 1;
                }
            }
        } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
        } else {
            return 2;
        }
    }

    CPy_Reraise();
    return 2;
}
#ifdef __cplusplus
}
#endif
