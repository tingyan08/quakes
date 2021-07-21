#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "raw-convert.h"

// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}


static PyObject* py_raw_convert(PyObject* NPY_UNUSED(self),
                                PyObject* args,
                                PyObject* kwargs)
{
    const char* filename_raw = NULL;
    int width  = -1;
    int height = -1;
    int saturate_recip = 1 << 16;
    PyArrayObject* out_rgb = NULL;
    PyObject* result = NULL;

    SET_SIGINT();

    char* keywords[] = { "filename_raw",
                         "width", "height",
                         "saturate_recip",
                         NULL };

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "sii|i",
                                     keywords,
                                     &filename_raw,
                                     &width, &height,
                                     &saturate_recip,
                                     NULL))
        goto done;

    if( width < 0 || height < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "width, height must be >0");
        goto done;
    }

    if( saturate_recip <= 2)
    {
        PyErr_SetString(PyExc_RuntimeError, "saturate_recip must be >2");
        goto done;
    }

    out_rgb = (PyArrayObject*)PyArray_SimpleNew(3, ((npy_intp[]){height,width,3}),
                                                NPY_UINT8);
    if(out_rgb == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to allocate out_rgb(width=%d, height=%d)", width, height);
        goto done;
    }

    if(!raw_convert( PyArray_DATA(out_rgb),
                     filename_raw, width, height, saturate_recip ))
    {
        PyErr_SetString(PyExc_RuntimeError, "raw_convert() failed!");
        goto done;
    }

    result = (PyObject*)out_rgb;

 done:
    if(result == NULL)
        Py_XDECREF(out_rgb);
    RESET_SIGINT();
    return result;
}

static const char py_raw_convert_docstring[] =
#include "raw_convert.docstring.h"
    ;
static PyMethodDef methods[] =
    {
     PYMETHODDEF_ENTRY(py_, raw_convert, METH_VARARGS | METH_KEYWORDS),
     {}
    };


static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "_raw_convert",
     "QUAKES raw converter",
     -1,
     methods
    };

__attribute__((visibility("default")))
PyMODINIT_FUNC PyInit__raw_convert(void)
{
    import_array();
    return PyModule_Create(&module_def);
}
