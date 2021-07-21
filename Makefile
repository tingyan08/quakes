PYTHON_VERSION_FOR_EXTENSIONS := 3

include /usr/include/mrbuild/Makefile.common.header

PROJECT_NAME := raw-convert
ABI_VERSION  := 1
TAIL_VERSION := 1

LIB_SOURCES := raw-convert.c raw-convert-cpp.cc

# The opencv people (or maybe the Debian people?) have renamed the opencv.pc
# file in opencv 4. So now I look for both version 4 and the default. What will
# happen with opencv5? We'll see!
CXXFLAGS_CV := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
LDLIBS_CV   := $(shell pkg-config --libs   opencv4 2>/dev/null || pkg-config --libs   opencv 2>/dev/null)
CCXXFLAGS += $(CXXFLAGS_CV)
LDLIBS    += $(LDLIBS_CV) -lpthread

CCXXFLAGS += -fvisibility=hidden

CFLAGS    += -std=gnu99
CCXXFLAGS += -Wno-unused-function -Wno-missing-field-initializers -Wno-unused-parameter -Wno-strict-aliasing -Wno-int-to-pointer-cast -Wno-unused-variable

########## python stuff

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
raw-convert-pywrap.o: CFLAGS += -Wno-cast-function-type

raw-convert-pywrap.o: CCXXFLAGS += $(PY_MRBUILD_CFLAGS)
raw-convert-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

_raw_convert$(PY_EXT_SUFFIX): raw-convert-pywrap.o libraw-convert.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@

DIST_PY3_MODULES := _raw_convert$(PY_EXT_SUFFIX)

all: _raw_convert$(PY_EXT_SUFFIX)

include /usr/include/mrbuild/Makefile.common.footer
