PYTHON_INCLUDE ?= /usr/include/python3.6m 
PYTHON_LIB ?= /usr/lib/python3.6

CFLAGS = -fPIC
LDFLAGS ?= -shared

default: _pidc.so pidc.py

_pidc.so: pidc_wrap.c pidc.o pidc_wrap.o
	ld $(LDFLAGS) pidc.o pidc_wrap.o -o _pidc.so -L$(PYTHON_LIB)

pidc_wrap.c pidc.py: pid.i
	swig -python -o pidc_wrap.c pid.i

pidc.o: pid.c
	gcc -c $(CFLAGS) pid.c -I$(PYTHON_INCLUDE)
	mv pid.o pidc.o

pidc_wrap.o: pidc_wrap.c
	gcc -c $(CFLAGS) pidc_wrap.c -I$(PYTHON_INCLUDE) -L$(PYTHON_LIB)

clean:
	rm pidc_wrap.c pidc.o pidc_wrap.o _pidc.so pidc.py

install:
	mv pidc.py ../components
	mv _pidc.so ../components
