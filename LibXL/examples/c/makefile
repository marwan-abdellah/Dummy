CC = gcc

ARCH = $(shell getconf LONG_BIT)
ifeq ($(ARCH), 32)
  LIBPATH = ../../lib
else
  LIBPATH = ../../lib64
endif

CFLAGS = -I ../../include_c -L $(LIBPATH) -lxl -Wl,-rpath,$(LIBPATH)

all: generate extract edit custom format invoice performance

generate: generate.c
	$(CC) -o generate $(CFLAGS) generate.c

extract: extract.c
	$(CC) -o extract $(CFLAGS) extract.c

edit: edit.c
	$(CC) -o edit $(CFLAGS) edit.c

custom: custom.c
	$(CC) -o custom $(CFLAGS) custom.c

format: format.c
	$(CC) -o format $(CFLAGS) format.c

invoice: invoice.c
	$(CC) -o invoice $(CFLAGS) invoice.c

performance: performance.c
	$(CC) -o performance $(CFLAGS) performance.c

clean:
	rm -f generate extract edit custom format invoice performance *.xls *.xlsx

