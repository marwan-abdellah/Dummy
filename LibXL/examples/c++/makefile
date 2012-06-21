CC = g++

ARCH = $(shell getconf LONG_BIT)
ifeq ($(ARCH), 32)
  LIBPATH = ../../lib
else
  LIBPATH = ../../lib64
endif

CFLAGS = -I ../../include_cpp -L $(LIBPATH) -lxl -Wl,-rpath,$(LIBPATH)

all: write read edit custom format invoice performance

write: write.cpp
	$(CC) -o write $(CFLAGS) write.cpp

read: read.cpp
	$(CC) -o read $(CFLAGS) read.cpp

edit: edit.cpp
	$(CC) -o edit $(CFLAGS) edit.cpp

custom: custom.cpp
	$(CC) -o custom $(CFLAGS) custom.cpp

format: format.cpp
	$(CC) -o format $(CFLAGS) format.cpp

invoice: invoice.cpp
	$(CC) -o invoice $(CFLAGS) invoice.cpp

performance: performance.cpp
	$(CC) -o performance $(CFLAGS) performance.cpp

clean:
	rm -f write read edit custom format invoice performance *.xls *.xlsx

