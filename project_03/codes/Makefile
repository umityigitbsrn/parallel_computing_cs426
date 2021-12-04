all: kdtree_serial

#kdtree_perf: util.o src/kdtree_serial.c
#	g++ -fno-omit-frame-pointer -o kdtree_perf src/kdtree_serial.cpp util.o

kdtree_serial: util.o src/kdtree_serial.c
	gcc -g -o kdtree_serial src/kdtree_serial.c util.o

util.o: src/util.c src/util.h
	gcc -g -c src/util.c

.PHONY: clean
clean:
	rm util.o kdtree_serial #kdtree_perf
