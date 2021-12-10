all: kdtree_serial kdtree_perf kdtree_parallel

kdtree_perf: util.o src/kdtree_serial.c
	gcc -fno-omit-frame-pointer -o kdtree_perf src/kdtree_serial.c util.o -fopenmp

kdtree_serial: util.o src/kdtree_serial.c
	gcc -g -o kdtree_serial src/kdtree_serial.c util.o -fopenmp

util.o: src/util.c src/util.h
	gcc -g -c src/util.c

kdtree_parallel: util.o src/kdtree_parallel.c
	gcc -g -o kdtree_parallel src/kdtree_parallel.c util.o -fopenmp

.PHONY: clean
clean:
	rm util.o kdtree_serial kdtree_perf kdtree_parallel
