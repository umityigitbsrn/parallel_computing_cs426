all: kdtree_serial kdtree_parallel

kdtree_perf: util.o src/kdtree_serial.cpp
	g++ -fno-omit-frame-pointer -o kdtree_perf src/kdtree_serial.cpp util.o

kdtree_serial: util.o src/kdtree_serial.cpp
	g++ -o kdtree_serial src/kdtree_serial.cpp util.o

kdtree_parallel: util.o src/kdtree_parallel.cpp
	g++ -fopenmp -o kdtree_parallel src/kdtree_parallel.cpp util.o

util.o: src/util.c src/util.h
	g++ -c src/util.c
    

.PHONY: clean
clean:
	rm util.o kdtree_serial kdtree_parallel kdtree_perf
