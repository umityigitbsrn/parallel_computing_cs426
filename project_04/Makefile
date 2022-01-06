NVCC=nvcc
CUDAFLAGS= -std=c++14 -I .
all: kmer_serial

kmer_serial: kmer_serial.cu util.cu util.h
	$(NVCC) $(CUDAFLAGS) -g -o kmer_serial kmer_serial.cu util.cu

clean:
	rm -rf kmer_serial *.o
