NVCC=nvcc
CUDAFLAGS= -std=c++14 -I .
all: kmer_serial kmer_parallel

kmer_serial: kmer_serial.cu util.cu util.h
	$(NVCC) $(CUDAFLAGS) -g -o kmer_serial kmer_serial.cu util.cu

kmer_parallel: kmer_parallel.cu util_parallel.cu util_parallel.h
	$(NVCC) $(CUDAFLAGS) -g -o kmer_parallel kmer_parallel.cu util_parallel.cu

clean:
	rm -rf kmer_serial kmer_parallel *.o
