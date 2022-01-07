NVCC=nvcc
CUDAFLAGS= -std=c++14 -I .
all: kmer_serial kmer_parallel

kmer_serial: kmer_serial.cu util.cu util.h
	$(NVCC) $(CUDAFLAGS) -g -o kmer_serial kmer_serial.cu util.cu

kmer_parallel: kmer_parallel.cu util.cu util.h
	$(NVCC) $(CUDAFLAGS) -g -G -o kmer_parallel kmer_parallel.cu util.cu

clean:
	rm -rf kmer_serial kmer_parallel *.o
