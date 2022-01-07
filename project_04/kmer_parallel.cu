//
// Created by umityigitbsrn on 7.01.2022.
//

#include "util_parallel.h"
#include <time.h>

int num_kmer_in_read(char *read, int k);

int main(int argc, char** argv)
{
    if(argc != 5) {
        printf("Wrong argments usage: ./kmer_parallel [REFERENCE_FILE] [READ_FILE] [k] [OUTPUT_FILE]\n" );
    }

    FILE *fp;
    int k;

    //malloc instead of allocating in stack
    char *reference_str = (char*) malloc(MAX_REF_LENGTH * sizeof(char));
    char *read_str = (char*) malloc(MAX_READ_LENGTH * sizeof(char));

    char *reference_filename, *read_filename, *output_filename;
    int reference_length;

    reference_filename = argv[1];
    read_filename = argv[2];
    k = atoi(argv[3]);
    output_filename = argv[4];

    fp = fopen(reference_filename, "r");
    if (fp == NULL) {
        printf("Could not open file %s!\n",reference_filename);
        return 1;
    }

    if (fgets(reference_str, MAX_REF_LENGTH, fp) == NULL) { //A single line only
        printf("Problem in file format!\n");
        return 1;
    }
    reference_str[strcspn(reference_str, "\n")] = 0; //Remove the trailing \n character

    reference_length = strlen(reference_str);

//    printf("Reference str is = %s\n", reference_str);
    fclose(fp);

    //Read queries
    StringList queries;

    initStringList(&queries, 3);  // initially 3 elements

    int success = read_file(read_filename,&queries);
//    if(success){
//        for(int i = 0; i < queries.used; i++) {
//            printf("read : %s, len: %zu\n", queries.array[i], strlen(queries.array[i]));
//        }
//    }


    ////////////////////////////////////////////////////////////////////////
    ////////////// THIS IS A GOOD PLACE TO DO YOUR COMPUTATIONS ////////////
    ////////////////////////////////////////////////////////////////////////

    clock_t t;
    t = clock();
    // send from host to device
    int len_read = strlen(queries.array[0]);

    // reference str
    char *dev_ref;
    int size = reference_length * sizeof(char);
    cudaMalloc(&dev_ref, size);
    cudaMemcpy(dev_ref, reference_str, size, cudaMemcpyHostToDevice);

    // string list
//    StringList *dev_str_list;
//    size = sizeof(StringList);
//    cudaMalloc(&dev_str_list, size);
//    cudaMemcpy(dev_str_list, &queries, size, cudaMemcpyHostToDevice);

    char *read_tmp = (char *) malloc(sizeof(char) * len_read * queries.used);
    int tmp_index = 0;
    for (int i = 0; i < queries.used; ++i){
        for (int j = 0; j < len_read; ++j){
            read_tmp[tmp_index] = queries.array[i][j];
            tmp_index++;
        }
    }

//    printf("read_tmp: %s\n", read_tmp);

    char *dev_read;
    size = len_read * sizeof(char) * queries.used;
    cudaMalloc(&dev_read, size);
    cudaMemcpy(dev_read, read_tmp, size, cudaMemcpyHostToDevice);

    // output
    size = sizeof(int) * (len_read - k + 1) * queries.used;
//    size = sizeof(int) * (len_read - k + 1);
    int *dev_out;
    cudaMalloc(&dev_out, size);

    // kernel function

    // set thread and block numbers
    
    unsigned int num_of_threads = (len_read - k + 1) * queries.used;
    unsigned int num_of_blocks = num_of_threads / 1024;
    unsigned int remainder = num_of_threads % 1024;
    
    if (remainder > 0)
	    num_of_blocks++;

    dim3 dim_grid(num_of_blocks, 1);
    dim3 dim_block(1024, 1);
//    dim3 dim_block((len_read - k + 1), 1);

//    kernel_fnc<<<dim_grid, dim_block>>>(dev_ref, dev_str_list, dev_out, k, reference_length, len_read);
    kernel_fnc<<<dim_grid, dim_block>>>(dev_ref, dev_read, dev_out, k, reference_length, len_read, num_of_threads);
//    cudaThreadSynchronize();

    // device to host
    int *host_out = (int *) malloc(sizeof(int) * size);
    cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost);

//    for (int i = 0; i < queries.used; ++i){
//        prin f("[ ");
//        for (int j = 0; j < (len_read - k + 1); ++j){
//            printf("%d ", host_out[i * (len_read - k + 1) + j]);
//        }
//        printf("]\n");
//    }


//    for (int i = 0; i < queries.used; ++i){
//        printf("source: %s\n", queries.array[i]);
//        for (int j = k; j <= strlen(queries.array[i]); ++j){
//            substring(read_str, queries.array[i], j - k, j);
//            printf("%s", read_str);
//
//            if (host_out[i * (len_read - k + 1) + (j - k)] != -1){
//                printf("( count=1 at index %d )\n", host_out[i * (len_read - k + 1) + (j - k)]);
//            } else {
//                printf("( count=0 at index -1 )\n");
//            }
//        }
//    }

    t = clock() - t;
    double elapsed_time = ((double) t)/CLOCKS_PER_SEC;
    printf("-------elapsed time - parallel: %f-------\n", elapsed_time);

    write_file(output_filename, host_out, queries.used, len_read - k + 1);

    // f ee cuda
    cudaFree(dev_ref);
//    cudaFree(dev_str_list);
    cudaFree(dev_read);
    cudaFree(dev_out);

    //Free up
    freeStringList(&queries);

    free(reference_str);
    free(read_str);
    free(read_tmp);
    free(host_out);
}
