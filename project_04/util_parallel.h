#define MAX_READ_LENGTH 200
#define MAX_REF_LENGTH 2000000
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;

/**
 * usage:
   Array a;
   char* sample = "example";

   initArray(&a, 5);  // initially 5 elements
   for (int i = 0; i < 100; i++)
    insertArray(&a, i);  // automatically resizes as necessary
   printf("%s\n", a.array[9]);  // print 10th element
   printf("%d\n", a.used);  // print number of elements
   freeArray(&a);
 *
 */
typedef struct {
    char **array;
    size_t used;
    size_t size;
} StringList;

void initStringList(StringList *a, size_t initialSize);

void insertStringList(StringList *a, char *element);

void freeStringList(StringList *a);

int read_file(char *file_name, StringList *sequences);

//void substring(char *source, int begin_index, int end_index);

void substring(char *destination, char *source, int begin_index, int end_index);

__device__ int d_strlen(const char* string);

__device__ int d_strncmp( const char * s1, const char * s2, size_t n);

//__global__ void kernel_fnc(char * dev_ref, StringList * dev_str_list, int *dev_out, int k, int len_ref, int len_read);
__global__ void kernel_fnc(char * dev_ref, char *dev_read, int *dev_out, int k, int len_ref, int len_read, unsigned int num_of_threads);
