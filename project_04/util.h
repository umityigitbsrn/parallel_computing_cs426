#define MAX_READ_LENGTH 200
#define MAX_REF_LENGTH 2000000
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unordered_map>

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

typedef struct {
    int count;
    int *index_arr;
    int index_arr_index;
} hash_object;

// TODO: reference
template <class _Tp>
struct my_equal_to : public binary_function<_Tp, _Tp, bool>
{
    bool operator()(const _Tp& __x, const _Tp& __y) const
    { return strcmp( __x, __y ) == 0; }
};

struct Hash_Func{
    //BKDR hash algorithm
    int operator()(char * str)const
    {
        int seed = 131;//31  131 1313 13131131313 etc//
        int hash = 0;
        while(*str)
        {
            hash = (hash * seed) + (*str);
            str ++;
        }

        return hash & (0x7FFFFFFF);
    }
};

typedef unordered_map<char*, hash_object, Hash_Func,  my_equal_to<char*> > my_unordered_map;

void initStringList(StringList *a, size_t initialSize);

void insertStringList(StringList *a, char *element);

void freeStringList(StringList *a);

int read_file(char *file_name, StringList *sequences);

void init_hashtable(StringList str_list, my_unordered_map *map);

void print_map(my_unordered_map map);

void free_map(my_unordered_map map);
//void substring(char *source, int begin_index, int end_index);

void substring(char *destination, char *source, int begin_index, int end_index);

void write_file(const char *filename, int *result, int word_count, int word_index_count);