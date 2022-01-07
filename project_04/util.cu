#include "util.h"

//void init_list(dyn_list *l, size_t initial_size){
//    l->item_arr = (int *) malloc()
//}

void initStringList(StringList *a, size_t initialSize) {
    a->array = (char**) malloc(initialSize * sizeof(char*));
    for (int i = 0; i < initialSize; i++) {
        a->array[i] = (char*) malloc(MAX_READ_LENGTH * sizeof(char));
    }
    a->used = 0;
    a->size = initialSize;
}

void insertStringList(StringList *a, char *element) {
    // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
    // Therefore a->used can go up to a->size
    if (a->used == a->size) {
        a->size *= 2;
        a->array = (char**) realloc(a->array, a->size * sizeof(char*));
        for (int i = (a->size)/2; i < a->size; i++) {
            a->array[i] = (char*) malloc(MAX_READ_LENGTH * sizeof(char));
        }
    }
    strcpy(a->array[a->used++], element);
}

void freeStringList(StringList *a) {
    for(int i = 0; i < a->size; i++) {
        free(a->array[i]);
    }
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

int read_file(char *file_name, StringList *sequences) {
    FILE *fp;
    fp = fopen(file_name, "r");
    if(fp) {
        char *line = (char *) malloc( MAX_READ_LENGTH * sizeof(char));
        while (fgets(line, MAX_READ_LENGTH, fp) != NULL) { //A single line only
            //printf("%s", line);
            line[strcspn(line, "\n")] = 0; //Remove the trailing \n character
            insertStringList(sequences,line);
        }
        free(line);
        fclose(fp);
        return 1;
    }
    return 0; //Means error
}

//Do not use substring methods for cuda kernel, try a more primitive approach
//without memory operations for performance
//void substring(char *source, int begin_index, int end_index)
//{
//    // copy n characters from source string starting from
//    // beg index into destination
//    memmove(source, (source + begin_index), end_index-begin_index);
//    source[end_index-begin_index] = '\0';
//}

void substring(char *destination, char *source, int begin_index, int end_index)
{
    // copy n characters from source string starting from
    // beg index into destination
    memcpy(destination, (source + begin_index), end_index-begin_index);
    destination[end_index-begin_index] = '\0';
}

void init_hashtable(StringList str_list, my_unordered_map *map){
    // initialize counts and lists
    for (int i = 0; i < str_list.used; ++i){
        hash_object tmp_obj;
        tmp_obj.count = 0;
        tmp_obj.index_arr = nullptr;
        tmp_obj.index_arr_index = 0;
        if ((*map).find(str_list.array[i]) == (*map).end())
            (*map)[str_list.array[i]] = tmp_obj;
    }

    // set counts
    for (int i = 0; i < str_list.used; ++i){
        (*map)[str_list.array[i]].count++;
    }

    // init index
    for (int i = 0; i < str_list.used; ++i){
        int count = (*map)[str_list.array[i]].count;
        (*map)[str_list.array[i]].index_arr = (int *) malloc(sizeof (int) * count);
//        printf("count: %d\n", count);
    }

    //set index values
    for (int i = 0; i < str_list.used; ++i){
        (*map)[str_list.array[i]].index_arr[(*map)[str_list.array[i]].index_arr_index++] = i;
    }

//    print_map(*map);
};

void print_map(my_unordered_map map){
    my_unordered_map::iterator it = map.begin();
    while (it != map.end()){
        const char *key = it->first;
        printf("[%s] : ", key);

        hash_object object = it->second;

        printf("( ");
        for (int i = 0; i < object.count; ++i) {
            printf("%d ", object.index_arr[i]);
        }

        printf(")\n");
        it++;
    }
}

void free_map(my_unordered_map map){
    my_unordered_map::iterator it = map.begin();
    while (it != map.end()){
        hash_object object = it->second;
        free(object.index_arr);
        it++;
    }
}


//* You might use these for some simple string operations in GPU
//* Put these code into your program
__device__ int d_strlen(const char* string){
    int length = 0;
    while (*string++)
        length++;
    return (length);
}

//Compares string until nth character
__device__ int d_strncmp( const char * s1, const char * s2, size_t n )
{
    while ( n && *s1 && ( *s1 == *s2 ) )
    {
        ++s1;
        ++s2;
        --n;
    }
    if ( n == 0 )
    {
        return 0;
    }
    else
    {
        return ( *(unsigned char *)s1 - *(unsigned char *)s2 );
    }
}

//__global__ void kernel_fnc(char *dev_ref, StringList *dev_str_list, int *dev_out, int k, int len_ref, int len_read){
//    printf("ENTER\n");
//    char *ref_it = dev_ref;
////    printf("dev_str_list item 0: %s\n", (*dev_str_list).array[0]);
//    char *read_thread_ptr = (*dev_str_list).array[threadIdx.y] + threadIdx.x;
//
//    int l;
//    for (l = 0; l < len_ref - k + 1; l++){
//        if (d_strncmp(ref_it, read_thread_ptr, k) == 0){
//            dev_out[(len_read - k + 1) * threadIdx.y + threadIdx.x] = l;
//            break;
//        }
//
//        ref_it++;
//    }
//
//    if (l == len_ref - k + 1)
//        dev_out[(len_read - k + 1) * threadIdx.y + threadIdx.x] = -1;
//}

__global__ void kernel_fnc(char *dev_ref, char *dev_read, int *dev_out, int k, int len_ref, int len_read){
    char *ref_it = dev_ref;
//    printf("dev_str_list item 0: %s\n", (*dev_str_list).array[0]);
    char *read_thread_ptr = dev_read + threadIdx.x;

    int l;
    for (l = 0; l < len_ref - k + 1; l++){
        if (d_strncmp(ref_it, read_thread_ptr, k) == 0){
            dev_out[threadIdx.x] = l;
            break;
        }

        ref_it++;
    }

    if (l == len_ref - k + 1)
        dev_out[threadIdx.x] = -1;
}