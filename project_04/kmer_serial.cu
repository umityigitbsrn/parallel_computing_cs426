//
// Created by umityigitbsrn on 2.01.2022.
//

#include "util.h"

int num_kmer_in_read(char *read, int k);

int main(int argc, char** argv)
{
    if(argc != 5) {
        printf("Wrong argments usage: ./kmer_serial [REFERENCE_FILE] [READ_FILE] [k] [OUTPUT_FILE]\n" );
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

    printf("Reference str is = %s\n", reference_str);
    fclose(fp);

    //Read queries
    StringList queries;

    initStringList(&queries, 3);  // initially 3 elements

    int success = read_file(read_filename,&queries);
    if(success){
        for(int i = 0; i < queries.used; i++) {
            printf("read : %s\n", queries.array[i]);
        }
    }

    // substring test
//    int i, j;
//
//    printf("reference: %s\n", reference_str);
//    for (j = k; j <= reference_length; ++j){
//        substring(read_str, reference_str, j - k, j);
//        printf("%s\n", read_str);
//    }
//
//    for (i = 0; i < queries.used; ++i){
//        printf("source: %s\n", queries.array[i]);
//        for (j = k; j <= strlen(queries.array[i]); ++j){
//            substring(read_str, queries.array[i], j - k, j);
//            printf("%s\n", read_str);
//        }
//    }

    ////////////////////////////////////////////////////////////////////////
    ////////////// THIS IS A GOOD PLACE TO DO YOUR COMPUTATIONS ////////////
    ////////////////////////////////////////////////////////////////////////

    StringList ref_str_list;
    initStringList(&ref_str_list, 3);

    for (int j = k; j <= reference_length; ++j){
        substring(read_str, reference_str, j - k, j);
        insertStringList(&ref_str_list, read_str);
    }

    my_unordered_map map;
    init_hashtable(ref_str_list, &map);
//    print_map(map);

    for (int i = 0; i < queries.used; ++i){
        printf("source: %s\n", queries.array[i]);
        for (int j = k; j <= strlen(queries.array[i]); ++j){
            substring(read_str, queries.array[i], j - k, j);
            printf("%s", read_str);

            if (map.find(read_str) != map.end()){
                printf("( count=%d at index ", map[read_str].count);
                for (int l = 0; l < map[read_str].count; ++l)
                    printf("%d ", map[read_str].index_arr[l]);
                printf(")\n");
            } else {
                printf("( count=0 at index -1 )\n");
            }
        }
    }

    //Free up
    freeStringList(&queries);
    freeStringList(&ref_str_list);

    free_map(map);

    free(reference_str);
    free(read_str);
}

