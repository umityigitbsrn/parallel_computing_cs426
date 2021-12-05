#include <stdio.h>
#include <stdlib.h>

#include "util.h"

#include "omp.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Wrong argments usage: kdtree_serial [INPUT_POINTS_FILE] [INPUT_QUERIES_FILE] [OUTPUT_FILE]\n");
        return -1;
    }

    int dimension, numPoints, numQueries, dimensionQ;
    Range *queries;
    Point *points;

    int success = read_points_file(argv[1], &numPoints, &dimension, &points);
    if (success) {
        printf("Successful, number of points are %d, dimension is %d\n", numPoints, dimension);
    } else {
        printf("Unsuccessful file read operation for points file, exiting the program!\n");
        return -1;
    }
    success = read_queries_file(argv[2], &numQueries, &dimensionQ, &queries);
    if (success) {
        printf("Successful, number of queries are  are %d\n", numQueries);
    } else {
        printf("Unsuccessful file read operation for queries file, exiting the program!\n");
        return -1;
    }
    if (dimension != dimensionQ) {
        printf("Points dimensions and query points dimensions don't match!\n");
        return -1;
    }

    int i;
    for (i = 0; i < numPoints; i++) {
        printPoint(points[i], dimension);
        printf("\n");
    }
    printf("Queries are:\n");
    for (i = 0; i < numQueries; i++) {
        printf("Query %d, from ", i);
        printPoint(queries[i].leftPoint, dimension);
        printf(" to ");
        printPoint(queries[i].rightPoint, dimension);
        printf("\n");
    }

    Node *root;

    /////////////////////////////////////////////////
    ///////         CREATE K-d Tree      ////////////
    /////////////////////////////////////////////////

    #pragma omp parallel shared(points, numPoints, dimension)
    {
        #pragma omp single
        root = kd_tree_construct_parallel(points, 0, numPoints, dimension, 0);
    }
    printf("Printing the tree..\n");
    printTree(root, dimension);


    Result *results = (Result *) malloc(numQueries * sizeof(Result));
    for (i = 0; i < numQueries; ++i)
        results[i] = 0;

    /////////////////////////////////////////////////
    ///////         Search Query Ranges  ////////////
    /////////////////////////////////////////////////
    for (i = 0; i < numQueries; ++i)
        range_search(root, dimension, 0, queries[i], &results[i]);

    printf("Results:\n");
    for (i = 0; i < numQueries; ++i)
        printf("%d\n", results[i]);

    write_results(argv[3], results, numQueries, dimension);
    free_kd_tree(root);
    free(points);
    free(queries);
    free(results);
    return 0;
}
