#include <stdio.h>
#include <stdlib.h>

#include "util.h"

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

    for (size_t i = 0; i < numPoints; i++) {
        printPoint(points[i], dimension);
        printf("\n");
    }
    printf("Queries are:\n");
    for (int i = 0; i < numQueries; i++) {
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

    root = kd_tree_construct(points, 0, numPoints, dimension, 0);
    printf("Printing the tree..\n");
    printTree(root, dimension);

    free(points);
    free(queries);
    return 0;

    Result *results = malloc(numQueries * sizeof(Result));

    /////////////////////////////////////////////////
    ///////         Search Query Ranges  ////////////
    /////////////////////////////////////////////////

    for (int i = 0; i < numQueries; i++) {
        results[i].size = i; //Some fake values
    }

    printf("Results:\n");

    for (int i = 0; i < numQueries; i++) {
        printf("%d:", results[i].size);
        for (int j = 0; j < results[i].size; j++) {
            //printPoint(results[i].resultPoints[j],dimension); //YOU DON'T NEED TO COMPUTE THE POINTS
            printf(" ");
        }
        printf("\n");
    }

    write_results(argv[3], results, numQueries, dimension);


    free(points); //points should be deleted after we process the queries, kd-tree nodes points to elements in the points array


    return 0;
}
