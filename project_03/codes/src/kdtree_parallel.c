#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include <time.h>

#include "omp.h"

void print_range(Range range, int dimension){
    int i;
    printf("left point: ( ");
    for (i = 0; i < dimension; ++i){
        printf("%d ", range.leftPoint[i]);
    }
    printf(") ");
    printf("right point: ( ");
    for (i = 0; i < dimension; ++i){
        printf("%d ", range.leftPoint[i]);
    }
    printf(")\n");
}

Node *kd_tree_construct_parallel(Point *points, unsigned long init, unsigned long n, int dimension, int k){
//    printf("init: %u, n: %u, dimension: %d, k: %d\n");
//    printf("thread %d/%d construct tree btw init: %lu, n: %lu\n", omp_get_thread_num(), omp_get_num_threads(), init, n);
    if (init >= n)
        return NULL;
    Node *N = newNode(select_median_point(points, init, n, k));
//    printPoint(N->data, dimension);
//    printf("\n");
    if (init + 1 == n)
        return N;

    k = (k + 1) % dimension;

    #pragma omp task shared(N, points, dimension) firstprivate(k, init, n)
    N->left = kd_tree_construct_parallel(points, init, (n + init) / 2, dimension, k);

    #pragma omp task shared(N, points, dimension) firstprivate(k, init, n)
    N->right = kd_tree_construct_parallel(points, ((n + init) / 2) + 1, n, dimension, k);

    #pragma omp taskwait
    return N;
}

void range_search_parallel(Node *node, int dimension, int k, Range *range, Result *result){
    if (node != NULL){
//        print_range(range, dimension);
        int i;
        char dimension_flag = 1;
        for (i = 0; i < dimension && dimension_flag; ++i) {
//            printf("node->data[%d] = %d\n", i, node->data[i]);
            dimension_flag = (char) (node->data[i] >= range->leftPoint[i] && node->data[i] <= range->rightPoint[i]);
        }
        if (dimension_flag) {
            #pragma omp critical
            {
//                printf("thread %d/%d is in critical section\n", omp_get_thread_num(), omp_get_num_threads());
                *result = *result + 1;
            }
        }

        int k_next = (k + 1) % dimension;

        if (node->data[k] >= range->leftPoint[k]) {
            #pragma omp task shared(dimension, range, result) firstprivate(node, k)
            {
//                printf("node-left: ");
//                printPoint(node->left->data, dimension);
//                printf("\n");
                range_search_parallel(node->left, dimension, k_next, range, result);
            }
        }

        if (node->data[k] <= range->rightPoint[k]) {
            #pragma omp task shared(dimension, range, result) firstprivate(node, k)
            {
//                printf("node-right: ");
//                printPoint(node->right->data, dimension);
//                printf("\n");
                range_search_parallel(node->right, dimension, k_next, range, result);
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Wrong argments usage: kdtree_serial [INPUT_POINTS_FILE] [INPUT_QUERIES_FILE] [OUTPUT_FILE]\n");
        return -1;
    }

//    omp_set_dynamic(0);
//    omp_set_num_threads(4);

    double start, end;
    start = omp_get_wtime();

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
//    for (i = 0; i < numPoints; i++) {
//        printPoint(points[i], dimension);
//        printf("\n");
//    }
//    printf("Queries are:\n");
//    for (i = 0; i < numQueries; i++) {
//        printf("Query %d, from ", i);
//        printPoint(queries[i].leftPoint, dimension);
//        printf(" to ");
//        printPoint(queries[i].rightPoint, dimension);
//        printf("\n");
//    }

    Node *root;

    /////////////////////////////////////////////////
    ///////         CREATE K-d Tree      ////////////
    /////////////////////////////////////////////////

    double f_start, f_end;
    f_start = omp_get_wtime();
    #pragma omp parallel shared(points, numPoints, dimension)
    {
//        printf("thread %d/%d created\n", omp_get_thread_num(), omp_get_num_threads());
        #pragma omp single
        {
//            printf("thread %d/%d entered\n", omp_get_thread_num(), omp_get_num_threads());
            root = kd_tree_construct_parallel(points, 0, numPoints, dimension, 0);
        }
    }
    f_end = omp_get_wtime();
    printf("kd_tree construction operation executes in %f seconds\n", f_end - f_start);

//    printf("Printing the tree..\n");
//    printTree(root, dimension);


    Result *results = (Result *) malloc(numQueries * sizeof(Result));
    for (i = 0; i < numQueries; ++i)
        results[i] = 0;

    /////////////////////////////////////////////////
    ///////         Search Query Ranges  ////////////
    /////////////////////////////////////////////////

//    #pragma omp parallel shared(root, dimension, queries, results) private(i)
//    {
//        #pragma omp for schedule(auto) nowait
        f_start = omp_get_wtime();
//        for (i = 0; i < numQueries; ++i) {
            #pragma omp parallel shared(root, dimension, queries, results)
            {
                #pragma omp master
                {
                    int j;
                    for (j = 0; j < numQueries; ++j)
                        range_search_parallel(root, dimension, 0, &queries[j], &results[j]);
                }
            }
//        }
        f_end = omp_get_wtime();
        printf("range search operation executes in %f seconds\n", f_end - f_start);
//        for (i = 0; i < numQueries; ++i)
//            printf("results[%d]: %d\n", i, results[i]);
//    }
//    printf("Results:\n");
//    for (i = 0; i < numQueries; ++i)
//        printf("%d\n", results[i]);

    write_results(argv[3], results, numQueries, dimension);
    free_kd_tree(root);
    free(points);
    free(queries);
    free(results);

    end = omp_get_wtime();
    printf("The program took %f seconds to execute\n", end - start);

    return 0;
}
