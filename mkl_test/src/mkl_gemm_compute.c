#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "mkl_types.h"
#include "mkl_cblas.h"
#include <time.h>
#include <omp.h>

#ifndef IFLOAT
#define IFLOAT float
#endif
#define FLOAT float
#define GEMM cblas_sgemm

#define MAX_NUM 1.0

int main(int argc, char *argv[]) {
    int sizes[][3] = {
        // Llama3
        {4096, 128, 4096},
        {128, 8192, 4096},
        {128, 4096, 8192},
        {4096, 4096, 4096},

        // Gemma27B
        {4608, 256, 4096},
        {256, 8192, 4608},
        {256, 4608, 8192},
        {4608, 4608, 36864},

        // Gemma9B
        {3584, 256, 4096},
        {256, 8192, 3584},
        {256, 3584, 8192},
        {3584, 3584, 14336},

        // Gemma7B
        {3072, 256, 4096},
        {256, 8192, 3072},
        {256, 3072, 8192},
        {3072, 3072, 24576},

        // Gemma2B
        {2048, 256, 4096},
        {256, 8192, 2048},
        {256, 2048, 8192},
        {2048, 2048, 16384},
    };

    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    long nT = mkl_get_max_threads();
    printf("Number of threads: %ld\n", nT);
    mkl_set_num_threads(nT);

    IFLOAT *a, *b;
    FLOAT *c;
    FLOAT alpha = 2.0;
    FLOAT beta = 0.0;
    int m, n, k, lda, ldb, ldc;
    
    int loops = 10; 

    for (int index = 0; index < numSizes; index++) {
        m = sizes[index][0];
        n = sizes[index][1];
        k = sizes[index][2];

        lda = k;  // Since matrix A is not transposed
        ldb = n;  // Since matrix B is not transposed
        ldc = n;  // Standard leading dimension for matrix C

        a = (IFLOAT *)malloc(sizeof(IFLOAT) * m * k);
        b = (IFLOAT *)malloc(sizeof(IFLOAT) * k * n);
        c = (FLOAT *)malloc(sizeof(FLOAT) * m * n);

        if (a == NULL || b == NULL || c == NULL) {
            fprintf(stderr, "Out of Memory!!\n");
            exit(1);
        }

        for (int i = 0; i < m * k; i++) a[i] = (IFLOAT)rand() / (IFLOAT)RAND_MAX;
        for (int i = 0; i < k * n; i++) b[i] = (IFLOAT)rand() / (IFLOAT)RAND_MAX;
        for (int i = 0; i < m * n; i++) c[i] = 0;

        double start, end, time1, timeg;
        start = omp_get_wtime();
        for (int ite = 0; ite < loops; ite++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        end = omp_get_wtime();

        time1 = end - start;
        timeg = time1 / loops;


        double gflops = (2.0 * m * n * k) / (timeg * 1e9); // 2*m*n*k because each multiplication and addition counts as one operation
        
        printf("Matmul testcase no, %d, gflpos, %f, time, %f\n", index, gflops, timeg);

        free(a);
        free(b);
        free(c);
    }

    return 0;
}
