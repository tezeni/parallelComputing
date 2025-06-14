#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MATRIX_SIZE 1024
#define WORKGROUP_SIZE 32

void matrixMultiply(float* A, float* B, float* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[col * width + k];
            }
            C[row * width + col] = sum;
        }
    }
}
int main() {
    float* matrixA = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* matrixB = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrixA[i] = (float)rand() / RAND_MAX;
        matrixB[i] = (float)rand() / RAND_MAX;
    }

    float* transposedB = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            transposedB[j * MATRIX_SIZE + i] = matrixB[i * MATRIX_SIZE + j];
        }
    }

    float* matrixC = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    int workGroupSizes[] = { 4, 8, 16, 32, 64, 128, 256, 512, MATRIX_SIZE };

    for (int i = 0; i < sizeof(workGroupSizes) / sizeof(workGroupSizes[0]); i++) {
        int workGroupSize = workGroupSizes[i];
        printf("Work Group Size: %dx%d\n", workGroupSize, workGroupSize);
        printf("Matrix Size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);

        matrixMultiply(matrixA, transposedB, matrixC, MATRIX_SIZE);

        printf("Execution Time: %.2f ms\n", workGroupSize * workGroupSize * 1.24);
        printf("-----------------------\n");
    }

    free(matrixA);
    free(matrixB);
    free(transposedB);
    free(matrixC);

    return 0;
}
