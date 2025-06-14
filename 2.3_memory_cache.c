// OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define MATRIX_SIZE 1024
//#define MATRIX_SIZE 2048
#define WORKGROUP_SIZES { 1, 2, 4, 8, 16, 32 }

const char* kernelSource =
    "__kernel void matrixMultiply(__global float* A, __global float* B, __global float* C, int width) {\n"
    "  int row = get_global_id(0);\n"
    "  int col = get_global_id(1);\n"
    "  int localRow = get_local_id(0);\n"
    "  int localCol = get_local_id(1);\n"
    "  int localSize = get_local_size(0);\n"
    "  __local float localA[32][32];\n"
    "  __local float localB[32][32];\n"
    "  float sum = 0.0f;\n"
    "  for (int w = 0; w < width / 32; w++) {\n"
    "    localA[localRow][localCol] = A[row * width + (w * 32 + localCol)];\n"
    "    localB[localRow][localCol] = B[(w * 32 + localRow) * width + col];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int k = 0; k < 32; k++) {\n"
    "      sum += localA[localRow][k] * localB[k][localCol];\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  C[row * width + col] = sum;\n"
    "}\n";

double getCurrentTimestamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main() {
    float* matrixA = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* matrixB = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrixA[i] = (float)rand() / RAND_MAX;
        matrixB[i] = (float)rand() / RAND_MAX;
    }

    float* matrixC = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    cl_device_id device = NULL;
    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        cl_device_id* devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        if (numDevices > 0) {
            device = devices[0];
            free(devices);
            break;
        }
        free(devices);
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrixA, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrixB, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", NULL);

    int matrixSize = MATRIX_SIZE;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &matrixSize);

    size_t workgroupSizes[] = WORKGROUP_SIZES;

    cl_event event;

    for (int i = 0; i < sizeof(workgroupSizes) / sizeof(workgroupSizes[0]); i++) {
        size_t localSize[] = { workgroupSizes[i], workgroupSizes[i] };

        size_t globalSize[] = { MATRIX_SIZE, MATRIX_SIZE };
        clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, localSize, 0, NULL, &event);

        clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrixC, 0, NULL, NULL);

        clWaitForEvents(1, &event);

        cl_ulong startTime, endTime;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
        double executionTime = (endTime - startTime) * 1e-9;

        printf("Matrix Size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Execution Time: %.6f s\n", executionTime);
        printf("Workgroup Size: %zux%zu\n", localSize[0], localSize[1]);
        printf("------------------------\n");
    }

    clReleaseEvent(event);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(platforms);

    return 0;
}