#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MATRIX_SIZE 1024
//#define MATRIX_SIZE 2048
#define WORKGROUP_SIZE 32

const char* kernelSource =
    "__kernel void matrixMultiply(__global float* A, __global float* B, __global float* C, int width) {\n"
    "  int row = get_global_id(0);\n"
    "  int col = get_global_id(1);\n"
    "  float sum = 0.0f;\n"
    "  for (int k = 0; k < width; k++) {\n"
    "    sum += A[row * width + k] * B[k * width + col];\n"
    "  }\n"
    "  C[row * width + col] = sum;\n"
    "}\n";

int main() {
    float* matrixA = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float* matrixB = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrixA[i] = (float)rand() / RAND_MAX;
        matrixB[i] = (float)rand() / RAND_MAX;
    }

    float* matrixC = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    commandQueue = clCreateCommandQueueWithProperties(context, device, properties, &err);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrixA, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrixB, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, &err);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "matrixMultiply", &err);

    size_t globalWorkSize[2] = { MATRIX_SIZE, MATRIX_SIZE };

    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

    int matrixSize = MATRIX_SIZE;
    for (int workGroupSize = 1; workGroupSize <= maxWorkGroupSize; workGroupSize *= 2) {
    printf("Work Group Size: %dx%d\n", workGroupSize, workGroupSize);
    printf("Matrix Size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    size_t localWorkSize[2] = { workGroupSize, workGroupSize };

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferB);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferC);
    err = clSetKernelArg(kernel, 3, sizeof(int), &matrixSize);

    cl_event event;
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);

    err = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), matrixC, 0, NULL, NULL);

    clFinish(commandQueue);

    cl_ulong startTime, endTime;
    cl_ulong executionTime;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
    executionTime = endTime - startTime;

    printf("Execution Time: %.6f s\n", (double)executionTime * 1e-9);
    printf("-----------------------\n");
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
