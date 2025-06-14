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

#define N (64 * 1024 * 1024)
#define MAX_STRIDE 16

const char* kernelSource =
"__kernel void kernelB(__global char* A, __global char* B, int stride) {\n"
"    int i = get_global_id(0);\n"
"    A[i] = B[i * stride];\n"
"}\n";

void measureKernelExecutionTime(cl_context context, cl_device_id device, cl_command_queue queue,
                               cl_program program, cl_kernel kernel, char* A, char* B, int stride) {
    cl_mem bufferA, bufferB;
    cl_int err;

    bufferA = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (N / 16) * sizeof(char), NULL, &err);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(char), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, (N / 16) * sizeof(char), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, N * sizeof(char), B, 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err = clSetKernelArg(kernel, 2, sizeof(int), &stride);

    cl_event event;
    size_t globalSize = N / 16;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);

    cl_ulong startTime, endTime;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);

    clReleaseEvent(event);

    double executionTime = (endTime - startTime) / 1.0e9;

    printf("Stride: %d\tExecution Time: %.6f seconds\n", stride, executionTime);

    err = clEnqueueReadBuffer(queue, bufferA, CL_TRUE, 0, (N / 16) * sizeof(char), A, 0, NULL, NULL);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
}

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;
    char* A = (char*)malloc((N / 16) * sizeof(char));
    char* B = (char*)malloc(N * sizeof(char));

    for (int i = 0; i < N; i++) {
        B[i] = i;
    }

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "kernelB", &err);

    for (int stride = 1; stride <= MAX_STRIDE; stride++) {
        measureKernelExecutionTime(context, device, queue, program, kernel, A, B, stride);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A);
    free(B);

    return 0;
}