//
//  main.c
//  OpenCL Routing
//
//  Created by Ning Shi on 6/29/13.
//  Copyright (c) 2013 Ning Shi. All rights reserved.
//

#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <OpenCL/OpenCL.h>

#include "kernels.cl.h"

#define NUM_VALUES (512 * 1000)

static void print_device_info(cl_device_id device)
{
    char name[128];
    char vendor[128];
    size_t max_workitem_size;
    size_t max_workgroup_size;

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                    &max_workgroup_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t),
                    &max_workitem_size, NULL);
    fprintf(stdout, "%s: %s, wg size: %zd, item size: %zd\n",
            vendor, name, max_workgroup_size, max_workitem_size);
}

static char *
load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

int main(int argc, const char * argv[])
{
    int err;                            // error code returned from api calls

    float in_data[NUM_VALUES];          // original data set given to device
    float out_data[NUM_VALUES];         // results returned from device

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device;                // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem in_buf;                      // device memory used for the input array
    cl_mem out_buf;                     // device memory used for the output array

    // Get device info
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to locate a compute device!\n");
        return EXIT_FAILURE;
    }

    print_device_info(device);

    // Set up execution context and command queue
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (!context) {
        return EXIT_FAILURE;
    }

    commands = clCreateCommandQueue(context, device, 0, &err);
    if (!commands) {
        return EXIT_FAILURE;
    }

    // Load kernel code from file
    char *source = load_program_source("kernels.cl");
    if (!source) {
        printf("Error: Failed to load kernel file\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Generate dummy input data
    int i = 0;
    for (i = 0; i < NUM_VALUES; i++) {
        in_data[i] = i;
    }

    // Set up input and output buffers on the device
    size_t buf_size = sizeof(cl_float) * NUM_VALUES;
    in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, buf_size, NULL, NULL);
    out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, NULL);
    if (!in_buf || !out_buf) {
        printf("Error: failed to allocate input output buffer on device\n");
        return EXIT_FAILURE;
    }

    // Transfer input data from host to device
    clEnqueueWriteBuffer(commands, in_buf, CL_TRUE, 0, buf_size, in_data, 0, NULL, NULL);

    // Set the arguments to our compute kernel
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    printf("Max local work group size is %zd\n", local);

    if (local > NUM_VALUES) {
        local = NUM_VALUES;
    }
    printf("Using work group size of %zd\n", local);

    clock_t begin, end;
    double time_spent;

    begin = clock();

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    global = NUM_VALUES;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);

    // Measure elapsed CPU time
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Spent %.5f seconds executing\n", time_spent);

    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(commands, out_buf, CL_TRUE, 0, sizeof(float) * NUM_VALUES,
                              out_data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // Validate our results
    unsigned int correct = 0;
    for(i = 0; i < NUM_VALUES; i++)
    {
        if(out_data[i] == in_data[i] * in_data[i])
            correct++;
    }

    // Print a brief summary detailing the results
    printf("Computed '%d/%d' correct values!\n", correct, NUM_VALUES);

    // Shutdown and cleanup
    clReleaseMemObject(in_buf);
    clReleaseMemObject(out_buf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

