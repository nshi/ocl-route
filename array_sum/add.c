#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* KERNEL_SOURCE_PATH = "/home/chi/src/opencl/add.cu";
const int SIZE = 12345678;

char* get_k_source()
{
    char *k_source;
    size_t f_size;
    int fd = open(KERNEL_SOURCE_PATH, O_RDONLY); 
    int i = 0;

    if (fd < 0) {
        printf("bad file handler\n");
        exit(-1);
    }

    f_size = lseek(fd, 0L, SEEK_END);
    k_source = malloc(f_size + 1);
    if (k_source <= 0) {
        printf("insufficient memory\n");
        exit(-1);
    }
    // remember to seek back to beginning
    lseek(fd, 0L, SEEK_SET);

    while(i < f_size) {
        i += read(fd, k_source + i, f_size - i);
    }

    close(fd);

    // not sure if this is really necessary... will read pick up EOF?
    k_source[f_size] = 0; 

    return k_source;
}

int multiple(int base, int res) 
{
    int div = res / base;

    if (res % base > 0)
        return (div + 1) * base;
    else
        return div * base;
}

int main() {
    const size_t mem_size = sizeof(float) * SIZE;

    // host memory
    float *src_h = malloc(SIZE * sizeof(float));
    float *res_h = malloc(SIZE * sizeof(float));

    // opencl environment crap
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel vector_add_k;
    cl_int err = CL_SUCCESS;

    cl_mem src_a_d, src_b_d, res_d;

    const char *source = get_k_source();

    const size_t local_ws = 256;    // Number of work-items per work-group
    const size_t global_ws = multiple(local_ws, SIZE);

    int i;
    for (i = 0; i < SIZE; i++) {
        src_h[i] = i;
    }

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("couldn't get platform id\n");
        return -1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting device ids %d\n", err);
        return -1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context) {
        printf("Error creating compute context\n");
        return -1;
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue) {
        printf("Error creating command queue\n");
        return -1;
    }

    // create the program
    program = clCreateProgramWithSource(context, 1, &source, 
                                        NULL, &err);
    if (!program) {
        printf("Error creating program\n");
        return -1;
    }

    // build the program (what's the diff between building and creating?)
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error building program\n");
        return -1;
    }

    // 'Extracting' the kernel
    vector_add_k = clCreateKernel(program, "vector_add_gpu", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating kernel\n");
        return -1;
    }

    // initalize memory on the device wih values from host
    src_a_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                             mem_size, src_h, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating memory 1\n");
        return -1;
    }

    src_b_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             mem_size, src_h, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating memory 2\n");
        return -1;
    }

    res_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating memory 3\n");
        return -1;
    }

    // set all 4 arguments to the kernel
    err = clSetKernelArg(vector_add_k, 0, sizeof(cl_mem), &src_a_d);
    err |= clSetKernelArg(vector_add_k, 1, sizeof(cl_mem), &src_b_d);
    err |= clSetKernelArg(vector_add_k, 2, sizeof(cl_mem), &res_d);
    err |= clSetKernelArg(vector_add_k, 3, sizeof(int), &SIZE);
    if (err != CL_SUCCESS) {
        printf("error setting kernel paramaters...\n");
        return -1;
    }

    // launch kernel
    err = clEnqueueNDRangeKernel(queue, vector_add_k, 1, NULL, &global_ws,
                                 &local_ws, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("something went wrong... %d \n", err);
        return -1;
    }

    // launching kernel is async, so block until command queue is done
    clFinish(queue);

    // Reading back
    clEnqueueReadBuffer(queue, res_d, CL_TRUE, 0, mem_size, res_h,
                        0, NULL, NULL);

    // check results
    for (i = 0; i < SIZE; i++) {
        if (res_h[i] != src_h[i] + src_h[i]) {
            printf("element %d, expected %d, got %d\n", i, res_h[i],
                   src_h[i] + src_h[i]);
            return -1;
        }
    }

    printf("everything worked out!\n");

    // there's a whole bunch of cleanup that you *should* do
    // but I don't care enough to do it so whateves
    return 0;
}
