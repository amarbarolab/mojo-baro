#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 1000003
#define CHECK(e, s) if (e != CL_SUCCESS) { printf("FAIL %s: cl error %d\n", s, e); return 1; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    unsigned char *il = malloc(len); fread(il, 1, len, f); fclose(f);
    cl_platform_id p; cl_device_id devs[4]; cl_uint nd; cl_int e;
    e = clGetPlatformIDs(1, &p, NULL); CHECK(e, "platform");
    e = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 4, devs, &nd); CHECK(e, "devices");
    cl_device_id d = devs[0]; char name[256];
    for (cl_uint i = 0; i < nd; i++) { clGetDeviceInfo(devs[i], CL_DEVICE_NAME, 256, name, NULL); if (strstr(name, "Radeon")) d = devs[i]; }
    clGetDeviceInfo(d, CL_DEVICE_NAME, 256, name, NULL); printf("device: %s\n", name);
    cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e); CHECK(e, "context");
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, d, NULL, &e); CHECK(e, "queue");
    cl_program prog = clCreateProgramWithIL(ctx, il, len, &e); CHECK(e, "il");
    e = clBuildProgram(prog, 1, &d, "", NULL, NULL);
    if (e != CL_SUCCESS) { char log[8192]; clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 8192, log, NULL); printf("FAIL build: %d\n%s\n", e, log); return 1; }
    cl_kernel k = clCreateKernel(prog, "swiglu", &e); CHECK(e, "kernel");
    float *g = malloc(4 * N), *u = malloc(4 * N), *o = malloc(4 * N);
    for (int i = 0; i < N; i++) { g[i] = (float)((i % 2001) - 1000) / 100.0f; u[i] = (float)((i * 7) % 997) / 311.0f - 1.5f; o[i] = -99; }
    int count = N;
    cl_mem bg = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * N, g, &e); CHECK(e, "bg");
    cl_mem bu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * N, u, &e); CHECK(e, "bu");
    cl_mem bo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 4 * N, NULL, &e); CHECK(e, "bo");
    cl_mem bc = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &count, &e); CHECK(e, "bc");
    clSetKernelArg(k, 0, sizeof(cl_mem), &bg); clSetKernelArg(k, 1, sizeof(cl_mem), &bu);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bo); e = clSetKernelArg(k, 3, sizeof(cl_mem), &bc); CHECK(e, "args");
    size_t local = 256, global = ((N + local - 1) / local) * local;
    e = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL); CHECK(e, "enqueue");
    e = clEnqueueReadBuffer(q, bo, CL_TRUE, 0, 4 * N, o, 0, NULL, NULL); CHECK(e, "read");
    double maxerr = 0;
    for (int i = 0; i < N; i++) { float ref = g[i] / (1 + exp(-g[i])) * u[i]; double err = fabs(o[i] - ref); if (err > maxerr) maxerr = err; }
    printf("n %d  out[5] %f  ref %f  max|err| %.3e\n", N, o[5], g[5] / (1 + exp(-g[5])) * u[5], maxerr);
    if (maxerr > 1e-4) { printf("FAIL parity\n"); return 1; }
    printf("PASS swiglu from Mojo IR on %s\n", name);
    return 0;
}
