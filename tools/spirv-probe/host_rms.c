#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ROWS 64
#define W 4096
#define CHECK(e, s) if (e != CL_SUCCESS) { printf("FAIL %s: cl error %d\n", s, e); return 1; }
static cl_mem mk(cl_context c, cl_mem_flags fl, size_t n, void *p) { cl_int e; cl_mem m = clCreateBuffer(c, fl, n, p, &e); if (e) { printf("FAIL buffer %d\n", e); exit(1); } return m; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    unsigned char *il = malloc(len); fread(il, 1, len, f); fclose(f);
    cl_platform_id p; cl_device_id devs[4], d; cl_uint nd; cl_int e; char name[256];
    clGetPlatformIDs(1, &p, NULL); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 4, devs, &nd); d = devs[0];
    for (cl_uint i = 0; i < nd; i++) { clGetDeviceInfo(devs[i], CL_DEVICE_NAME, 256, name, NULL); if (strstr(name, "Radeon")) d = devs[i]; }
    clGetDeviceInfo(d, CL_DEVICE_NAME, 256, name, NULL); printf("device: %s\n", name);
    cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e); CHECK(e, "context");
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, d, NULL, &e); CHECK(e, "queue");
    cl_program prog = clCreateProgramWithIL(ctx, il, len, &e); CHECK(e, "il");
    e = clBuildProgram(prog, 1, &d, "", NULL, NULL);
    if (e) { char log[16384]; clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 16384, log, NULL); printf("FAIL build %d\n%s\n", e, log); return 1; }
    float *x = malloc(4 * ROWS * W), *g = malloc(4 * W), *o = calloc(ROWS * W, 4);
    for (int i = 0; i < ROWS * W; i++) x[i] = sin(i * 0.37) * (1 + (i / W) % 5);
    for (int i = 0; i < W; i++) g[i] = 0.5f + (i % 13) / 10.0f;
    int n = W; float eps = 1e-6f;
    cl_mem bx = mk(ctx, CL_MEM_COPY_HOST_PTR, 4 * ROWS * W, x), bg = mk(ctx, CL_MEM_COPY_HOST_PTR, 4 * W, g);
    cl_mem bo = mk(ctx, CL_MEM_READ_WRITE, 4 * ROWS * W, NULL), bn = mk(ctx, CL_MEM_COPY_HOST_PTR, 4, &n), be = mk(ctx, CL_MEM_COPY_HOST_PTR, 4, &eps);
    cl_mem bp = mk(ctx, CL_MEM_READ_WRITE, 4 * ROWS * 256, NULL), bs = mk(ctx, CL_MEM_READ_WRITE, 4 * ROWS * 8, NULL);
    cl_mem args[7] = {bx, bg, bo, bn, be, bp, bs};
    const char *ph[3] = {"rms_a", "rms_b", "rms_c"};
    size_t global = ROWS * 256, local = 256;
    for (int k = 0; k < 3; k++) {
        cl_kernel kk = clCreateKernel(prog, ph[k], &e); CHECK(e, ph[k]);
        for (int a = 0; a < 7; a++) { e = clSetKernelArg(kk, a, sizeof(cl_mem), &args[a]); CHECK(e, "arg"); }
        e = clEnqueueNDRangeKernel(q, kk, 1, NULL, &global, &local, 0, NULL, NULL); CHECK(e, "enqueue");
    }
    e = clEnqueueReadBuffer(q, bo, CL_TRUE, 0, 4 * ROWS * W, o, 0, NULL, NULL); CHECK(e, "read");
    double maxrel = 0;
    for (int r = 0; r < ROWS; r++) {
        double ss = 0; for (int i = 0; i < W; i++) ss += (double)x[r * W + i] * x[r * W + i];
        double sc = 1.0 / sqrt(ss / W + eps);
        for (int i = 0; i < W; i++) { double ref = x[r * W + i] * sc * g[i]; double rel = fabs(o[r * W + i] - ref) / (fabs(ref) + 1e-3); if (rel > maxrel) maxrel = rel; }
    }
    printf("rows %d width %d  out[1] %f  max rel err %.3e\n", ROWS, W, o[1], maxrel);
    if (maxrel > 1e-4) { printf("FAIL parity\n"); return 1; }
    printf("PASS rmsnorm (3-phase split of Mojo IR) on %s\n", name);
    return 0;
}
