/* One harness for every amar_* elementwise kernel: load <dir>/<name>.spv, run it on the
   Radeon through rusticl, check against a CPU reference of the kernel's formula.
   usage: host <spv-dir> [kernel...]     last line: PASS N/N  or  FAIL k/N <names>
   Shapes mirror tools/spirv-probe/probe_all.mojo (layout strides are baked into the IR).
   Bars: float outputs max rel err <= 1e-5 vs an fp64 reference; integer outputs, bf16 -> f32
   and q8 codes/scales exact; f32 -> bf16 outputs must be the correct rounding of a value
   within 1e-5 of the reference (bf16 has 8 mantissa bits, so a last-ulp f32 difference can
   flip the rounding; the window says exactly that and nothing looser). */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R 8
#define H 4096
#define V 5003
#define TV 1024
#define K 64
#define QM 8
#define QK 4096
#define EW 256
#define REPS 3
#define TOL 1e-5

typedef struct { cl_context ctx; cl_command_queue q; cl_device_id dev; const char *dir; char why[256]; } Env;

static uint32_t rng = 12345;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 8) / 8388608.0f - 1.0f; }
static uint16_t to_bf16(float f) { __bf16 b = (__bf16)f; uint16_t u; memcpy(&u, &b, 2); return u; }
static float from_bf16(uint16_t u) { __bf16 b; memcpy(&b, &u, 2); return (float)b; }
static uint16_t to_f16(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }

static int fail(Env *e, const char *fmt, double a, double b) { snprintf(e->why, sizeof e->why, fmt, a, b); return 1; }

static cl_kernel load(Env *e, const char *name) {
    char path[512]; snprintf(path, sizeof path, "%s/%s.spv", e->dir, name);
    FILE *f = fopen(path, "rb"); if (!f) { fail(e, "missing spv (conversion failed upstream)", 0, 0); return NULL; }
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    unsigned char *il = malloc(len); if (fread(il, 1, len, f) != (size_t)len) { fclose(f); fail(e, "short read", 0, 0); return NULL; } fclose(f);
    cl_int err; cl_program p = clCreateProgramWithIL(e->ctx, il, len, &err); free(il);
    if (err) { fail(e, "clCreateProgramWithIL %g", err, 0); return NULL; }
    err = clBuildProgram(p, 1, &e->dev, "", NULL, NULL);
    if (err) { char log[4096] = ""; clGetProgramBuildInfo(p, e->dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); printf("build log %s: %s\n", name, log); fail(e, "clBuildProgram %g", err, 0); return NULL; }
    cl_kernel k = clCreateKernel(p, name, &err); if (err) { fail(e, "clCreateKernel %g", err, 0); return NULL; }
    return k;
}
static cl_mem buf(Env *e, size_t n, const void *init) {
    cl_int err; cl_mem m = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), n, (void *)init, &err);
    if (err) { printf("FAIL buffer: cl error %d\n", err); exit(1); } return m;
}
static cl_mem ci(Env *e, int v) { return buf(e, 4, &v); }
static cl_mem cf(Env *e, float v) { return buf(e, 4, &v); }
static int launch(Env *e, const char *name, int n, cl_mem *a, size_t groups_x, size_t groups_y, size_t local) {
    cl_kernel k = load(e, name); if (!k) return 1;
    for (int i = 0; i < n; i++) { cl_int err = clSetKernelArg(k, i, sizeof(cl_mem), &a[i]); if (err) return fail(e, "clSetKernelArg %g = %g", i, err); }
    size_t g[2] = {groups_x * local, groups_y}, l[2] = {local, 1};
    cl_int err = clEnqueueNDRangeKernel(e->q, k, 2, NULL, g, l, 0, NULL, NULL); if (err) return fail(e, "enqueue %g", err, 0);
    err = clFinish(e->q); if (err) return fail(e, "clFinish %g", err, 0);
    return 0;
}
static int rd(Env *e, cl_mem m, size_t n, void *p) { cl_int err = clEnqueueReadBuffer(e->q, m, CL_TRUE, 0, n, p, 0, NULL, NULL); return err ? fail(e, "read %g", err, 0) : 0; }
static double rel(double got, double ref) { return fabs(got - ref) / (fabs(ref) + 1e-30); }
static int bf16_window(uint16_t got, double ref) {
    float g = from_bf16(got), a = from_bf16(to_bf16((float)(ref * (1 - TOL)))), b = from_bf16(to_bf16((float)(ref * (1 + TOL))));
    return g >= fminf(a, b) && g <= fmaxf(a, b);
}

#define SENT -77.0f
static float *X, *G; static double *REF;
static void rms_inputs(int n) {
    for (int i = 0; i < R * H; i++) X[i] = rnd() * 3.0f * (1 + (i / H) % 5);
    for (int i = 0; i < H; i++) G[i] = rnd() + (rnd() > 0 ? 1.0f : -1.0f);
    for (int r = 0; r < R; r++) {
        double ss = 0; for (int i = 0; i < n; i++) ss += (double)X[r * H + i] * X[r * H + i];
        double sc = 1.0 / sqrt(ss / n + (double)1e-6f);
        for (int i = 0; i < n; i++) REF[r * H + i] = X[r * H + i] * sc * G[i];
    }
}
static int t_rmsnorm(Env *e, double *w) {
    int n = 4001; rms_inputs(n);
    float *o = malloc(4 * R * H); for (int i = 0; i < R * H; i++) o[i] = SENT;
    cl_mem bo = buf(e, 4 * R * H, o), a[5] = {buf(e, 4 * R * H, X), buf(e, 4 * H, G), bo, ci(e, n), cf(e, 1e-6f)};
    if (launch(e, "amar_rmsnorm", 5, a, R, 1, EW) || rd(e, bo, 4 * R * H, o)) return 1;
    for (int r = 0; r < R; r++) for (int i = 0; i < H; i++) {
        if (i >= n) { if (o[r * H + i] != SENT) return fail(e, "wrote past n at row %g col %g", r, i); continue; }
        double x = rel(o[r * H + i], REF[r * H + i]); if (x > *w) *w = x;
    }
    return *w > TOL ? fail(e, "max rel err %.3e > %.0e", *w, TOL) : 0;
}
static int rms_cast(Env *e, double *w, int two) {
    int n = 4001; rms_inputs(n);
    uint16_t *o = malloc(2 * R * H); for (int i = 0; i < R * H; i++) o[i] = 0xABCD;
    float *f = malloc(4 * H); for (int i = 0; i < H; i++) f[i] = SENT;
    cl_mem bo = buf(e, 2 * R * H, o), bf = buf(e, 4 * H, f), a[6] = {buf(e, 4 * R * H, X), buf(e, 4 * H, G), bo};
    int na = 3; if (two) a[na++] = bf; a[na++] = ci(e, n); a[na++] = cf(e, 1e-6f);
    if (launch(e, two ? "amar_rmsnorm_cast2" : "amar_rmsnorm_cast", na, a, R, 1, EW) || rd(e, bo, 2 * R * H, o) || rd(e, bf, 4 * H, f)) return 1;
    long exact = 0;
    for (int r = 0; r < R; r++) for (int i = 0; i < H; i++) {
        if (i >= n) { if (o[r * H + i] != 0xABCD) return fail(e, "wrote past n at row %g col %g", r, i); continue; }
        if (!bf16_window(o[r * H + i], REF[r * H + i])) return fail(e, "bf16 outside 1e-5 rounding window at %g (got %g)", r * H + i, from_bf16(o[r * H + i]));
        exact += o[r * H + i] == to_bf16((float)REF[r * H + i]);
    }
    if (two) for (int i = 0; i < H; i++) if (f[i] != (i < n ? from_bf16(o[i]) : SENT)) return fail(e, "F[%g] = %g is not the f32 of O[0,i]", i, f[i]);
    *w = 1.0 - (double)exact / (R * n);
    return 0;
}
static int t_rmsnorm_cast(Env *e, double *w) { return rms_cast(e, w, 0); }
static int t_rmsnorm_cast2(Env *e, double *w) { return rms_cast(e, w, 1); }

static int t_swiglu(Env *e, double *w) {
    int n = 1000003; float *g = malloc(4 * n), *u = malloc(4 * n), *o = malloc(4 * n);
    for (int i = 0; i < n; i++) { g[i] = rnd() * 10; u[i] = rnd() * 3; }
    cl_mem bo = buf(e, 4 * n, NULL), a[4] = {buf(e, 4 * n, g), buf(e, 4 * n, u), bo, ci(e, n)};
    if (launch(e, "amar_swiglu", 4, a, (n + EW - 1) / EW, 1, EW) || rd(e, bo, 4 * n, o)) return 1;
    for (int i = 0; i < n; i++) { double x = rel(o[i], g[i] / (1 + exp(-(double)g[i])) * u[i]); if (x > *w) *w = x; }
    return *w > TOL ? fail(e, "max rel err %.3e > %.0e", *w, TOL) : 0;
}
static int t_rope(Env *e, double *w) {
    int heads = 8, hd = 128, pos = 5, half = hd / 2; float theta = 10000.0f;
    float *x = malloc(4 * R * H), *o = malloc(4 * R * H); for (int i = 0; i < R * H; i++) x[i] = rnd() * 4;
    cl_mem bx = buf(e, 4 * R * H, x), a[5] = {bx, ci(e, heads), ci(e, hd), ci(e, pos), cf(e, theta)};
    if (launch(e, "amar_rope_rows", 5, a, R, 1, EW) || rd(e, bx, 4 * R * H, o)) return 1;
    for (int r = 0; r < R; r++) {
        for (int i = heads * hd; i < H; i++) if (o[r * H + i] != x[r * H + i]) return fail(e, "touched column %g of row %g", i, r);
        for (int h = 0; h < heads; h++) for (int i = 0; i < half; i++) {
            double ang = (pos + r) * exp(-2.0 * i / hd * log((double)theta)), c = cos(ang), s = sin(ang);
            double x0 = x[r * H + h * hd + i], x1 = x[r * H + h * hd + half + i], w0 = x0 * c - x1 * s, w1 = x0 * s + x1 * c;
            double err = fmax(fabs(o[r * H + h * hd + i] - w0), fabs(o[r * H + h * hd + half + i] - w1)) / (fmax(fabs(w0), fabs(w1)) + 1e-30);
            if (err > *w) *w = err;
        }
    }
    return *w > TOL ? fail(e, "max pair-norm rel err %.3e > %.0e", *w, TOL) : 0;
}
static float *logits(void) {
    float *s = malloc(4 * R * V); for (int i = 0; i < R * V; i++) s[i] = rnd() * 9;
    for (int r = 0; r < R; r++) { int a = (r * 911 + 300) % (V - 600); s[r * V + a] = s[r * V + a + 513] = 9.5f + r; }
    return s;
}
static int t_softmax(Env *e, double *w) {
    float *s = logits(), *o = malloc(4 * R * V);
    cl_mem bs = buf(e, 4 * R * V, s), a[2] = {bs, ci(e, V)};
    if (launch(e, "amar_softmax_rows", 2, a, R, 1, EW) || rd(e, bs, 4 * R * V, o)) return 1;
    for (int r = 0; r < R; r++) {
        double mx = -1e300, tot = 0; for (int i = 0; i < V; i++) mx = fmax(mx, s[r * V + i]);
        for (int i = 0; i < V; i++) tot += exp(s[r * V + i] - mx);
        for (int i = 0; i < V; i++) { double x = rel(o[r * V + i], exp(s[r * V + i] - mx) / tot); if (x > *w) *w = x; }
    }
    return *w > TOL ? fail(e, "max rel err %.3e > %.0e", *w, TOL) : 0;
}
static int argmax(Env *e, int pos) {
    float *s = logits(); int32_t out[K]; for (int i = 0; i < K; i++) out[i] = -1;
    cl_mem bo = buf(e, 4 * K, out), a[4] = {buf(e, 4 * R * V, s), bo, ci(e, V), ci(e, 3)};
    if (launch(e, pos ? "amar_argmax_pos" : "amar_argmax_row", pos ? 4 : 3, a, R, 1, EW) || rd(e, bo, 4 * K, out)) return 1;
    for (int j = 0; j < K; j++) {
        int r = j - (pos ? 3 : 0), want = -1;
        if (r >= 0 && r < R) { want = 0; for (int i = 1; i < V; i++) if (s[r * V + i] > s[r * V + want]) want = i; }
        if (out[j] != want) return fail(e, "Out[%g] = %g (tied maxima planted 513 apart, lowest index must win)", j, out[j]);
    }
    return 0;
}
static int t_argmax_pos(Env *e, double *w) { (void)w; return argmax(e, 1); }
static int t_argmax_row(Env *e, double *w) { (void)w; return argmax(e, 0); }

static int embed(Env *e, int pos) {
    int n = 4000; uint16_t *tab = malloc(2 * TV * H); float *o = malloc(4 * R * H); int32_t toks[K];
    for (int i = 0; i < TV * H; i++) { rng = rng * 1664525u + 1013904223u; tab[i] = (uint16_t)(rng >> 12); if ((tab[i] & 0x7F80) == 0x7F80) tab[i] &= 0xBFFF; }
    for (int i = 0; i < R * H; i++) o[i] = SENT;
    for (int i = 0; i < K; i++) toks[i] = (int)((rnd() + 1) * 0.5f * (TV - 1));
    cl_mem bo = buf(e, 4 * R * H, o), a[5] = {buf(e, 2 * TV * H, tab), bo}; int na = 2;
    if (pos) { a[na++] = buf(e, 4 * K, toks); a[na++] = ci(e, 3); } else a[na++] = ci(e, 123);
    a[na++] = ci(e, n);
    if (launch(e, pos ? "amar_embed_lookup_pos" : "amar_embed_lookup", na, a, (H + EW - 1) / EW, R, EW) || rd(e, bo, 4 * R * H, o)) return 1;
    for (int r = 0; r < R; r++) for (int i = 0; i < H; i++) {
        float want = i < n ? from_bf16(tab[(pos ? toks[3 + r] : 123) * H + i]) : SENT;
        if (memcmp(&o[r * H + i], &want, 4)) return fail(e, "row %g col %g differs bitwise", r, i);
    }
    return 0;
}
static int t_embed_lookup(Env *e, double *w) { (void)w; return embed(e, 0); }
static int t_embed_lookup_pos(Env *e, double *w) { (void)w; return embed(e, 1); }

static int t_tok_copy(Env *e, double *w) {
    (void)w; int32_t src[K], dst[K], out[K]; for (int i = 0; i < K; i++) { src[i] = 1000 + i * 7; dst[i] = -5 - i; }
    cl_mem bd = buf(e, 4 * K, dst), a[5] = {buf(e, 4 * K, src), bd, ci(e, 1), ci(e, 2), ci(e, 8)};
    if (launch(e, "amar_tok_copy", 5, a, 1, 1, EW) || rd(e, bd, 4 * K, out)) return 1;
    for (int i = 0; i < K; i++) if (out[i] != (i >= 2 && i < 10 ? src[i - 1] : dst[i])) return fail(e, "Dst[%g] = %g", i, out[i]);
    return 0;
}
static int t_tok_remap(Env *e, double *w) {
    (void)w; int n = 50; int32_t map[K], tok[K], out[K]; for (int i = 0; i < K; i++) { map[i] = 50000 - i * 13; tok[i] = (i * 37 + 5) % K; }
    cl_mem bt = buf(e, 4 * K, tok), a[3] = {buf(e, 4 * K, map), bt, ci(e, n)};
    if (launch(e, "amar_tok_remap", 3, a, 1, 1, EW) || rd(e, bt, 4 * K, out)) return 1;
    for (int i = 0; i < K; i++) if (out[i] != (i < n ? map[tok[i]] : tok[i])) return fail(e, "Dtok[%g] = %g", i, out[i]);
    return 0;
}
static int t_quantize_q8_rows(Env *e, double *w) {
    (void)w; long bad = 0; uint16_t *a16 = malloc(2 * QM * QK), *sc = malloc(2 * QM * (QK / 32)); int8_t *q = malloc(QM * QK);
    for (int i = 0; i < QM * QK; i++) a16[i] = to_bf16(rnd() * (1 + (i / 32) % 7) * ((i / 32) % 11 == 3 ? 1e-3f : 1.0f));
    for (int i = 0; i < 32; i++) a16[5 * 32 + i] = (i & 1) ? 0x8000 : 0;
    cl_mem bq = buf(e, QM * QK, NULL), bs = buf(e, 2 * QM * (QK / 32), NULL), a[5] = {buf(e, 2 * QM * QK, a16), bq, bs, ci(e, QM), ci(e, QK)};
    if (launch(e, "amar_quantize_q8_rows", 5, a, QM, QK / 32, 32) || rd(e, bq, QM * QK, q) || rd(e, bs, 2 * QM * (QK / 32), sc)) return 1;
    for (int b = 0; b < QM * (QK / 32); b++) {
        float wmax = 0; for (int i = 0; i < 32; i++) wmax = fmaxf(wmax, fabsf(from_bf16(a16[b * 32 + i])));
        float scale = wmax > 0 ? wmax / 127.0f : 1.0f;
        if (sc[b] != to_f16(scale)) return fail(e, "scale of block %g = f16 bits %g", b, sc[b]);
        for (int i = 0; i < 32; i++) {
            float v = from_bf16(a16[b * 32 + i]) / scale; int want = v >= 0 ? (int)(v + 0.5f) : -(int)(-v + 0.5f);
            want = want > 127 ? 127 : want < -127 ? -127 : want;
            if (q[b * 32 + i] != want && !bad++) snprintf(e->why, sizeof e->why, "code %d = %d want %d (x %.9g scale %.9g x/scale %.9g)", b * 32 + i, q[b * 32 + i], want, from_bf16(a16[b * 32 + i]), scale, v);
        }
    }
    if (bad) { size_t l = strlen(e->why); snprintf(e->why + l, sizeof e->why - l, "; %ld of %d codes differ", bad, QM * QK); return 1; }
    return 0;
}

/* G1 scouting, not part of the default 13: the engine's m = 1 q4 GEMV (ggml Q4_0 layout) and its reduce. */
#define GN 1024
#define GK 4096
static int gemv(Env *e, double *w, int reduce) {
    uint16_t *a = malloc(2 * GK), *sc = malloc(2 * GN * (GK / 32)); uint8_t *q = malloc(GN * (GK / 2)); float *p = malloc(4 * GN), *c = malloc(4 * GN);
    for (int i = 0; i < GK; i++) a[i] = to_bf16(rnd() * 2);
    for (int i = 0; i < GN * (GK / 32); i++) sc[i] = to_f16((rnd() + 1.001f) * 0.05f * (i % 9 == 4 ? 1e-4f : 1.0f));
    for (int i = 0; i < GN * (GK / 2); i++) { rng = rng * 1664525u + 1013904223u; q[i] = (uint8_t)(rng >> 13); }
    for (int i = 0; i < GN; i++) p[i] = c[i] = SENT;
    cl_mem bp = buf(e, 4 * GN, p), bc = buf(e, 4 * GN, c), g[7] = {buf(e, 2 * GK, a), buf(e, GN * (GK / 2), q), buf(e, 2 * GN * (GK / 32), sc), bp, ci(e, 1), ci(e, GN), ci(e, GK)};
    if (launch(e, "amar_matmul_skinny_q4rowb", 7, g, GN / 8, 1, EW) || rd(e, bp, 4 * GN, p)) return 1;
    if (reduce) {
        cl_mem r[4] = {bp, bc, ci(e, 1), ci(e, GN)};
        if (launch(e, "amar_skinny_reduce", 4, r, (GN + EW - 1) / EW, 1, EW) || rd(e, bc, 4 * GN, c)) return 1;
        for (int i = 0; i < GN; i++) if (c[i] != p[i]) return fail(e, "C[%g] = %g is not Cp", i, c[i]);
        return 0;
    }
    for (int row = 0; row < GN; row++) {
        double dot = 0, mag = 0;
        for (int b = 0; b < GK / 32; b++) {
            _Float16 h; memcpy(&h, &sc[row * (GK / 32) + b], 2); double d = (double)h;
            for (int j = 0; j < 16; j++) {
                uint8_t by = q[row * (GK / 2) + b * 16 + j];
                double t0 = ((by & 15) - 8) * d * from_bf16(a[b * 32 + j]), t1 = ((by >> 4) - 8) * d * from_bf16(a[b * 32 + 16 + j]);
                dot += t0 + t1; mag += fabs(t0) + fabs(t1);
            }
        }
        double x = fabs(p[row] - dot) / (mag + 1e-30); if (x > *w) *w = x;
    }
    return *w > TOL ? fail(e, "max err relative to sum|terms| %.3e > %.0e", *w, TOL) : 0;
}
static int t_q4rowb(Env *e, double *w) { return gemv(e, w, 0); }
static int t_reduce(Env *e, double *w) { return gemv(e, w, 1); }

#define DEFAULT_CASES 13
static const struct { const char *name; int (*run)(Env *, double *); const char *metric; } CASES[] = {
    {"amar_rmsnorm", t_rmsnorm, "max rel err"}, {"amar_rmsnorm_cast", t_rmsnorm_cast, "in 1e-5 window, fraction not bit-equal to fp64-ref rounding"},
    {"amar_rmsnorm_cast2", t_rmsnorm_cast2, "in 1e-5 window, F exact, fraction not bit-equal"}, {"amar_swiglu", t_swiglu, "max rel err"},
    {"amar_rope_rows", t_rope, "max pair-norm rel err"}, {"amar_softmax_rows", t_softmax, "max rel err"},
    {"amar_embed_lookup", t_embed_lookup, "bit-exact, err"}, {"amar_embed_lookup_pos", t_embed_lookup_pos, "bit-exact, err"},
    {"amar_argmax_pos", t_argmax_pos, "exact, err"}, {"amar_argmax_row", t_argmax_row, "exact, err"},
    {"amar_tok_copy", t_tok_copy, "exact, err"}, {"amar_tok_remap", t_tok_remap, "exact, err"},
    {"amar_quantize_q8_rows", t_quantize_q8_rows, "codes and f16 scales exact, err"},
    {"amar_matmul_skinny_q4rowb", t_q4rowb, "max err relative to sum|terms|"}, {"amar_skinny_reduce", t_reduce, "exact copy of the GEMV partials, err"},
};

int main(int argc, char **argv) {
    if (argc < 2) { printf("FAIL usage: host <spv-dir> [kernel...]\n"); return 2; }
    Env e = {.dir = argv[1]}; cl_platform_id p; cl_device_id devs[8]; cl_uint nd = 0; cl_int err; char name[256] = "";
    if (clGetPlatformIDs(1, &p, NULL) || clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 8, devs, &nd) || !nd) { printf("FAIL device: no OpenCL GPU (RUSTICL_ENABLE=radeonsi set?)\n"); return 1; }
    e.dev = NULL;
    for (cl_uint i = 0; i < nd; i++) { clGetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof name, name, NULL); if (strstr(name, "Radeon")) { e.dev = devs[i]; break; } }
    if (!e.dev) { printf("FAIL device: no Radeon among %u GPU devices (last: %s)\n", nd, name); return 1; }
    printf("device: %s\n", name);
    e.ctx = clCreateContext(NULL, 1, &e.dev, NULL, NULL, &err); if (err) { printf("FAIL context: %d\n", err); return 1; }
    e.q = clCreateCommandQueueWithProperties(e.ctx, e.dev, NULL, &err); if (err) { printf("FAIL queue: %d\n", err); return 1; }
    X = malloc(4 * R * H); G = malloc(4 * H); REF = malloc(8 * R * H);
    int total = 0, bad = 0; char names[1024] = "";
    for (size_t c = 0; c < sizeof CASES / sizeof CASES[0]; c++) {
        int want = argc == 2 && c < DEFAULT_CASES; for (int i = 2; i < argc; i++) want |= !strcmp(argv[i], CASES[c].name);
        if (!want) continue;
        total++; int failed = 0; double worst = 0;
        for (int rep = 0; rep < REPS && !failed; rep++) { double w = 0; rng = 12345 + 977 * rep; failed = CASES[c].run(&e, &w); if (w > worst) worst = w; }
        if (failed) { bad++; strcat(names, " "); strcat(names, CASES[c].name); printf("FAIL %s: %s\n", CASES[c].name, e.why); }
        else printf("PASS %s x%d  %s %.3e\n", CASES[c].name, REPS, CASES[c].metric, worst);
    }
    if (!total) { printf("FAIL no kernel matched\n"); return 1; }
    if (bad) { printf("FAIL %d/%d%s\n", bad, total, names); return 1; }
    printf("PASS %d/%d\n", total, total);
    return 0;
}
