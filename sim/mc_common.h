/* Shared pieces of the Monte Carlo risk engine (serial / OpenMP / MPI).
 *
 * Model: correlated geometric Brownian motion
 *   S_i(t+dt) = S_i(t) exp((m_i - sigma_i^2/2) dt + sqrt(dt) (L z)_i),  z ~ N(0, I)
 * where L L^T = Cov (annualised) and m_i = mu_i (real-world, VaR) or r (risk-neutral, options).
 * Cost per path: steps * (n normals + n(n+1)/2 multiply-adds for L z).
 *
 * Reproducibility: path p always uses the RNG stream seeded by (seed, p), so the
 * set of simulated paths is identical regardless of how paths are split across
 * threads or ranks. Parallel results can therefore be checked against serial ones.
 */
#ifndef MC_COMMON_H
#define MC_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MC_MAX_ASSETS 2048 /* bounds the per-path stack buffers only */

typedef enum { MODE_VAR, MODE_OPTION } mc_mode;

typedef struct {
    int n;               /* number of assets */
    char (*ticker)[16];
    double *S0, *mu, *sigma;
    double *cov;         /* n x n, row-major */
    double *L;           /* lower Cholesky factor of cov, n x n row-major */
} mc_market;

typedef struct {
    mc_mode mode;
    long paths;
    int steps;
    double T;          /* horizon in years */
    double r;          /* risk-free rate (option mode) */
    double K_rel;      /* strike as a fraction of S0 (option mode) */
    double value;      /* portfolio value in KRW, equally weighted (VaR mode) */
    uint64_t seed;
    const char *params;
} mc_config;

/* ---------- RNG: splitmix64 seeding + xoshiro256** ---------- */

typedef struct { uint64_t s[4]; double spare; int has_spare; } mc_rng;

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void mc_rng_init(mc_rng *g, uint64_t seed, uint64_t stream) {
    uint64_t x = seed ^ (stream * 0xD1342543DE82EF95ULL + 0x632BE59BD9B4E019ULL);
    for (int i = 0; i < 4; i++) g->s[i] = splitmix64(&x);
    g->has_spare = 0;
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t mc_next(mc_rng *g) {
    uint64_t *s = g->s;
    uint64_t res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}

/* uniform in (0, 1) */
static inline double mc_uniform(mc_rng *g) {
    return ((mc_next(g) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/* standard normal, Box-Muller (caches the second value) */
static inline double mc_normal(mc_rng *g) {
    if (g->has_spare) { g->has_spare = 0; return g->spare; }
    double u1 = mc_uniform(g), u2 = mc_uniform(g);
    double rad = sqrt(-2.0 * log(u1)), th = 6.283185307179586 * u2;
    g->spare = rad * sin(th); g->has_spare = 1;
    return rad * cos(th);
}

/* ---------- market parameters ---------- */

static int mc_market_alloc(mc_market *m, int n) {
    m->n = n;
    m->ticker = calloc(n, sizeof *m->ticker);
    m->S0 = calloc(n, sizeof(double));
    m->mu = calloc(n, sizeof(double));
    m->sigma = calloc(n, sizeof(double));
    m->cov = calloc((size_t)n * n, sizeof(double));
    m->L = calloc((size_t)n * n, sizeof(double));
    return (m->ticker && m->S0 && m->mu && m->sigma && m->cov && m->L) ? 0 : -1;
}

static void mc_market_free(mc_market *m) {
    free(m->ticker); free(m->S0); free(m->mu); free(m->sigma); free(m->cov); free(m->L);
}

static int mc_cholesky(mc_market *m) {
    int n = m->n;
    double *L = m->L;
    memset(L, 0, sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            double s = m->cov[i * n + j];
            for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (s <= 0) return -1;
                L[i * n + i] = sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    for (int i = 0; i < n; i++) m->sigma[i] = sqrt(m->cov[i * n + i]);
    return 0;
}

static int mc_load_market(const char *path, mc_market *m) {
    int n;
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    if (fscanf(f, "%d", &n) != 1 || n < 1 || n > MC_MAX_ASSETS) goto bad;
    if (mc_market_alloc(m, n)) { fprintf(stderr, "out of memory\n"); fclose(f); return -1; }
    for (int i = 0; i < n; i++)
        if (fscanf(f, "%15s %lf %lf", m->ticker[i], &m->S0[i], &m->mu[i]) != 3) goto bad;
    for (long i = 0; i < (long)n * n; i++)
        if (fscanf(f, "%lf", &m->cov[i]) != 1) goto bad;
    fclose(f);
    if (mc_cholesky(m)) { fprintf(stderr, "covariance is not positive definite\n"); return -1; }
    return 0;
bad:
    fprintf(stderr, "%s: malformed params file\n", path);
    fclose(f);
    return -1;
}

/* ---------- one path ---------- */

/* Simulate path p and return the terminal prices in ST[]. */
static inline void mc_path(const mc_market *m, const mc_config *c, long p, double *ST) {
    int n = m->n;
    const double *L = m->L;
    double dt = c->T / c->steps, sdt = sqrt(dt);
    double drift[MC_MAX_ASSETS], logS[MC_MAX_ASSETS], z[MC_MAX_ASSETS];
    for (int i = 0; i < n; i++) {
        double mi = (c->mode == MODE_OPTION) ? c->r : m->mu[i];
        drift[i] = (mi - 0.5 * m->cov[i * n + i]) * dt;
        logS[i] = 0.0;
    }
    mc_rng g;
    mc_rng_init(&g, c->seed, (uint64_t)p);
    for (int s = 0; s < c->steps; s++) {
        for (int i = 0; i < n; i++) z[i] = mc_normal(&g);
        for (int i = 0; i < n; i++) {
            const double *Li = L + (size_t)i * n;
            double e = 0.0;
            for (int k = 0; k <= i; k++) e += Li[k] * z[k];
            logS[i] += drift[i] + sdt * e;
        }
    }
    for (int i = 0; i < n; i++) ST[i] = m->S0[i] * exp(logS[i]);
}

/* Accumulators: sums and sums of squares of n+1 quantities.
 * VaR mode   : slot 0 = loss V0 - V_T of the equally weighted portfolio.
 * Option mode: slot i = e^{-rT} max(S_T,i - K_i, 0); slot n = basket call.
 * Stored contiguously (sum then sumsq) so MPI can reduce it in one call. */
typedef struct { int len; double *sum, *sumsq; } mc_acc;

static int mc_acc_init(mc_acc *a, int n) {
    a->len = n + 1;
    a->sum = calloc(2 * (size_t)a->len, sizeof(double));
    a->sumsq = a->sum + a->len;
    return a->sum ? 0 : -1;
}

static void mc_acc_free(mc_acc *a) { free(a->sum); }

static inline void mc_acc_merge(mc_acc *dst, const mc_acc *src) {
    for (int i = 0; i < 2 * dst->len; i++) dst->sum[i] += src->sum[i];
}

/* Process path p; returns the loss (VaR mode) and accumulates into a. */
static inline double mc_eval_path(const mc_market *m, const mc_config *c, long p, mc_acc *a) {
    double ST[MC_MAX_ASSETS];
    int n = m->n;
    mc_path(m, c, p, ST);
    if (c->mode == MODE_VAR) {
        double w = c->value / n, vT = 0.0;
        for (int i = 0; i < n; i++) vT += w * ST[i] / m->S0[i];
        double loss = c->value - vT;
        a->sum[0] += loss; a->sumsq[0] += loss * loss;
        return loss;
    }
    double disc = exp(-c->r * c->T), basket = 0.0;
    for (int i = 0; i < n; i++) {
        double x = disc * fmax(ST[i] - c->K_rel * m->S0[i], 0.0);
        a->sum[i] += x; a->sumsq[i] += x * x;
        basket += ST[i] / m->S0[i] / n;          /* normalised basket, starts at 1 */
    }
    double xb = disc * fmax(basket - c->K_rel, 0.0);
    a->sum[n] += xb; a->sumsq[n] += xb * xb;
    return 0.0;
}

/* ---------- statistics ---------- */

/* k-th smallest element (0-based); reorders a[]. */
static double mc_select(double *a, long len, long k) {
    long lo = 0, hi = len - 1;
    while (lo < hi) {
        double piv = a[lo + (hi - lo) / 2];
        long i = lo, j = hi;
        while (i <= j) {
            while (a[i] < piv) i++;
            while (a[j] > piv) j--;
            if (i <= j) { double t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
    }
    return a[k];
}

/* VaR_alpha = alpha-quantile of the loss; CVaR = mean loss at or beyond VaR. */
static void mc_var_cvar(double *loss, long len, double alpha, double *var, double *cvar) {
    long k = (long)ceil(alpha * len) - 1;
    if (k < 0) k = 0;
    *var = mc_select(loss, len, k);
    double s = 0.0; long cnt = 0;
    for (long i = 0; i < len; i++) if (loss[i] >= *var) { s += loss[i]; cnt++; }
    *cvar = s / cnt;
}

static double bs_call(double S, double K, double r, double sig, double T) {
    double d1 = (log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt(T)), d2 = d1 - sig * sqrt(T);
    double N1 = 0.5 * erfc(-d1 / sqrt(2.0)), N2 = 0.5 * erfc(-d2 / sqrt(2.0));
    return S * N1 - K * exp(-r * T) * N2;
}

static inline double mc_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---------- CLI ---------- */

static void mc_usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--mode var|option] [--paths N] [--steps S] [--T years] [--r rate]\n"
        "          [--K rel_strike] [--value KRW] [--seed N] [--params file]\n", prog);
}

static int mc_parse(int argc, char **argv, mc_config *c) {
    *c = (mc_config){ MODE_VAR, 100000, 10, 10.0 / 252.0, 0.03, 1.0, 1e8, 410, "data/stats/params.txt" };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { mc_usage(argv[0]); return -1; }
        if (!strcmp(a, "--mode")) {
            if (!strcmp(v, "var")) c->mode = MODE_VAR;
            else if (!strcmp(v, "option")) c->mode = MODE_OPTION;
            else { mc_usage(argv[0]); return -1; }
        }
        else if (!strcmp(a, "--paths")) c->paths = atol(v);
        else if (!strcmp(a, "--steps")) c->steps = atoi(v);
        else if (!strcmp(a, "--T")) c->T = atof(v);
        else if (!strcmp(a, "--r")) c->r = atof(v);
        else if (!strcmp(a, "--K")) c->K_rel = atof(v);
        else if (!strcmp(a, "--value")) c->value = atof(v);
        else if (!strcmp(a, "--seed")) c->seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--params")) c->params = v;
        else { mc_usage(argv[0]); return -1; }
        i++;
    }
    if (c->paths < 2 || c->steps < 1 || c->T <= 0) { mc_usage(argv[0]); return -1; }
    return 0;
}

/* Print results as one JSON object on stdout. extra = implementation-specific fields
 * (already formatted as ,"k":v pairs) or NULL. */
static void mc_report(const char *impl, int workers, const mc_market *m, const mc_config *c,
                      const mc_acc *a, double *loss, double elapsed, const char *extra) {
    long N = c->paths;
    printf("{\"impl\":\"%s\",\"workers\":%d,\"mode\":\"%s\",\"paths\":%ld,\"steps\":%d,"
           "\"assets\":%d,\"T\":%.10g,\"seed\":%llu,\"elapsed\":%.6f%s",
           impl, workers, c->mode == MODE_VAR ? "var" : "option", N, c->steps, m->n, c->T,
           (unsigned long long)c->seed, elapsed, extra ? extra : "");
    if (c->mode == MODE_VAR) {
        double var95, cvar95, var99, cvar99;
        mc_var_cvar(loss, N, 0.95, &var95, &cvar95);
        mc_var_cvar(loss, N, 0.99, &var99, &cvar99);
        printf(",\"value\":%.10g,\"mean_loss\":%.10g,\"VaR95\":%.10g,\"CVaR95\":%.10g,"
               "\"VaR99\":%.10g,\"CVaR99\":%.10g",
               c->value, a->sum[0] / N, var95, cvar95, var99, cvar99);
    } else {
        printf(",\"r\":%.10g,\"K_rel\":%.10g,\"options\":[", c->r, c->K_rel);
        for (int i = 0; i <= m->n; i++) {
            double mean = a->sum[i] / N;
            double se = sqrt(fmax(a->sumsq[i] / N - mean * mean, 0.0) / (N - 1));
            if (i < m->n) {
                double bs = bs_call(m->S0[i], c->K_rel * m->S0[i], c->r, m->sigma[i], c->T);
                printf("{\"ticker\":\"%s\",\"mc\":%.10g,\"se\":%.10g,\"bs\":%.10g},",
                       m->ticker[i], mean, se, bs);
            } else {
                printf("{\"ticker\":\"BASKET\",\"mc\":%.10g,\"se\":%.10g}", mean, se);
            }
        }
        printf("]");
    }
    printf("}\n");
}

#endif
