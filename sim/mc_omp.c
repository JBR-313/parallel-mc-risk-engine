/* OpenMP version: paths are distributed over threads; each thread keeps a
 * private accumulator that is merged once at the end (no false sharing on the
 * hot loop, no atomics). Path p uses RNG stream p, so results match the serial
 * code up to floating-point summation order, and VaR/CVaR match exactly. */
#include <omp.h>
#include "mc_common.h"

int main(int argc, char **argv) {
    mc_config c;
    mc_market m;
    mc_acc acc;
    if (mc_parse(argc, argv, &c) || mc_load_market(c.params, &m) || mc_acc_init(&acc, m.n)) return 1;

    double *loss = (c.mode == MODE_VAR) ? malloc(sizeof(double) * c.paths) : NULL;
    if (c.mode == MODE_VAR && !loss) { perror("malloc"); return 1; }

    int nthreads = 1;
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        mc_acc local;
        mc_acc_init(&local, m.n);
        #pragma omp single
        nthreads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (long p = 0; p < c.paths; p++) {
            double l = mc_eval_path(&m, &c, p, &local);
            if (loss) loss[p] = l;
        }
        #pragma omp critical
        mc_acc_merge(&acc, &local);
        mc_acc_free(&local);
    }
    double elapsed = omp_get_wtime() - t0;

    mc_report("openmp", nthreads, &m, &c, &acc, loss, elapsed, NULL);
    free(loss);
    mc_acc_free(&acc);
    mc_market_free(&m);
    return 0;
}
