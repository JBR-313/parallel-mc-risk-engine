/* MPI version.
 *
 *  1. Rank 0 reads the parameters and factorises the covariance; n, S0, mu, cov
 *     and L are sent to all ranks with MPI_Bcast.
 *  2. Paths are block-partitioned: rank r simulates [lo_r, hi_r). Path p still
 *     uses RNG stream p, so the union over ranks is exactly the serial path set.
 *  3. Accumulators are combined with one MPI_Reduce; for VaR the per-path losses
 *     are collected on rank 0 with MPI_Gatherv, because a quantile needs the whole
 *     distribution (it cannot be reduced like a sum).
 *
 * Reported times: elapsed = total wall time on rank 0 (compute + communication),
 * compute_max / compute_min = slowest / fastest rank's simulation time (load
 * balance), comm = elapsed - compute_max.
 */
#include <mpi.h>
#include "mc_common.h"

static void bcast_market(mc_market *m, int rank) {
    int n = (rank == 0) ? m->n : 0;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0 && mc_market_alloc(m, n)) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Bcast(m->ticker, 16 * n, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(m->S0, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m->mu, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m->sigma, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m->cov, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m->L, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mc_config c;
    mc_market m;
    if (mc_parse(argc, argv, &c)) MPI_Abort(MPI_COMM_WORLD, 1);
    int ok = 1;
    if (rank == 0) ok = (mc_load_market(c.params, &m) == 0);
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) { MPI_Finalize(); return 1; }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    bcast_market(&m, rank);

    long lo = c.paths * rank / size, hi = c.paths * (rank + 1) / size, cnt = hi - lo;
    mc_acc acc = {0}, total = {0};
    if (mc_acc_init(&acc, m.n) || mc_acc_init(&total, m.n)) MPI_Abort(MPI_COMM_WORLD, 1);
    double *loss = (c.mode == MODE_VAR) ? malloc(sizeof(double) * (cnt > 0 ? cnt : 1)) : NULL;
    if (c.mode == MODE_VAR && !loss) MPI_Abort(MPI_COMM_WORLD, 1);

    double tc = MPI_Wtime();
    for (long p = lo; p < hi; p++) {
        double l = mc_eval_path(&m, &c, p, &acc);
        if (loss) loss[p - lo] = l;
    }
    double compute = MPI_Wtime() - tc;

    MPI_Reduce(acc.sum, total.sum, 2 * acc.len, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double *all = NULL;
    if (c.mode == MODE_VAR) {
        int *counts = NULL, *displs = NULL;
        if (rank == 0) {
            all = malloc(sizeof(double) * c.paths);
            counts = malloc(sizeof(int) * size);
            displs = malloc(sizeof(int) * size);
            if (!all || !counts || !displs) MPI_Abort(MPI_COMM_WORLD, 1);
            for (int r = 0; r < size; r++) {
                long a = c.paths * r / size, b = c.paths * (r + 1) / size;
                counts[r] = (int)(b - a);
                displs[r] = (int)a;
            }
        }
        MPI_Gatherv(loss, (int)cnt, MPI_DOUBLE, all, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(counts);
        free(displs);
    }
    double elapsed = MPI_Wtime() - t0;

    double cmax, cmin;
    MPI_Reduce(&compute, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute, &cmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char extra[160];
        snprintf(extra, sizeof extra, ",\"compute_max\":%.6f,\"compute_min\":%.6f,\"comm\":%.6f",
                 cmax, cmin, elapsed - cmax);
        mc_report("mpi", size, &m, &c, &total, all, elapsed, extra);
    }
    free(all);
    free(loss);
    mc_acc_free(&acc);
    mc_acc_free(&total);
    mc_market_free(&m);
    MPI_Finalize();
    return 0;
}
