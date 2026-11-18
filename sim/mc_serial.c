/* Serial baseline of the Monte Carlo risk engine. */
#include "mc_common.h"

int main(int argc, char **argv) {
    mc_config c;
    mc_market m;
    mc_acc acc;
    if (mc_parse(argc, argv, &c) || mc_load_market(c.params, &m) || mc_acc_init(&acc, m.n)) return 1;

    double *loss = (c.mode == MODE_VAR) ? malloc(sizeof(double) * c.paths) : NULL;
    if (c.mode == MODE_VAR && !loss) { perror("malloc"); return 1; }

    double t0 = mc_wtime();
    for (long p = 0; p < c.paths; p++) {
        double l = mc_eval_path(&m, &c, p, &acc);
        if (loss) loss[p] = l;
    }
    double elapsed = mc_wtime() - t0;

    mc_report("serial", 1, &m, &c, &acc, loss, elapsed, NULL);
    free(loss);
    mc_acc_free(&acc);
    mc_market_free(&m);
    return 0;
}
