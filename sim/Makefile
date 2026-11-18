CC      ?= gcc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
LDLIBS  = -lm
BIN     = ../build

all: $(BIN)/mc_serial $(BIN)/mc_omp $(BIN)/mc_mpi

$(BIN)/mc_serial: mc_serial.c mc_common.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BIN)/mc_omp: mc_omp.c mc_common.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

# Requires an MPI toolchain (e.g. sudo apt install libopenmpi-dev openmpi-bin).
$(BIN)/mc_mpi: mc_mpi.c mc_common.h | $(BIN)
	mpicc $(CFLAGS) -o $@ $< $(LDLIBS)

$(BIN):
	mkdir -p $@

clean:
	rm -rf $(BIN)

.PHONY: all clean
