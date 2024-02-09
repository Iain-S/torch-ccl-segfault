#!/usr/bin/env bash

# e.g.
# mpirun -n 2 ./run_allgather.sh ccl 10 xpu
# mpirun -n 2 ./run_allgather.sh gloo 100 cpu

ulimit -n 1000000

export CCL_WORKER_OFFLOAD=0
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=verbs

# Pass all arguments to allgather module.
python allgather.py "${@:1}"

