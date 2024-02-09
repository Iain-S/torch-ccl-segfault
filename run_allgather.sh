#!/usr/bin/env bash

# e.g.
# mpirun -n 2 ./run_allgather.sh ccl 10 xpu
# mpirun -n 2 ./run_allgather.sh gloo 100 cpu

# "mpi" also works but prints some killed-9 error at the end.
export CCL_ATL_TRANSPORT=ofi

# Pass all arguments to allgather module.
python allgather.py "${@:1}"

