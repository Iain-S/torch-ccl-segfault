# torch-ccl-segfault

[MRE](https://stackoverflow.com/help/minimal-reproducible-example) for [CCL bindings for PyTorch](https://github.com/intel/torch-ccl/tree/master) crash.

## Steps

1. Install PyTorch, the Intel Extensions and the CCL Bindings with
   ```shell
   python -m pip install \
   torch==2.1.0a0 \
   torchvision==0.16.0a0 \
   torchaudio==2.1.0a0 \
   intel-extension-for-pytorch==2.1.10+xpu \
   oneccl-bind-pt==2.1.100+xpu \
   --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
   ```
1. Use MPI and the `run_allgather.sh` script to launch `allgather.py`: `mpiexec.hydra -n 2 run_allgather.sh ccl 2_000_000 xpu`

Note that `mpiexec.hydra -n 2 run_allgather.sh ccl 2_000_000 xpu` works as expected but `mpiexec.hydra -n 2 run_allgather.sh ccl 3_000_000 xpu` does not work.
Running on the CPU with `mpiexec.hydra -n 2 run_allgather.sh ccl 3_000_000 cpu` does work.
