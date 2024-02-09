import argparse
import os

import torch
import intel_extension_for_pytorch as ipex  # has side-effects
import oneccl_bindings_for_pytorch  # has side-effects

from torch.distributed import init_process_group
import torch.distributed as dist

# PMI_RANK set by mpirun
RANK = os.environ["PMI_RANK"]
os.environ["RANK"] = RANK

# PMI_SIZE set by mpirun
WORLD_SIZE = os.environ["PMI_SIZE"]
os.environ["WORLD_SIZE"] = WORLD_SIZE

os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "29876"


def main(backend, shape, device):
    print("Initialising process group with backend", backend, flush=True)
    init_process_group(
        backend=backend,
    )

    TSHAPE = [shape]
    print(f"{TSHAPE=}", flush=True)

    tensor = torch.rand(TSHAPE)

    if device == "xpu":
        # Explicitly send tensors to different XPUs
        DEVICE = "xpu:" + str(RANK)
    print("moving to", DEVICE, flush=True)
    tensor = tensor.to(DEVICE)

    # First method of size calculation
    tensor_size = 1
    for dim in tensor.size():
        tensor_size = tensor_size * dim

    size_all_mb = tensor_size / 1024**2
    print("tensor size 2: {:.3f}MB".format(size_all_mb), flush=True)

    # Second method of size calculation
    tensor_size = tensor.element_size() * tensor.nelement()
    size_all_mb = tensor_size / 1024**2
    print("tensor size 1: {:.3f}MB".format(size_all_mb), flush=True)

    # A list into which we can receive data from other ranks
    print("making space", flush=True)
    tensor_list = [
        torch.zeros(TSHAPE, dtype=torch.float32).to(DEVICE)
        for _ in range(int(WORLD_SIZE))
    ]

    print("gathering", flush=True)
    dist.all_gather(tensor_list, tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="a description")
    parser.add_argument(
        "backend", choices=("ccl", "gloo"), help="The type of process group backend."
    )
    parser.add_argument("shape", type=int, help="Make a 1-D Tensor of shape [shape].")
    parser.add_argument(
        "device", choices=("cpu", "xpu"), help="The device to load the tensor(s) to."
    )
    args = parser.parse_args()
    main(args.backend, args.shape, args.device)
