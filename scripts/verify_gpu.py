import sys

import torch


def main() -> int:
    print("Python executable:", sys.executable)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda runtime:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        print("CUDA is not available in this environment.")
        return 1

    print("device name:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU tensor test: passed")
    print("result shape:", tuple(y.shape))
    print("result device:", y.device)
    print("allocated memory MB:", round(torch.cuda.memory_allocated(0) / 1024**2, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
