"""Parse the raw MNIST idx files into NumPy arrays.

The idx format is trivially simple, which is why we read it directly instead of
pulling in torchvision:

    [0:4]   magic number, big-endian int32.
            Low byte = element type (0x08 = unsigned byte).
            High byte of the low half = number of dimensions (3 for images, 1 for labels).
    [4:...] one big-endian int32 per dimension (e.g. 60000, 28, 28)
    [...]   the raw bytes, C-ordered

Everything is big-endian, which is why the dtype below is '>u4' and not just 'uint32'.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_idx(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    # Byte 3 of the magic number tells us how many dimensions follow.
    ndim = blob[3]
    # Read `ndim` big-endian uint32s starting right after the 4-byte magic number.
    shape = tuple(np.frombuffer(blob, dtype=">u4", count=ndim, offset=4).astype(int))
    payload = np.frombuffer(blob, dtype=np.uint8, offset=4 + 4 * ndim)
    return payload.reshape(shape)


def load(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for 'train' or 'test'.

    images: float64, shape (N, 784), scaled to [0, 1]
    labels: uint8,   shape (N,),     values 0-9

    Note the flattening to 784: phase 1 is a plain fully-connected net, so each
    image is one long vector. Phase 3 (the CNN) will want the 28x28 shape back.
    """
    prefix = {"train": "train", "test": "t10k"}[split]
    images = _read_idx(DATA_DIR / f"{prefix}-images-idx3-ubyte")
    labels = _read_idx(DATA_DIR / f"{prefix}-labels-idx1-ubyte")
    # uint8 0-255 -> float 0.0-1.0. Networks train badly on raw 0-255 inputs;
    # ask yourself why before phase 2.
    images = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0
    return images, labels


if __name__ == "__main__":
    for split in ("train", "test"):
        X, y = load(split)
        print(f"{split:5} X={X.shape} {X.dtype} min={X.min()} max={X.max()}  y={y.shape} {y.dtype}")
    X, y = load("train")
    print("\nfirst training label:", y[0])
    # Crude ASCII render so you can eyeball that the parse is correct.
    for row in X[0].reshape(28, 28):
        print("".join(" .:-=+*#%@"[int(v * 9)] for v in row))
