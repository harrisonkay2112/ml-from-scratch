r"""Download the raw MNIST idx files.

Source: https://ossci-datasets.s3.amazonaws.com/mnist/  (the mirror torchvision uses;
the original yann.lecun.com host now blocks most automated requests).
Total download: ~11 MB. Files land in data/ and are gitignored.

Run:  .venv\Scripts\python.exe scripts\get_mnist.py
"""

import gzip
import urllib.request
from pathlib import Path

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        out = DATA_DIR / name.removesuffix(".gz")
        if out.exists():
            print(f"have   {out.name}")
            continue
        print(f"fetch  {name} ...", end=" ", flush=True)
        with urllib.request.urlopen(BASE + name) as resp:
            blob = gzip.decompress(resp.read())
        out.write_bytes(blob)
        print(f"{len(blob) / 1e6:.1f} MB -> {out.name}")


if __name__ == "__main__":
    main()
