import argparse
import pickle
import io
from pickle import Unpickler
from pathlib import Path

import torch

VER_MAPPING = {
    "mmdet3d.core.bbox.structures.box_3d_mode": "mmdet3d.structures",
    "mmdet3d.core.bbox.structures.lidar_box3d": "mmdet3d.structures.bbox_3d.lidar_box3d",
}


class BEVFormerUnpickler(Unpickler):
    def __init__(self, file, verbose=False):
        super(BEVFormerUnpickler, self).__init__(file)
        self.verbose = verbose

    def find_class(self, module, name):
        if self.verbose:
            print(module, name)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        if module in VER_MAPPING:
            return super().find_class(VER_MAPPING[module], name)
        else:
            return super().find_class(module, name)


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert old mmcv pickles into new ones"
    )
    parser.add_argument("--filename", help="Pickle file path")
    parser.add_argument("--output_path", default=None, help="Pickle output filepath")
    parser.add_argument(
        "-v", action="store_true", dest="verb", help="Pickle output filepath"
    )

    return parser.parse_args()


def main():
    args = get_args()
    filename = Path(args.filename)
    try:
        output_path = Path(args.output_path)
    except TypeError:
        output_path = args.output_path

    if filename.is_dir():
        assert output_path is None or output_path.is_dir()
        if output_path is None:
            output_path = filename

        for file in filename.glob("*.pkl"):
            with open(file, "rb") as f:
                unpickler = BEVFormerUnpickler(f, args.verb)
                data = unpickler.load()
            with open(output_path / file.name, "wb") as f:
                pickle.dump(data, f)

    else:
        with open(args.filename, "rb") as f:
            unpickler = BEVFormerUnpickler(f, args.verb)
            data = unpickler
        with open(output_path, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
